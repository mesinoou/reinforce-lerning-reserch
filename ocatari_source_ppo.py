#!/usr/bin/env python3
"""Train and diagnose a source-only object-centric PPO agent on OCAtari REM.

This program deliberately excludes transfer learning. Its only purpose is to
establish that PPO can learn Space Invaders from the common object-centric
representation before source knowledge is used by a target task.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep Matplotlib's cache inside the project on restricted workstations. Linux
# users can override this in the shell before starting the program.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ocatari_transfer_full_experiment import (
    ActorCriticMLP,
    EncoderConfig,
    ObjectEnv,
    ensure_dir,
    module_version,
    moving_average,
    save_csv,
    save_json,
)


EPISODE_FIELDS = [
    "episode",
    "env_steps",
    "ale_frames",
    "raw_return",
    "training_return",
    "score_component_return",
    "survival_component_return",
    "life_loss_component_return",
    "episode_length",
    "life_losses",
    "elapsed_seconds",
]
UPDATE_FIELDS = [
    "update",
    "env_steps",
    "ale_frames",
    "learning_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "gradient_norm",
    "explained_variance",
    "value_mean",
    "value_std",
    "advantage_mean",
    "advantage_std",
    "return_mean",
    "raw_reward_mean",
    "raw_reward_std",
    "training_reward_mean",
    "training_reward_std",
    "score_component_mean",
    "survival_component_mean",
    "life_loss_component_mean",
    "ale_frames_advanced_mean",
    "epochs_completed",
]
EVALUATION_FIELDS = [
    "stage",
    "env_steps",
    "policy",
    "episodes",
    "raw_return_mean",
    "raw_return_std",
    "raw_return_min",
    "raw_return_max",
    "episode_length_mean",
]


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    anneal_learning_rate: bool = True
    clip_epsilon: float = 0.1
    value_clip_epsilon: float = 0.1
    value_clipping: bool = True
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 1024
    hidden_size: int = 256
    target_kl: Optional[float] = 0.03


@dataclass
class RewardConfig:
    mode: str = "scaled_raw"
    reward_scale: float = 0.1
    survival_reward_per_frame: float = 0.001
    life_loss_penalty: float = -1.0
    # Legacy shaped mode parameters retained for exact reproduction.
    survival_bonus: float = 0.5
    death_penalty: float = -5.0


@dataclass(frozen=True)
class RewardComponents:
    score: float
    survival: float
    life_loss: float

    @property
    def total(self) -> float:
        return self.score + self.survival + self.life_loss


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.raw_rewards = np.zeros(capacity, dtype=np.float32)
        self.score_components = np.zeros(capacity, dtype=np.float32)
        self.survival_components = np.zeros(capacity, dtype=np.float32)
        self.life_loss_components = np.zeros(capacity, dtype=np.float32)
        self.ale_frames_advanced = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        raw_reward: float = 0.0,
        score_component: float = 0.0,
        survival_component: float = 0.0,
        life_loss_component: float = 0.0,
        ale_frames_advanced: int = 1,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("Rollout buffer overflow")
        index = self.ptr
        self.obs[index] = obs
        self.actions[index] = action
        self.rewards[index] = reward
        self.raw_rewards[index] = raw_reward
        self.score_components[index] = score_component
        self.survival_components[index] = survival_component
        self.life_loss_components[index] = life_loss_component
        self.ale_frames_advanced[index] = ale_frames_advanced
        self.dones[index] = float(done)
        self.log_probs[index] = log_prob
        self.values[index] = value
        self.ptr += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        for step in reversed(range(self.ptr)):
            nonterminal = 1.0 - self.dones[step]
            next_value = last_value if step == self.ptr - 1 else self.values[step + 1]
            delta = (
                self.rewards[step]
                + gamma * next_value * nonterminal
                - self.values[step]
            )
            last_gae = (
                delta
                + gamma * gae_lambda * nonterminal * last_gae
            )
            advantages[step] = last_gae
        return advantages, advantages + self.values[: self.ptr]


class ObservationStatistics:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)
        self.nonzero_count = np.zeros(dimension, dtype=np.int64)
        self.identical_stack_count = 0

    def update(self, observation: np.ndarray, stack_size: int) -> None:
        value = np.asarray(observation, dtype=np.float64)
        if value.shape != (self.dimension,):
            raise ValueError(
                f"Observation shape changed: {value.shape}, expected {(self.dimension,)}"
            )
        if not np.isfinite(value).all():
            raise FloatingPointError("NaN or Inf detected in an encoded observation")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = np.minimum(self.minimum, value)
        self.maximum = np.maximum(self.maximum, value)
        self.nonzero_count += value != 0.0
        frames = value.reshape(stack_size, -1)
        if np.allclose(frames, frames[0], rtol=0.0, atol=1e-7):
            self.identical_stack_count += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "nonzero_count": self.nonzero_count,
            "identical_stack_count": self.identical_stack_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state["dimension"]) != self.dimension:
            raise ValueError("Observation-statistics dimension mismatch")
        self.count = int(state["count"])
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.m2 = np.asarray(state["m2"], dtype=np.float64)
        self.minimum = np.asarray(state["minimum"], dtype=np.float64)
        self.maximum = np.asarray(state["maximum"], dtype=np.float64)
        self.nonzero_count = np.asarray(state["nonzero_count"], dtype=np.int64)
        self.identical_stack_count = int(state["identical_stack_count"])

    def rows(self, names: Sequence[str]) -> List[Dict[str, Any]]:
        if self.count == 0:
            return []
        variance = self.m2 / max(1, self.count)
        return [
            {
                "dimension": index,
                "name": names[index],
                "mean": float(self.mean[index]),
                "std": float(math.sqrt(max(0.0, variance[index]))),
                "min": float(self.minimum[index]),
                "max": float(self.maximum[index]),
                "nonzero_rate": float(self.nonzero_count[index] / self.count),
            }
            for index in range(self.dimension)
        ]


class ObjectDiagnostics:
    def __init__(self) -> None:
        self.frames = 0
        self.category_instances: Counter[str] = Counter()
        self.category_frames: Counter[str] = Counter()
        self.player_frames = 0
        self.enemy_frames = 0
        self.projectile_frames = 0

    def update(self, env: ObjectEnv) -> None:
        categories: List[str] = []
        has_player = False
        has_enemy = False
        has_projectile = False
        for obj in env.objects:
            try:
                if not bool(obj):
                    continue
            except Exception:
                pass
            category = env.encoder._category(obj)
            categories.append(category)
            has_player = has_player or category in env.encoder.PLAYER_NAMES
            has_enemy = has_enemy or env.encoder._is_enemy(category)
            has_projectile = (
                has_projectile or env.encoder._is_projectile(category)
            )
        self.frames += 1
        self.category_instances.update(categories)
        self.category_frames.update(set(categories))
        self.player_frames += int(has_player)
        self.enemy_frames += int(has_enemy)
        self.projectile_frames += int(has_projectile)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "frames": self.frames,
            "category_instances": dict(self.category_instances),
            "category_frames": dict(self.category_frames),
            "player_frames": self.player_frames,
            "enemy_frames": self.enemy_frames,
            "projectile_frames": self.projectile_frames,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.frames = int(state["frames"])
        self.category_instances = Counter(state["category_instances"])
        self.category_frames = Counter(state["category_frames"])
        self.player_frames = int(state["player_frames"])
        self.enemy_frames = int(state["enemy_frames"])
        self.projectile_frames = int(state["projectile_frames"])

    def category_rows(self) -> List[Dict[str, Any]]:
        denominator = max(1, self.frames)
        return [
            {
                "category": category,
                "total_instances": self.category_instances[category],
                "frames_present": self.category_frames[category],
                "frame_presence_rate": self.category_frames[category] / denominator,
            }
            for category in sorted(self.category_instances)
        ]

    def summary(self) -> Dict[str, Any]:
        denominator = max(1, self.frames)
        return {
            "observed_frames": self.frames,
            "player_detection_rate": self.player_frames / denominator,
            "enemy_detection_rate": self.enemy_frames / denominator,
            "projectile_detection_rate": self.projectile_frames / denominator,
        }


def set_global_seed(seed: int, deterministic_torch: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(deterministic_torch, warn_only=True)
    except Exception:
        pass


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def reward_components(
    raw_reward: float,
    life_lost: bool,
    config: RewardConfig,
    ale_frames_advanced: int = 1,
) -> RewardComponents:
    if ale_frames_advanced < 0:
        raise ValueError("ale_frames_advanced must be non-negative")
    if config.mode == "raw":
        return RewardComponents(float(raw_reward), 0.0, 0.0)
    if config.mode == "clipped":
        return RewardComponents(
            float(np.clip(raw_reward, -1.0, 1.0)),
            0.0,
            0.0,
        )
    if config.mode == "scaled_raw":
        return RewardComponents(
            float(raw_reward) * config.reward_scale,
            0.0,
            0.0,
        )
    if config.mode == "scaled_survival":
        return RewardComponents(
            score=float(raw_reward) * config.reward_scale,
            survival=(
                config.survival_reward_per_frame * ale_frames_advanced
            ),
            life_loss=config.life_loss_penalty if life_lost else 0.0,
        )
    if config.mode == "shaped":
        # Exact legacy behavior: one bonus per environment step, then scale
        # every component together. This is intentionally distinct from the
        # ALE-frame-based scaled_survival mode.
        return RewardComponents(
            score=float(raw_reward) * config.reward_scale,
            survival=config.survival_bonus * config.reward_scale,
            life_loss=(
                config.death_penalty * config.reward_scale
                if life_lost
                else 0.0
            ),
        )
    raise ValueError(f"Unknown reward mode: {config.mode}")


def transform_reward(
    raw_reward: float,
    life_lost: bool,
    config: RewardConfig,
    ale_frames_advanced: int = 1,
) -> float:
    """Return only the total for callers that do not need diagnostics."""
    return reward_components(
        raw_reward,
        life_lost,
        config,
        ale_frames_advanced,
    ).total


def count_ale_frames_advanced(
    previous_info: Dict[str, Any],
    next_info: Dict[str, Any],
    fallback_frameskip: int,
) -> int:
    """Measure emulator frames advanced by one ObjectEnv step.

    ALE's global frame counter is preferred. The episode-local counter is a
    fallback, followed by the configured frameskip for older OCAtari versions.
    """
    for key in ("frame_number", "episode_frame_number"):
        before = previous_info.get(key)
        after = next_info.get(key)
        if before is None or after is None:
            continue
        difference = int(after) - int(before)
        if difference > 0:
            return difference
    return int(fallback_frameskip)


def explained_variance(prediction: np.ndarray, target: np.ndarray) -> float:
    target_variance = float(np.var(target))
    if target_variance < 1e-12:
        return float("nan")
    return float(1.0 - np.var(target - prediction) / target_variance)


def ppo_update(
    model: ActorCriticMLP,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> Dict[str, float]:
    normalized_advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = obs.shape[0]
    indices = np.arange(sample_count)
    metrics: Dict[str, List[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "gradient_norm": [],
    }
    epochs_completed = 0

    for _epoch in range(config.ppo_epochs):
        np.random.shuffle(indices)
        epoch_kls: List[float] = []
        for start in range(0, sample_count, config.minibatch_size):
            mb = indices[start : start + config.minibatch_size]
            logits, values = model(obs[mb])
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[mb])
            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = log_ratio.exp()

            surrogate1 = ratio * normalized_advantages[mb]
            surrogate2 = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * normalized_advantages[mb]
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            if config.value_clipping:
                unclipped_value_loss = (values - returns[mb]).pow(2)
                clipped_values = old_values[mb] + torch.clamp(
                    values - old_values[mb],
                    -config.value_clip_epsilon,
                    config.value_clip_epsilon,
                )
                clipped_value_loss = (clipped_values - returns[mb]).pow(2)
                value_loss = 0.5 * torch.max(
                    unclipped_value_loss,
                    clipped_value_loss,
                ).mean()
            else:
                value_loss = 0.5 * (values - returns[mb]).pow(2).mean()

            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "NaN or Inf detected in PPO loss; inspect updates.csv"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("NaN or Inf detected in gradient norm")
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_epsilon
                ).float().mean()
            batch_values = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "gradient_norm": gradient_norm,
            }
            for name, value in batch_values.items():
                scalar = float(value.detach().cpu())
                metrics[name].append(scalar)
            epoch_kls.append(float(approx_kl.detach().cpu()))

        epochs_completed += 1
        if (
            config.target_kl is not None
            and epoch_kls
            and float(np.mean(epoch_kls)) > config.target_kl
        ):
            break

    result = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in metrics.items()
    }
    result["epochs_completed"] = float(epochs_completed)
    return result


def feature_names(config: EncoderConfig) -> List[str]:
    frame_names = [
        "player.present",
        "player.x",
        "player.y",
        "player.width",
        "player.height",
    ]
    for index in range(config.max_enemies):
        frame_names.extend(
            [
                f"enemy[{index}].present",
                f"enemy[{index}].relative_x",
                f"enemy[{index}].relative_y",
                f"enemy[{index}].width",
                f"enemy[{index}].height",
            ]
        )
    for index in range(config.max_projectiles):
        frame_names.extend(
            [
                f"projectile[{index}].present",
                f"projectile[{index}].relative_x",
                f"projectile[{index}].relative_y",
                f"projectile[{index}].width",
                f"projectile[{index}].height",
            ]
        )
    frame_names.extend(
        [
            "global.enemy_count",
            "global.projectile_count",
            "global.enemy_centroid_relative_x",
            "global.enemy_centroid_relative_y",
            "global.nearest_enemy_distance",
            "global.nearest_projectile_distance",
        ]
    )
    return [
        f"frame[{frame_index}].{name}"
        for frame_index in range(config.stack_size)
        for name in frame_names
    ]


def visible_object_sample(env: ObjectEnv) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    for obj in env.objects:
        try:
            if not bool(obj):
                continue
        except Exception:
            pass
        try:
            x, y, width, height = env.encoder._xywh(obj)
            objects.append(
                {
                    "category": str(
                        getattr(obj, "category", obj.__class__.__name__)
                    ),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )
        except Exception:
            continue
    return objects


def maybe_fire_after_reset(
    env: ObjectEnv,
    obs: np.ndarray,
    info: Dict[str, Any],
    enabled: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not enabled:
        return obs, info
    meanings = env.action_meanings()
    if "FIRE" not in meanings:
        return obs, info
    fire_action = meanings.index("FIRE")
    next_obs, _reward, done, next_info = env.step(fire_action)
    if done:
        return env.reset()
    return next_obs, next_info


@torch.no_grad()
def evaluate(
    *,
    model: Optional[ActorCriticMLP],
    env_id: str,
    encoder_config: EncoderConfig,
    object_mode: str,
    frameskip: int,
    repeat_action_probability: float,
    episodes: int,
    seed: int,
    device: torch.device,
    policy: str,
    max_episode_steps: int,
    auto_fire_reset: bool,
) -> Dict[str, float]:
    if policy not in {"random", "stochastic", "deterministic"}:
        raise ValueError(f"Unknown evaluation policy: {policy}")
    rng_state = capture_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    was_training = bool(model.training) if model is not None else False
    if model is not None:
        model.eval()
    env = ObjectEnv(
        env_id,
        encoder_config,
        object_mode,
        frameskip,
        repeat_action_probability,
    )
    random_generator = np.random.default_rng(seed)
    returns: List[float] = []
    lengths: List[int] = []
    try:
        for episode_index in range(episodes):
            obs, info = env.reset(seed=seed + episode_index)
            obs, info = maybe_fire_after_reset(
                env,
                obs,
                info,
                auto_fire_reset,
            )
            raw_return = 0.0
            length = 0
            done = False
            while not done and length < max_episode_steps:
                if policy == "random":
                    action = int(random_generator.integers(env.action_space.n))
                else:
                    if model is None:
                        raise ValueError("A model is required for policy evaluation")
                    observation_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                    logits, _value = model(observation_tensor)
                    if policy == "deterministic":
                        action = int(torch.argmax(logits, dim=-1).item())
                    else:
                        action = int(
                            torch.distributions.Categorical(logits=logits)
                            .sample()
                            .item()
                        )
                obs, reward, done, _info = env.step(action)
                raw_return += reward
                length += 1
            returns.append(raw_return)
            lengths.append(length)
    finally:
        env.close()
        if model is not None and was_training:
            model.train()
        restore_rng_state(rng_state)

    return_values = np.asarray(returns, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    return {
        "episodes": float(episodes),
        "raw_return_mean": float(return_values.mean()),
        "raw_return_std": float(return_values.std(ddof=0)),
        "raw_return_min": float(return_values.min()),
        "raw_return_max": float(return_values.max()),
        "episode_length_mean": float(length_values.mean()),
    }


def append_evaluation_rows(
    rows: List[Dict[str, Any]],
    *,
    stage: str,
    env_steps: int,
    model: ActorCriticMLP,
    args: argparse.Namespace,
    encoder_config: EncoderConfig,
    device: torch.device,
) -> List[Dict[str, Any]]:
    new_rows: List[Dict[str, Any]] = []
    for policy in ("deterministic", "stochastic"):
        result = evaluate(
            model=model,
            env_id=args.env,
            encoder_config=encoder_config,
            object_mode=args.object_mode,
            frameskip=args.frameskip,
            repeat_action_probability=args.repeat_action_probability,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            device=device,
            policy=policy,
            max_episode_steps=args.max_episode_steps,
            auto_fire_reset=args.auto_fire_reset,
        )
        row = {
            "stage": stage,
            "env_steps": env_steps,
            "policy": policy,
            **result,
        }
        rows.append(row)
        new_rows.append(row)
        print(
            f"[evaluation:{stage}:{policy}] steps={env_steps} "
            f"raw_return={result['raw_return_mean']:.2f}"
            f"±{result['raw_return_std']:.2f}"
        )
    return new_rows


def save_checkpoint(
    path: Path,
    *,
    model: ActorCriticMLP,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    encoder_config: EncoderConfig,
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    env_steps: int,
    update_index: int,
    episode_index: int,
    episode_rows: List[Dict[str, Any]],
    update_rows: List[Dict[str, Any]],
    evaluation_rows: List[Dict[str, Any]],
    action_rows: List[Dict[str, Any]],
    observation_statistics: ObservationStatistics,
    object_diagnostics: ObjectDiagnostics,
    best_evaluation_return: float,
    n_actions: int,
    action_meanings: List[str],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "environment_steps": env_steps,
            "update": update_index,
            "episode": episode_index,
            "best_evaluation_return": best_evaluation_return,
            "rng_state": capture_rng_state(),
            "configuration": vars(args),
            "encoder_config": asdict(encoder_config),
            "ppo_config": asdict(ppo_config),
            "reward_config": asdict(reward_config),
            "env_id": args.env,
            "obs_dim": encoder_config.stack_size
            * (
                5
                + encoder_config.max_enemies * 5
                + encoder_config.max_projectiles * 5
                + 6
            ),
            "n_actions": n_actions,
            "action_meanings": action_meanings,
            "episode_logs": episode_rows,
            "update_logs": update_rows,
            "evaluation_logs": evaluation_rows,
            "action_logs": action_rows,
            "observation_statistics": observation_statistics.state_dict(),
            "object_diagnostics": object_diagnostics.state_dict(),
        },
        path,
    )


def save_progress_files(
    output_dir: Path,
    episode_rows: List[Dict[str, Any]],
    update_rows: List[Dict[str, Any]],
    evaluation_rows: List[Dict[str, Any]],
    action_rows: List[Dict[str, Any]],
    observation_statistics: ObservationStatistics,
    object_diagnostics: ObjectDiagnostics,
    encoder_config: EncoderConfig,
) -> None:
    save_csv(output_dir / "episodes.csv", episode_rows, EPISODE_FIELDS)
    save_csv(output_dir / "updates.csv", update_rows, UPDATE_FIELDS)
    save_csv(
        output_dir / "periodic_evaluation.csv",
        evaluation_rows,
        EVALUATION_FIELDS,
    )
    action_fields = ["update", "env_steps"]
    if action_rows:
        action_fields.extend(
            sorted(key for key in action_rows[0] if key not in action_fields)
        )
    save_csv(output_dir / "action_distribution.csv", action_rows, action_fields)
    save_csv(
        output_dir / "object_categories.csv",
        object_diagnostics.category_rows(),
        [
            "category",
            "total_instances",
            "frames_present",
            "frame_presence_rate",
        ],
    )
    observation_rows = observation_statistics.rows(feature_names(encoder_config))
    save_csv(
        output_dir / "observation_statistics.csv",
        observation_rows,
        ["dimension", "name", "mean", "std", "min", "max", "nonzero_rate"],
    )
    summary = object_diagnostics.summary()
    summary.update(
        {
            "observation_count": observation_statistics.count,
            "identical_four_frame_stack_rate": (
                observation_statistics.identical_stack_count
                / max(1, observation_statistics.count)
            ),
            "constant_observation_dimensions": sum(
                1
                for row in observation_rows
                if abs(row["max"] - row["min"]) < 1e-12
            ),
            "always_zero_observation_dimensions": sum(
                1 for row in observation_rows if row["nonzero_rate"] == 0.0
            ),
        }
    )
    save_json(output_dir / "object_diagnostics.json", summary)


def plot_results(
    output_dir: Path,
    episode_rows: List[Dict[str, Any]],
    update_rows: List[Dict[str, Any]],
    evaluation_rows: List[Dict[str, Any]],
    moving_average_window: int,
) -> None:
    if episode_rows:
        steps = np.asarray(
            [row["env_steps"] for row in episode_rows],
            dtype=np.float64,
        )
        raw_returns = np.asarray(
            [row["raw_return"] for row in episode_rows],
            dtype=np.float64,
        )
        window = min(moving_average_window, len(raw_returns))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, raw_returns, alpha=0.20, label="Episode raw return")
        ax.plot(
            steps,
            moving_average(raw_returns, window),
            linewidth=2,
            label=f"Moving average ({window} episodes)",
        )
        ax.set_title("Space Invaders source PPO: training raw return")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Raw episode return")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "raw_return_curve.png", dpi=170)
        plt.close(fig)

        if "score_component_return" in episode_rows[0]:
            fig, ax = plt.subplots(figsize=(9, 5))
            component_series = (
                ("training_return", "Total training reward"),
                ("score_component_return", "Score component"),
                ("survival_component_return", "Survival component"),
                ("life_loss_component_return", "Life-loss component"),
            )
            for key, label in component_series:
                values = np.asarray(
                    [row.get(key, 0.0) for row in episode_rows],
                    dtype=np.float64,
                )
                ax.plot(
                    steps,
                    moving_average(values, window),
                    label=label,
                )
            ax.set_title("Episode training-reward components")
            ax.set_xlabel("Environment steps")
            ax.set_ylabel("Episode component return")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "reward_components_curve.png", dpi=170)
            plt.close(fig)

    reward_diagnostics = [
        row
        for row in update_rows
        if "raw_reward_std" in row and "training_reward_std" in row
    ]
    if reward_diagnostics:
        fig, ax = plt.subplots(figsize=(9, 5))
        update_steps = [
            row["env_steps"] for row in reward_diagnostics
        ]
        ax.plot(
            update_steps,
            [row["raw_reward_std"] for row in reward_diagnostics],
            label="Raw reward std",
        )
        ax.plot(
            update_steps,
            [row["training_reward_std"] for row in reward_diagnostics],
            label="Training reward std",
        )
        ax.set_title("Reward scaling diagnostic per rollout")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Reward standard deviation")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "reward_std_comparison.png", dpi=170)
        plt.close(fig)

    if evaluation_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for policy in ("deterministic", "stochastic"):
            selected = [
                row for row in evaluation_rows if row["policy"] == policy
            ]
            if not selected:
                continue
            ax.errorbar(
                [row["env_steps"] for row in selected],
                [row["raw_return_mean"] for row in selected],
                yerr=[row["raw_return_std"] for row in selected],
                marker="o",
                capsize=3,
                label=policy,
            )
        random_rows = [
            row for row in evaluation_rows if row["policy"] == "random"
        ]
        if random_rows:
            ax.axhline(
                random_rows[0]["raw_return_mean"],
                color="black",
                linestyle="--",
                label="random baseline",
            )
        ax.set_title("Fixed-seed periodic evaluation")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean raw return")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "periodic_evaluation.png", dpi=170)
        plt.close(fig)


def final_summary(
    *,
    args: argparse.Namespace,
    episode_rows: List[Dict[str, Any]],
    update_rows: List[Dict[str, Any]],
    evaluation_rows: List[Dict[str, Any]],
    starting_env_steps: int,
    env_steps: int,
    elapsed_seconds: float,
    object_diagnostics: ObjectDiagnostics,
) -> Dict[str, Any]:
    random_rows = [
        row for row in evaluation_rows if row["policy"] == "random"
    ]
    initial_stochastic = [
        row
        for row in evaluation_rows
        if row["stage"] == "initial" and row["policy"] == "stochastic"
    ]
    final_stochastic = [
        row
        for row in evaluation_rows
        if row["stage"] == "final" and row["policy"] == "stochastic"
    ]
    raw_values = np.asarray(
        [row["raw_return"] for row in episode_rows],
        dtype=np.float64,
    )
    final_fraction_start = int(0.8 * len(raw_values))
    summary: Dict[str, Any] = {
        "environment": args.env,
        "environment_steps": env_steps,
        "environment_steps_this_process": env_steps - starting_env_steps,
        "ale_frames": env_steps * args.frameskip,
        "episodes_completed": len(episode_rows),
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": (
            (env_steps - starting_env_steps) / max(elapsed_seconds, 1e-9)
        ),
        "reward_mode": args.reward_mode,
        "slot_strategy": args.slot_strategy,
        "object_diagnostics": object_diagnostics.summary(),
    }
    if raw_values.size:
        score_component_values = np.asarray(
            [
                row.get("score_component_return", 0.0)
                for row in episode_rows
            ],
            dtype=np.float64,
        )
        survival_component_values = np.asarray(
            [
                row.get("survival_component_return", 0.0)
                for row in episode_rows
            ],
            dtype=np.float64,
        )
        life_loss_component_values = np.asarray(
            [
                row.get("life_loss_component_return", 0.0)
                for row in episode_rows
            ],
            dtype=np.float64,
        )
        summary.update(
            {
                "training_raw_return_mean": float(raw_values.mean()),
                "training_raw_return_last_20_percent_mean": float(
                    raw_values[final_fraction_start:].mean()
                ),
                "training_raw_return_last_20_percent_std": float(
                    raw_values[final_fraction_start:].std(ddof=0)
                ),
                "episode_score_component_mean": float(
                    score_component_values.mean()
                ),
                "episode_survival_component_mean": float(
                    survival_component_values.mean()
                ),
                "episode_life_loss_component_mean": float(
                    life_loss_component_values.mean()
                ),
            }
        )
    reward_diagnostic_rows = [
        row
        for row in update_rows
        if "raw_reward_std" in row and "training_reward_std" in row
    ]
    if reward_diagnostic_rows:
        raw_reward_std_mean = float(
            np.mean(
                [row["raw_reward_std"] for row in reward_diagnostic_rows]
            )
        )
        training_reward_std_mean = float(
            np.mean(
                [
                    row["training_reward_std"]
                    for row in reward_diagnostic_rows
                ]
            )
        )
        summary.update(
            {
                "mean_rollout_raw_reward_std": raw_reward_std_mean,
                "mean_rollout_training_reward_std": (
                    training_reward_std_mean
                ),
                "training_to_raw_reward_std_ratio": (
                    training_reward_std_mean
                    / max(raw_reward_std_mean, 1e-12)
                ),
            }
        )
    if random_rows:
        summary["random_evaluation_raw_return_mean"] = random_rows[0][
            "raw_return_mean"
        ]
    if initial_stochastic:
        summary["initial_stochastic_raw_return_mean"] = initial_stochastic[0][
            "raw_return_mean"
        ]
    if final_stochastic:
        summary["final_stochastic_raw_return_mean"] = final_stochastic[-1][
            "raw_return_mean"
        ]
    if random_rows and final_stochastic:
        summary["final_minus_random_raw_return"] = (
            final_stochastic[-1]["raw_return_mean"]
            - random_rows[0]["raw_return_mean"]
        )
    if initial_stochastic and final_stochastic:
        summary["final_minus_initial_raw_return"] = (
            final_stochastic[-1]["raw_return_mean"]
            - initial_stochastic[0]["raw_return_mean"]
        )
    return summary


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable in PyTorch"
        )
    resume_checkpoint: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_checkpoint = torch.load(
            Path(args.resume),
            map_location=device,
            weights_only=False,
        )
        stored_arguments = resume_checkpoint.get("configuration", {})
        runtime_override_keys = {
            "total_steps",
            "device",
            "output_dir",
            "resume",
            "eval_interval",
            "eval_episodes",
            "random_eval_episodes",
            "eval_seed",
            "checkpoint_interval",
            "object_sample_frames",
            "moving_average_window",
            "log_interval_episodes",
        }
        for key, value in stored_arguments.items():
            if key not in runtime_override_keys and hasattr(args, key):
                setattr(args, key, value)

    output_dir = ensure_dir(
        Path(
            args.output_dir
            or (
                "source_results_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + f"_seed{args.seed}"
            )
        )
    )
    set_global_seed(args.seed, args.deterministic_torch)

    encoder_config = EncoderConfig(
        max_enemies=args.max_enemies,
        max_projectiles=args.max_projectiles,
        stack_size=args.stack_size,
        slot_strategy=args.slot_strategy,
        slot_match_distance=args.slot_match_distance,
    )
    ppo_config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        anneal_learning_rate=args.anneal_learning_rate,
        clip_epsilon=args.clip_epsilon,
        value_clip_epsilon=args.value_clip_epsilon,
        value_clipping=args.value_clipping,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        rollout_steps=args.rollout_steps,
        hidden_size=args.hidden_size,
        target_kl=args.target_kl if args.target_kl > 0.0 else None,
    )
    reward_config = RewardConfig(
        mode=args.reward_mode,
        reward_scale=args.reward_scale,
        survival_reward_per_frame=args.survival_reward_per_frame,
        life_loss_penalty=args.life_loss_penalty,
        survival_bonus=args.survival_bonus,
        death_penalty=args.death_penalty,
    )

    env = ObjectEnv(
        args.env,
        encoder_config,
        args.object_mode,
        args.frameskip,
        args.repeat_action_probability,
    )
    obs, info = env.reset(seed=args.seed)
    obs, info = maybe_fire_after_reset(
        env,
        obs,
        info,
        args.auto_fire_reset,
    )
    n_actions = int(env.action_space.n)
    action_meanings = env.action_meanings()
    model = ActorCriticMLP(
        env.obs_dim,
        n_actions,
        ppo_config.hidden_size,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=ppo_config.learning_rate,
        eps=1e-5,
    )
    buffer = RolloutBuffer(ppo_config.rollout_steps, env.obs_dim)
    observation_statistics = ObservationStatistics(env.obs_dim)
    object_diagnostics = ObjectDiagnostics()
    episode_rows: List[Dict[str, Any]] = []
    update_rows: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    object_samples: List[Dict[str, Any]] = []
    global_steps = 0
    update_index = 0
    episode_index = 0
    best_evaluation_return = -float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_checkpoint is None:
            raise RuntimeError("Resume checkpoint was not loaded")
        checkpoint = resume_checkpoint
        if checkpoint["env_id"] != args.env:
            raise ValueError(
                f"Resume environment mismatch: {checkpoint['env_id']} != {args.env}"
            )
        if int(checkpoint["obs_dim"]) != env.obs_dim:
            raise ValueError("Resume observation-dimension mismatch")
        if int(checkpoint["n_actions"]) != n_actions:
            raise ValueError("Resume action-space mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_steps = int(checkpoint["environment_steps"])
        update_index = int(checkpoint["update"])
        episode_index = int(checkpoint["episode"])
        best_evaluation_return = float(
            checkpoint.get("best_evaluation_return", -float("inf"))
        )
        episode_rows = list(checkpoint.get("episode_logs", []))
        update_rows = list(checkpoint.get("update_logs", []))
        evaluation_rows = list(checkpoint.get("evaluation_logs", []))
        action_rows = list(checkpoint.get("action_logs", []))
        if "observation_statistics" in checkpoint:
            observation_statistics.load_state_dict(
                checkpoint["observation_statistics"]
            )
        if "object_diagnostics" in checkpoint:
            object_diagnostics.load_state_dict(
                checkpoint["object_diagnostics"]
            )
        restore_rng_state(checkpoint["rng_state"])
        obs, info = env.reset(seed=args.seed + episode_index + 1)
        obs, info = maybe_fire_after_reset(
            env,
            obs,
            info,
            args.auto_fire_reset,
        )
        print(f"Resumed {resume_path} at environment step {global_steps}")

    starting_env_steps = global_steps

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "numpy": np.__version__,
        "ocatari": module_version("ocatari"),
        "gymnasium": module_version("gymnasium"),
        "ale-py": module_version("ale-py"),
    }
    save_json(
        output_dir / "config.json",
        {
            "arguments": vars(args),
            "encoder": asdict(encoder_config),
            "ppo": asdict(ppo_config),
            "reward": asdict(reward_config),
            "versions": versions,
            "observation_dimension": env.obs_dim,
            "action_count": n_actions,
            "action_meanings": action_meanings,
        },
    )
    try:
        env.save_detection_snapshot(
            output_dir / "object_detection_snapshot.png",
            f"Source PPO REM: {args.env}",
        )
    except Exception as exc:
        print(f"Warning: object snapshot could not be saved: {exc}")

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Device: {device}")
    print(
        f"Observation: {env.obs_dim} dimensions, "
        f"slot_strategy={args.slot_strategy}"
    )
    print(f"Actions: {n_actions} {action_meanings}")
    print(f"Training reward: {args.reward_mode}; raw return is logged separately")

    if not evaluation_rows:
        random_result = evaluate(
            model=None,
            env_id=args.env,
            encoder_config=encoder_config,
            object_mode=args.object_mode,
            frameskip=args.frameskip,
            repeat_action_probability=args.repeat_action_probability,
            episodes=args.random_eval_episodes,
            seed=args.eval_seed,
            device=device,
            policy="random",
            max_episode_steps=args.max_episode_steps,
            auto_fire_reset=args.auto_fire_reset,
        )
        evaluation_rows.append(
            {
                "stage": "random_baseline",
                "env_steps": 0,
                "policy": "random",
                **random_result,
            }
        )
        print(
            "[evaluation:random] "
            f"raw_return={random_result['raw_return_mean']:.2f}"
            f"±{random_result['raw_return_std']:.2f}"
        )
        append_evaluation_rows(
            evaluation_rows,
            stage="initial",
            env_steps=global_steps,
            model=model,
            args=args,
            encoder_config=encoder_config,
            device=device,
        )

    start_time = time.time()
    episode_raw_return = 0.0
    episode_training_return = 0.0
    episode_score_component_return = 0.0
    episode_survival_component_return = 0.0
    episode_life_loss_component_return = 0.0
    episode_length = 0
    episode_life_losses = 0
    current_info = info
    previous_lives_value = info.get("lives")
    previous_lives = (
        int(previous_lives_value)
        if previous_lives_value is not None
        else None
    )
    last_learning_terminal = False
    next_eval_step = (
        ((global_steps // args.eval_interval) + 1) * args.eval_interval
        if args.eval_interval > 0
        else math.inf
    )
    next_checkpoint_step = (
        ((global_steps // args.checkpoint_interval) + 1)
        * args.checkpoint_interval
        if args.checkpoint_interval > 0
        else math.inf
    )

    try:
        while global_steps < args.total_steps:
            buffer.reset()
            rollout_target = min(
                ppo_config.rollout_steps,
                args.total_steps - global_steps,
            )
            rollout_action_counts = np.zeros(n_actions, dtype=np.int64)

            for _ in range(rollout_target):
                observation_statistics.update(obs, encoder_config.stack_size)
                object_diagnostics.update(env)
                if len(object_samples) < args.object_sample_frames:
                    object_samples.append(
                        {
                            "env_step": global_steps,
                            "objects": visible_object_sample(env),
                            "observation_nonzero_count": int(
                                np.count_nonzero(obs)
                            ),
                            "encoded_observation": obs.tolist(),
                        }
                    )

                observation_tensor = (
                    torch.from_numpy(obs).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    logits, value = model(observation_tensor)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action_tensor = distribution.sample()
                    log_probability = distribution.log_prob(action_tensor)
                action = int(action_tensor.item())

                next_obs, raw_reward, environment_done, next_info = env.step(
                    action
                )
                ale_frames_advanced = count_ale_frames_advanced(
                    current_info,
                    next_info,
                    args.frameskip,
                )
                next_lives_value = next_info.get("lives", previous_lives)
                next_lives = (
                    int(next_lives_value)
                    if next_lives_value is not None
                    else None
                )
                life_lost = bool(
                    previous_lives is not None
                    and next_lives is not None
                    and next_lives < previous_lives
                )
                max_length_reached = (
                    episode_length + 1 >= args.max_episode_steps
                )
                episode_done = environment_done or max_length_reached
                learning_terminal = episode_done or (
                    args.life_loss_terminal and life_lost
                )
                components = reward_components(
                    raw_reward,
                    life_lost,
                    reward_config,
                    ale_frames_advanced,
                )
                training_reward = components.total
                if not np.isfinite(training_reward):
                    raise FloatingPointError(
                        "NaN or Inf detected in transformed reward"
                    )

                buffer.add(
                    obs,
                    action,
                    training_reward,
                    learning_terminal,
                    float(log_probability.item()),
                    float(value.item()),
                    raw_reward=raw_reward,
                    score_component=components.score,
                    survival_component=components.survival,
                    life_loss_component=components.life_loss,
                    ale_frames_advanced=ale_frames_advanced,
                )
                rollout_action_counts[action] += 1
                obs = next_obs
                current_info = next_info
                previous_lives = next_lives
                global_steps += 1
                episode_raw_return += raw_reward
                episode_training_return += training_reward
                episode_score_component_return += components.score
                episode_survival_component_return += components.survival
                episode_life_loss_component_return += components.life_loss
                episode_length += 1
                episode_life_losses += int(life_lost)
                last_learning_terminal = learning_terminal

                if episode_done:
                    episode_index += 1
                    elapsed = time.time() - start_time
                    episode_rows.append(
                        {
                            "episode": episode_index,
                            "env_steps": global_steps,
                            "ale_frames": global_steps * args.frameskip,
                            "raw_return": episode_raw_return,
                            "training_return": episode_training_return,
                            "score_component_return": (
                                episode_score_component_return
                            ),
                            "survival_component_return": (
                                episode_survival_component_return
                            ),
                            "life_loss_component_return": (
                                episode_life_loss_component_return
                            ),
                            "episode_length": episode_length,
                            "life_losses": episode_life_losses,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    obs, info = env.reset()
                    obs, info = maybe_fire_after_reset(
                        env,
                        obs,
                        info,
                        args.auto_fire_reset,
                    )
                    current_info = info
                    lives_value = info.get("lives")
                    previous_lives = (
                        int(lives_value)
                        if lives_value is not None
                        else None
                    )
                    episode_raw_return = 0.0
                    episode_training_return = 0.0
                    episode_score_component_return = 0.0
                    episode_survival_component_return = 0.0
                    episode_life_loss_component_return = 0.0
                    episode_length = 0
                    episode_life_losses = 0
                    if episode_index % args.log_interval_episodes == 0:
                        recent = episode_rows[-args.log_interval_episodes :]
                        print(
                            f"[train] steps={global_steps:>9}/{args.total_steps} "
                            f"episodes={episode_index:>6} "
                            f"avg_raw={np.mean([row['raw_return'] for row in recent]):>8.2f}"
                        )

                if global_steps >= args.total_steps:
                    break

            if buffer.ptr == 0:
                break
            if last_learning_terminal:
                last_value = 0.0
            else:
                with torch.no_grad():
                    _last_logits, last_value_tensor = model(
                        torch.from_numpy(obs).unsqueeze(0).to(device)
                    )
                    last_value = float(last_value_tensor.item())
            advantage_values, return_values = buffer.compute_gae(
                last_value,
                ppo_config.gamma,
                ppo_config.gae_lambda,
            )

            if ppo_config.anneal_learning_rate:
                learning_fraction = max(
                    0.0,
                    1.0
                    - (global_steps - buffer.ptr) / max(1, args.total_steps),
                )
                current_learning_rate = (
                    ppo_config.learning_rate * learning_fraction
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = current_learning_rate
            else:
                current_learning_rate = ppo_config.learning_rate

            observations_tensor = torch.from_numpy(
                buffer.obs[: buffer.ptr]
            ).to(device)
            actions_tensor = torch.from_numpy(
                buffer.actions[: buffer.ptr]
            ).to(device)
            old_log_probabilities_tensor = torch.from_numpy(
                buffer.log_probs[: buffer.ptr]
            ).to(device)
            old_values_tensor = torch.from_numpy(
                buffer.values[: buffer.ptr]
            ).to(device)
            advantages_tensor = torch.from_numpy(advantage_values).to(device)
            returns_tensor = torch.from_numpy(return_values).to(device)
            losses = ppo_update(
                model,
                optimizer,
                observations_tensor,
                actions_tensor,
                old_log_probabilities_tensor,
                old_values_tensor,
                advantages_tensor,
                returns_tensor,
                ppo_config,
            )
            update_index += 1
            update_rows.append(
                {
                    "update": update_index,
                    "env_steps": global_steps,
                    "ale_frames": global_steps * args.frameskip,
                    "learning_rate": current_learning_rate,
                    **losses,
                    "explained_variance": explained_variance(
                        buffer.values[: buffer.ptr],
                        return_values,
                    ),
                    "value_mean": float(
                        np.mean(buffer.values[: buffer.ptr])
                    ),
                    "value_std": float(
                        np.std(buffer.values[: buffer.ptr], ddof=0)
                    ),
                    "advantage_mean": float(np.mean(advantage_values)),
                    "advantage_std": float(
                        np.std(advantage_values, ddof=0)
                    ),
                    "return_mean": float(np.mean(return_values)),
                    "raw_reward_mean": float(
                        np.mean(buffer.raw_rewards[: buffer.ptr])
                    ),
                    "raw_reward_std": float(
                        np.std(
                            buffer.raw_rewards[: buffer.ptr],
                            ddof=0,
                        )
                    ),
                    "training_reward_mean": float(
                        np.mean(buffer.rewards[: buffer.ptr])
                    ),
                    "training_reward_std": float(
                        np.std(buffer.rewards[: buffer.ptr], ddof=0)
                    ),
                    "score_component_mean": float(
                        np.mean(
                            buffer.score_components[: buffer.ptr]
                        )
                    ),
                    "survival_component_mean": float(
                        np.mean(
                            buffer.survival_components[: buffer.ptr]
                        )
                    ),
                    "life_loss_component_mean": float(
                        np.mean(
                            buffer.life_loss_components[: buffer.ptr]
                        )
                    ),
                    "ale_frames_advanced_mean": float(
                        np.mean(
                            buffer.ale_frames_advanced[: buffer.ptr]
                        )
                    ),
                }
            )
            action_row: Dict[str, Any] = {
                "update": update_index,
                "env_steps": global_steps,
            }
            for action_index, count in enumerate(rollout_action_counts):
                safe_name = (
                    action_meanings[action_index]
                    .lower()
                    .replace(" ", "_")
                )
                action_row[f"count_{action_index}_{safe_name}"] = int(count)
                action_row[f"fraction_{action_index}_{safe_name}"] = float(
                    count / max(1, buffer.ptr)
                )
            action_rows.append(action_row)

            if global_steps >= next_eval_step:
                new_evaluations = append_evaluation_rows(
                    evaluation_rows,
                    stage="periodic",
                    env_steps=global_steps,
                    model=model,
                    args=args,
                    encoder_config=encoder_config,
                    device=device,
                )
                deterministic_return = next(
                    row["raw_return_mean"]
                    for row in new_evaluations
                    if row["policy"] == "deterministic"
                )
                if deterministic_return > best_evaluation_return:
                    best_evaluation_return = deterministic_return
                    save_checkpoint(
                        output_dir / "checkpoint_best.pt",
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        encoder_config=encoder_config,
                        ppo_config=ppo_config,
                        reward_config=reward_config,
                        env_steps=global_steps,
                        update_index=update_index,
                        episode_index=episode_index,
                        episode_rows=episode_rows,
                        update_rows=update_rows,
                        evaluation_rows=evaluation_rows,
                        action_rows=action_rows,
                        observation_statistics=observation_statistics,
                        object_diagnostics=object_diagnostics,
                        best_evaluation_return=best_evaluation_return,
                        n_actions=n_actions,
                        action_meanings=action_meanings,
                    )
                while next_eval_step <= global_steps:
                    next_eval_step += args.eval_interval

            if (
                global_steps >= next_checkpoint_step
                or global_steps >= args.total_steps
            ):
                save_progress_files(
                    output_dir,
                    episode_rows,
                    update_rows,
                    evaluation_rows,
                    action_rows,
                    observation_statistics,
                    object_diagnostics,
                    encoder_config,
                )
                save_checkpoint(
                    output_dir / "checkpoint_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    encoder_config=encoder_config,
                    ppo_config=ppo_config,
                    reward_config=reward_config,
                    env_steps=global_steps,
                    update_index=update_index,
                    episode_index=episode_index,
                    episode_rows=episode_rows,
                    update_rows=update_rows,
                    evaluation_rows=evaluation_rows,
                    action_rows=action_rows,
                    observation_statistics=observation_statistics,
                    object_diagnostics=object_diagnostics,
                    best_evaluation_return=best_evaluation_return,
                    n_actions=n_actions,
                    action_meanings=action_meanings,
                )
                while next_checkpoint_step <= global_steps:
                    next_checkpoint_step += args.checkpoint_interval

        append_evaluation_rows(
            evaluation_rows,
            stage="final",
            env_steps=global_steps,
            model=model,
            args=args,
            encoder_config=encoder_config,
            device=device,
        )
        elapsed_seconds = time.time() - start_time
        save_progress_files(
            output_dir,
            episode_rows,
            update_rows,
            evaluation_rows,
            action_rows,
            observation_statistics,
            object_diagnostics,
            encoder_config,
        )
        save_json(output_dir / "object_samples.json", object_samples)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "env_id": args.env,
                "obs_dim": env.obs_dim,
                "n_actions": n_actions,
                "action_meanings": action_meanings,
                "encoder_config": asdict(encoder_config),
                "ppo_config": asdict(ppo_config),
                "reward_config": asdict(reward_config),
                "steps": global_steps,
                "seed": args.seed,
            },
            output_dir / "model.pt",
        )
        save_checkpoint(
            output_dir / "checkpoint_latest.pt",
            model=model,
            optimizer=optimizer,
            args=args,
            encoder_config=encoder_config,
            ppo_config=ppo_config,
            reward_config=reward_config,
            env_steps=global_steps,
            update_index=update_index,
            episode_index=episode_index,
            episode_rows=episode_rows,
            update_rows=update_rows,
            evaluation_rows=evaluation_rows,
            action_rows=action_rows,
            observation_statistics=observation_statistics,
            object_diagnostics=object_diagnostics,
            best_evaluation_return=best_evaluation_return,
            n_actions=n_actions,
            action_meanings=action_meanings,
        )
        summary = final_summary(
            args=args,
            episode_rows=episode_rows,
            update_rows=update_rows,
            evaluation_rows=evaluation_rows,
            starting_env_steps=starting_env_steps,
            env_steps=global_steps,
            elapsed_seconds=elapsed_seconds,
            object_diagnostics=object_diagnostics,
        )
        save_json(output_dir / "summary.json", summary)
        plot_results(
            output_dir,
            episode_rows,
            update_rows,
            evaluation_rows,
            args.moving_average_window,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return output_dir
    finally:
        env.close()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train source-only PPO on an OCAtari REM object representation. "
            "Transfer learning is intentionally not run."
        )
    )
    parser.add_argument("--env", default="ALE/SpaceInvaders-v5")
    parser.add_argument(
        "--object-mode",
        choices=["ram", "vision"],
        default="ram",
        help="Use ram for this research stage; vision remains diagnostic only.",
    )
    parser.add_argument("--total-steps", type=positive_int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--frameskip", type=positive_int, default=4)
    parser.add_argument(
        "--repeat-action-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument("--stack-size", type=positive_int, default=4)
    parser.add_argument("--max-enemies", type=positive_int, default=12)
    parser.add_argument("--max-projectiles", type=positive_int, default=4)
    parser.add_argument(
        "--slot-strategy",
        choices=["temporal", "distance"],
        default="temporal",
    )
    parser.add_argument("--slot-match-distance", type=float, default=0.20)

    parser.add_argument(
        "--reward-mode",
        choices=[
            "raw",
            "clipped",
            "scaled_raw",
            "scaled_survival",
            "shaped",
        ],
        default="scaled_raw",
    )
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument(
        "--survival-reward-per-frame",
        type=float,
        default=0.001,
        help=(
            "Final learning-reward bonus per actual ALE frame in "
            "scaled_survival mode."
        ),
    )
    parser.add_argument(
        "--life-loss-penalty",
        type=float,
        default=-1.0,
        help="Final learning-reward penalty per lost life in scaled_survival.",
    )
    parser.add_argument("--survival-bonus", type=float, default=0.5)
    parser.add_argument("--death-penalty", type=float, default=-5.0)

    parser.add_argument("--rollout-steps", type=positive_int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument(
        "--anneal-learning-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--value-clip-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--value-clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=positive_int, default=4)
    parser.add_argument("--minibatch-size", type=positive_int, default=256)
    parser.add_argument("--hidden-size", type=positive_int, default=256)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="Set to 0 to disable target-KL early stopping.",
    )

    parser.add_argument("--eval-interval", type=nonnegative_int, default=100_000)
    parser.add_argument("--eval-episodes", type=positive_int, default=10)
    parser.add_argument(
        "--random-eval-episodes",
        type=positive_int,
        default=20,
    )
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint-interval",
        type=nonnegative_int,
        default=100_000,
    )
    parser.add_argument(
        "--max-episode-steps",
        type=positive_int,
        default=27_000,
    )
    parser.add_argument(
        "--life-loss-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask GAE at life loss without ending the raw-return episode.",
    )
    parser.add_argument(
        "--auto-fire-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--object-sample-frames",
        type=nonnegative_int,
        default=20,
    )
    parser.add_argument(
        "--moving-average-window",
        type=positive_int,
        default=100,
    )
    parser.add_argument(
        "--log-interval-episodes",
        type=positive_int,
        default=10,
    )
    args = parser.parse_args()

    if not 0.0 <= args.repeat_action_probability <= 1.0:
        parser.error("--repeat-action-probability must be in [0, 1]")
    if args.slot_match_distance <= 0.0:
        parser.error("--slot-match-distance must be positive")
    if args.reward_scale <= 0.0:
        parser.error("--reward-scale must be positive")
    if args.survival_reward_per_frame < 0.0:
        parser.error("--survival-reward-per-frame must be non-negative")
    if args.life_loss_penalty > 0.0:
        parser.error("--life-loss-penalty must be non-positive")
    if args.target_kl < 0.0:
        parser.error("--target-kl must be non-negative")
    if args.object_mode != "ram":
        print(
            "Warning: this stage is defined for OCAtari REM; "
            "--object-mode vision is not a baseline condition.",
            file=sys.stderr,
        )
    if args.quick:
        args.total_steps = min(args.total_steps, 2_048)
        args.rollout_steps = min(args.rollout_steps, 512)
        args.ppo_epochs = min(args.ppo_epochs, 2)
        args.minibatch_size = min(args.minibatch_size, 128)
        args.eval_interval = min(
            args.eval_interval if args.eval_interval > 0 else 1_024,
            1_024,
        )
        args.checkpoint_interval = min(
            (
                args.checkpoint_interval
                if args.checkpoint_interval > 0
                else 1_024
            ),
            1_024,
        )
        args.eval_episodes = min(args.eval_episodes, 2)
        args.random_eval_episodes = min(args.random_eval_episodes, 2)
        args.max_episode_steps = min(args.max_episode_steps, 5_000)
    if args.resume and not args.output_dir:
        args.output_dir = str(Path(args.resume).resolve().parent)
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
