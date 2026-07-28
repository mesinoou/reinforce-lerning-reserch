#!/usr/bin/env python3
"""
OCAtari object-centric PPO transfer experiment (all-in-one)
============================================================

Runs the complete verification pipeline:
  1) Train a SOURCE PPO agent on an OCAtari environment.
  2) Train a TARGET PPO baseline without transfer.
  3) Train a TARGET PPO agent with advantage transfer from the source critic.
  4) Evaluate all trained agents with deterministic actions.
  5) Save checkpoints, CSV logs, JSON summaries, object-detection snapshots,
     and comparison graphs.

The transfer rule follows the uploaded research program and the paper:

    A_S(s_t, a_t) = r_t + gamma * V_S(s_{t+1}) - V_S(s_t)
    A_total         = (1 - alpha) * A_T + alpha * A_S

A_T is target GAE, as in the user's previous implementation. The target critic
is trained only from target returns; A_total is used only for the target actor.

Default experiment:
  SOURCE: ALE/SpaceInvaders-v5
  TARGET: ALE/Galaxian-v5
  OCAtari extraction: RAM (REM)

Install:
  pip install ocatari "gymnasium[atari,accept-rom-license]" ale-py \
      torch numpy matplotlib

Quick pipeline check:
  python ocatari_transfer_full_experiment.py --quick

More meaningful verification:
  python ocatari_transfer_full_experiment.py \
      --source_steps 500000 --target_steps 500000 --eval_episodes 20

Notes:
- The source and target must expose the same number of discrete actions.
- The common object encoder projects both games to the same semantic vector:
  player, nearest enemies, nearest projectiles, and global counts/centroids.
- Current OCAtari versions have differed in the ordering/naming of terminated
  and truncated flags. This script uses their logical OR, which is sufficient
  for episode boundaries and GAE masking.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# Training is normally executed on a headless Linux host.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from ocatari.core import OCAtari
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "OCAtari is not installed. Run:\n"
        "  pip install ocatari \"gymnasium[atari,accept-rom-license]\" ale-py\n"
    ) from exc


# ---------------------------------------------------------------------------
# Reproducibility and utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels can reduce speed, but are preferable for a small
    # verification experiment.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), arr.size))
    kernel = np.ones(window, dtype=np.float64) / window
    # Prefix with expanding means so the output length remains unchanged.
    prefix = np.array([arr[: i + 1].mean() for i in range(window - 1)])
    body = np.convolve(arr, kernel, mode="valid")
    return np.concatenate([prefix, body])


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def module_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Common object-centric state encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    max_enemies: int = 12
    max_projectiles: int = 4
    screen_width: float = 160.0
    screen_height: float = 210.0
    stack_size: int = 4
    slot_strategy: str = "temporal"
    slot_match_distance: float = 0.20


class CommonObjectEncoder:
    """Map game-specific OCAtari objects into one common semantic vector.

    Per-frame vector:
      player: 5 values
        [present, absolute_x, absolute_y, width, height]
      nearest enemies: max_enemies * 5
        [present, relative_x, relative_y, width, height]
      nearest projectiles: max_projectiles * 5
        [present, relative_x, relative_y, width, height]
      global: 6 values
        [enemy_count, projectile_count,
         enemy_centroid_relative_x, enemy_centroid_relative_y,
         nearest_enemy_distance, nearest_projectile_distance]

    Four consecutive vectors are concatenated by default, preserving movement
    information without relying on game-specific object IDs.
    """

    PLAYER_NAMES = {"player"}
    ENEMY_TOKENS = ("alien", "satellite", "enemyship", "divingenemy")
    PROJECTILE_TOKENS = ("bullet", "missile")

    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        if cfg.slot_strategy not in {"temporal", "distance"}:
            raise ValueError("slot_strategy must be 'temporal' or 'distance'")
        if cfg.slot_match_distance <= 0.0:
            raise ValueError("slot_match_distance must be positive")
        self._frames: Deque[np.ndarray] = deque(maxlen=cfg.stack_size)
        self.last_player_center = (cfg.screen_width / 2.0, cfg.screen_height * 0.88)
        self._enemy_slot_centers: List[Optional[Tuple[float, float]]] = [
            None
        ] * cfg.max_enemies
        self._projectile_slot_centers: List[Optional[Tuple[float, float]]] = [
            None
        ] * cfg.max_projectiles

    @property
    def frame_dim(self) -> int:
        return 5 + self.cfg.max_enemies * 5 + self.cfg.max_projectiles * 5 + 6

    @property
    def obs_dim(self) -> int:
        return self.frame_dim * self.cfg.stack_size

    def reset(self, objects: Iterable[Any]) -> np.ndarray:
        self._frames.clear()
        self.last_player_center = (
            self.cfg.screen_width / 2.0,
            self.cfg.screen_height * 0.88,
        )
        self._enemy_slot_centers = [None] * self.cfg.max_enemies
        self._projectile_slot_centers = [None] * self.cfg.max_projectiles
        frame = self.encode_frame(objects)
        for _ in range(self.cfg.stack_size):
            self._frames.append(frame.copy())
        return self.stacked()

    def step(self, objects: Iterable[Any]) -> np.ndarray:
        self._frames.append(self.encode_frame(objects))
        return self.stacked()

    def stacked(self) -> np.ndarray:
        if len(self._frames) != self.cfg.stack_size:
            raise RuntimeError("Object-state stack is not initialized")
        return np.concatenate(list(self._frames), axis=0).astype(np.float32)

    @staticmethod
    def _visible(obj: Any) -> bool:
        try:
            return bool(obj)
        except Exception:
            return True

    @staticmethod
    def _category(obj: Any) -> str:
        value = getattr(obj, "category", obj.__class__.__name__)
        return str(value).replace("_", "").lower()

    @staticmethod
    def _xywh(obj: Any) -> Tuple[float, float, float, float]:
        if hasattr(obj, "xywh"):
            x, y, w, h = obj.xywh
        else:
            x, y = getattr(obj, "xy", (0.0, 0.0))
            w, h = getattr(obj, "wh", (0.0, 0.0))
        return float(x), float(y), float(w), float(h)

    def _center(self, obj: Any) -> Tuple[float, float]:
        x, y, w, h = self._xywh(obj)
        return x + w / 2.0, y + h / 2.0

    def _is_enemy(self, category: str) -> bool:
        return any(token in category for token in self.ENEMY_TOKENS)

    def _is_projectile(self, category: str) -> bool:
        return any(token in category for token in self.PROJECTILE_TOKENS)

    def _object_feature(
        self,
        obj: Any,
        player_center: Tuple[float, float],
        relative: bool,
    ) -> List[float]:
        x, y, w, h = self._xywh(obj)
        cx, cy = x + w / 2.0, y + h / 2.0
        if relative:
            fx = np.clip((cx - player_center[0]) / self.cfg.screen_width, -1.0, 1.0)
            fy = np.clip((cy - player_center[1]) / self.cfg.screen_height, -1.0, 1.0)
        else:
            fx = np.clip(cx / self.cfg.screen_width, 0.0, 1.0)
            fy = np.clip(cy / self.cfg.screen_height, 0.0, 1.0)
        return [
            1.0,
            float(fx),
            float(fy),
            float(np.clip(w / self.cfg.screen_width, 0.0, 1.0)),
            float(np.clip(h / self.cfg.screen_height, 0.0, 1.0)),
        ]

    def _normalized_center(self, obj: Any) -> Tuple[float, float]:
        cx, cy = self._center(obj)
        return cx / self.cfg.screen_width, cy / self.cfg.screen_height

    def _assign_temporal_slots(
        self,
        objects: List[Any],
        previous_centers: List[Optional[Tuple[float, float]]],
        player_center: Tuple[float, float],
    ) -> Tuple[List[Optional[Any]], List[Optional[Tuple[float, float]]]]:
        """Greedily preserve object identity across adjacent encoded frames.

        Existing slots are matched first by normalized screen distance. Objects
        that cannot be matched are inserted into empty slots in player-distance
        order. This removes the all-slots shift caused by sorting every frame.
        """
        slots: List[Optional[Any]] = [None] * len(previous_centers)
        new_centers: List[Optional[Tuple[float, float]]] = [None] * len(previous_centers)
        object_centers = [self._normalized_center(obj) for obj in objects]
        unmatched = set(range(len(objects)))

        candidate_pairs: List[Tuple[float, int, int]] = []
        for slot_index, previous in enumerate(previous_centers):
            if previous is None:
                continue
            for object_index, center in enumerate(object_centers):
                candidate_pairs.append(
                    (
                        math.hypot(center[0] - previous[0], center[1] - previous[1]),
                        slot_index,
                        object_index,
                    )
                )

        occupied_slots = set()
        for distance_value, slot_index, object_index in sorted(candidate_pairs):
            if distance_value > self.cfg.slot_match_distance:
                break
            if slot_index in occupied_slots or object_index not in unmatched:
                continue
            slots[slot_index] = objects[object_index]
            new_centers[slot_index] = object_centers[object_index]
            occupied_slots.add(slot_index)
            unmatched.remove(object_index)

        def player_distance(object_index: int) -> float:
            cx, cy = self._center(objects[object_index])
            return math.hypot(
                (cx - player_center[0]) / self.cfg.screen_width,
                (cy - player_center[1]) / self.cfg.screen_height,
            )

        empty_slots = [i for i, value in enumerate(slots) if value is None]
        for slot_index, object_index in zip(
            empty_slots,
            sorted(unmatched, key=player_distance),
        ):
            slots[slot_index] = objects[object_index]
            new_centers[slot_index] = object_centers[object_index]

        return slots, new_centers

    def encode_frame(self, objects: Iterable[Any]) -> np.ndarray:
        filtered = [obj for obj in objects if self._visible(obj)]

        player = None
        enemies: List[Any] = []
        projectiles: List[Any] = []

        for obj in filtered:
            cat = self._category(obj)
            if cat in self.PLAYER_NAMES and player is None:
                player = obj
            elif self._is_enemy(cat):
                enemies.append(obj)
            elif self._is_projectile(cat):
                projectiles.append(obj)

        if player is not None:
            player_center = self._center(player)
            self.last_player_center = player_center
            player_feat = self._object_feature(player, player_center, relative=False)
        else:
            player_center = self.last_player_center
            # Retain the last known position but mark the player as absent.
            player_feat = [
                0.0,
                float(np.clip(player_center[0] / self.cfg.screen_width, 0.0, 1.0)),
                float(np.clip(player_center[1] / self.cfg.screen_height, 0.0, 1.0)),
                0.0,
                0.0,
            ]

        def distance(obj: Any) -> float:
            cx, cy = self._center(obj)
            return math.hypot(
                (cx - player_center[0]) / self.cfg.screen_width,
                (cy - player_center[1]) / self.cfg.screen_height,
            )

        enemies.sort(key=distance)
        projectiles.sort(key=distance)

        if self.cfg.slot_strategy == "temporal":
            enemy_slots, self._enemy_slot_centers = self._assign_temporal_slots(
                enemies,
                self._enemy_slot_centers,
                player_center,
            )
            projectile_slots, self._projectile_slot_centers = self._assign_temporal_slots(
                projectiles,
                self._projectile_slot_centers,
                player_center,
            )
        else:
            enemy_slots = list(enemies[: self.cfg.max_enemies])
            enemy_slots.extend([None] * (self.cfg.max_enemies - len(enemy_slots)))
            projectile_slots = list(projectiles[: self.cfg.max_projectiles])
            projectile_slots.extend(
                [None] * (self.cfg.max_projectiles - len(projectile_slots))
            )

        features: List[float] = list(player_feat)

        for enemy in enemy_slots:
            if enemy is not None:
                features.extend(self._object_feature(enemy, player_center, relative=True))
            else:
                features.extend([0.0] * 5)

        for projectile in projectile_slots:
            if projectile is not None:
                features.extend(
                    self._object_feature(projectile, player_center, relative=True)
                )
            else:
                features.extend([0.0] * 5)

        enemy_count = min(len(enemies), 40) / 40.0
        projectile_count = min(len(projectiles), 8) / 8.0

        if enemies:
            centers = np.asarray([self._center(obj) for obj in enemies], dtype=np.float32)
            centroid = centers.mean(axis=0)
            centroid_dx = float(np.clip((centroid[0] - player_center[0]) / self.cfg.screen_width, -1.0, 1.0))
            centroid_dy = float(np.clip((centroid[1] - player_center[1]) / self.cfg.screen_height, -1.0, 1.0))
            nearest_enemy = float(np.clip(distance(enemies[0]) / math.sqrt(2.0), 0.0, 1.0))
        else:
            centroid_dx = centroid_dy = 0.0
            nearest_enemy = 1.0

        if projectiles:
            nearest_projectile = float(np.clip(distance(projectiles[0]) / math.sqrt(2.0), 0.0, 1.0))
        else:
            nearest_projectile = 1.0

        features.extend(
            [
                float(enemy_count),
                float(projectile_count),
                centroid_dx,
                centroid_dy,
                nearest_enemy,
                nearest_projectile,
            ]
        )

        out = np.asarray(features, dtype=np.float32)
        if out.shape != (self.frame_dim,):
            raise RuntimeError(f"Unexpected encoded frame shape: {out.shape}, expected {(self.frame_dim,)}")
        return out


# ---------------------------------------------------------------------------
# OCAtari environment adapter
# ---------------------------------------------------------------------------

class ObjectEnv:
    def __init__(
        self,
        env_id: str,
        encoder_cfg: EncoderConfig,
        mode: str,
        frameskip: int,
        repeat_action_probability: float,
    ):
        self.env_id = env_id
        self.encoder = CommonObjectEncoder(encoder_cfg)

        kwargs: Dict[str, Any] = {
            "mode": mode,
            "hud": False,
            "obs_mode": "ori",
            "render_mode": None,
            "create_buffer_stacks": [],
            "frameskip": frameskip,
            "repeat_action_probability": repeat_action_probability,
            "full_action_space": False,
        }
        try:
            self.env = OCAtari(env_id, **kwargs)
        except TypeError:
            # Compatibility fallback for releases that reject one of the ALE
            # constructor options.
            kwargs.pop("repeat_action_probability", None)
            try:
                self.env = OCAtari(env_id, **kwargs)
            except TypeError:
                kwargs.pop("frameskip", None)
                self.env = OCAtari(env_id, **kwargs)

        self.action_space = self.env.action_space
        self.obs_dim = self.encoder.obs_dim
        self.current_obs: Optional[np.ndarray] = None
        self.last_info: Dict[str, Any] = {}

    @property
    def objects(self) -> List[Any]:
        return list(getattr(self.env, "objects", []))

    def action_meanings(self) -> List[str]:
        for candidate in (self.env, getattr(self.env, "unwrapped", None), getattr(self.env, "_env", None)):
            if candidate is None:
                continue
            try:
                return list(candidate.get_action_meanings())
            except Exception:
                pass
        return [str(i) for i in range(int(self.action_space.n))]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is None:
            _obs, info = self.env.reset()
        else:
            _obs, info = self.env.reset(seed=seed)
        self.last_info = dict(info or {})
        self.current_obs = self.encoder.reset(self.objects)
        return self.current_obs.copy(), self.last_info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        _obs, reward, flag3, flag4, info = self.env.step(int(action))
        done = bool(flag3 or flag4)
        self.last_info = dict(info or {})
        self.current_obs = self.encoder.step(self.objects)
        return self.current_obs.copy(), float(reward), done, self.last_info

    def rgb_frame(self) -> np.ndarray:
        try:
            return np.asarray(self.env.getScreenRGB())
        except Exception:
            value = getattr(self.env, "get_rgb_state")
            return np.asarray(value() if callable(value) else value)

    def close(self) -> None:
        self.env.close()

    def save_detection_snapshot(self, path: Path, title: str) -> None:
        frame = self.rgb_frame()
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.imshow(frame)
        for obj in self.objects:
            try:
                if not bool(obj):
                    continue
            except Exception:
                pass
            try:
                x, y, w, h = obj.xywh
                cat = getattr(obj, "category", obj.__class__.__name__)
                rgb = np.asarray(getattr(obj, "rgb", (255, 255, 255)), dtype=np.float32) / 255.0
                rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.2, edgecolor=rgb)
                ax.add_patch(rect)
                ax.text(x, max(0, y - 2), str(cat), fontsize=6, color="white", backgroundcolor="black")
            except Exception:
                continue
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------------------
# PPO model and rollout buffer
# ---------------------------------------------------------------------------

class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        self.apply(self._init_layer)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @staticmethod
    def _init_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("Rollout buffer overflow")
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[: self.ptr]
        return advantages, returns


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_epsilon: float = 0.1
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 1024
    hidden_size: int = 256


@dataclass
class RewardConfig:
    survival_bonus: float = 0.5
    death_penalty: float = -5.0
    reward_scale: float = 0.1


@dataclass
class TrainResult:
    model: ActorCriticMLP
    episode_rows: List[Dict[str, Any]]
    diagnostic_rows: List[Dict[str, Any]]
    checkpoint_path: Path


def ppo_update(
    model: ActorCriticMLP,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> Dict[str, float]:
    normalized_adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = obs.shape[0]
    indices = np.arange(n)

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]
            logits, values = model(obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            surrogate1 = ratio * normalized_adv[mb]
            surrogate2 = torch.clamp(
                ratio,
                1.0 - cfg.clip_epsilon,
                1.0 + cfg.clip_epsilon,
            ) * normalized_adv[mb]
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = 0.5 * (returns[mb] - values).pow(2).mean()
            loss = (
                policy_loss
                + cfg.value_coefficient * value_loss
                - cfg.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


@torch.no_grad()
def source_td_advantage(
    source_model: ActorCriticMLP,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    _logits, values = source_model(obs)
    _next_logits, next_values = source_model(next_obs)
    return rewards + gamma * next_values * (1.0 - dones) - values


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    source_env: str
    target_env: str
    object_mode: str
    source_steps: int
    target_steps: int
    eval_episodes: int
    alpha: float
    alpha_decay: float
    alpha_min: float
    frameskip: int
    repeat_action_probability: float
    seed: int
    moving_average_window: int
    output_dir: str
    device: str


def shape_reward(
    raw_reward: float,
    previous_lives: Optional[int],
    info: Dict[str, Any],
    cfg: RewardConfig,
) -> Tuple[float, Optional[int]]:
    reward = float(raw_reward) + cfg.survival_bonus
    lives_value = info.get("lives", previous_lives)
    lives = int(lives_value) if lives_value is not None else None
    if previous_lives is not None and lives is not None and lives < previous_lives:
        reward += cfg.death_penalty
    return reward * cfg.reward_scale, lives


def train_agent(
    *,
    label: str,
    env_id: str,
    total_steps: int,
    seed: int,
    output_dir: Path,
    encoder_cfg: EncoderConfig,
    ppo_cfg: PPOConfig,
    reward_cfg: RewardConfig,
    object_mode: str,
    frameskip: int,
    repeat_action_probability: float,
    device: torch.device,
    initial_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    source_model: Optional[ActorCriticMLP] = None,
    alpha: float = 0.0,
    alpha_decay: float = 1.0,
    alpha_min: float = 0.0,
) -> TrainResult:
    set_global_seed(seed)
    ensure_dir(output_dir)

    env = ObjectEnv(
        env_id,
        encoder_cfg,
        object_mode,
        frameskip,
        repeat_action_probability,
    )
    obs, info = env.reset(seed=seed)
    n_actions = int(env.action_space.n)

    model = ActorCriticMLP(env.obs_dim, n_actions, ppo_cfg.hidden_size).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(copy.deepcopy(initial_state_dict), strict=True)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    if source_model is not None:
        if source_model.policy_head.out_features != n_actions:
            raise ValueError(
                f"Action-space mismatch: source={source_model.policy_head.out_features}, "
                f"target={n_actions}. An explicit action mapping is required."
            )
        if source_model.backbone[0].in_features != env.obs_dim:
            raise ValueError("Source and target object encoders do not have the same dimension")
        source_model.eval()
        for p in source_model.parameters():
            p.requires_grad_(False)

    env.save_detection_snapshot(output_dir / "object_detection_snapshot.png", f"{label}: {env_id}")

    buffer = RolloutBuffer(ppo_cfg.rollout_steps, env.obs_dim)
    episode_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    global_steps = 0
    episode_index = 0
    update_index = 0
    episode_raw_return = 0.0
    episode_shaped_return = 0.0
    episode_length = 0
    previous_lives = info.get("lives")
    previous_lives = int(previous_lives) if previous_lives is not None else None
    current_alpha = float(alpha)
    start_time = time.time()
    last_transition_done = False

    while global_steps < total_steps:
        buffer.reset()
        remaining = total_steps - global_steps
        rollout_target = min(ppo_cfg.rollout_steps, remaining)

        for _ in range(rollout_target):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)

            next_obs, raw_reward, done, next_info = env.step(int(action_tensor.item()))
            shaped_reward, current_lives = shape_reward(
                raw_reward,
                previous_lives,
                next_info,
                reward_cfg,
            )
            previous_lives = current_lives

            buffer.add(
                obs,
                next_obs,
                int(action_tensor.item()),
                shaped_reward,
                done,
                float(log_prob.item()),
                float(value.item()),
            )

            obs = next_obs
            global_steps += 1
            episode_raw_return += raw_reward
            episode_shaped_return += shaped_reward
            episode_length += 1
            last_transition_done = done

            if done:
                episode_index += 1
                elapsed = time.time() - start_time
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "env_steps": global_steps,
                        "raw_return": episode_raw_return,
                        "shaped_return": episode_shaped_return,
                        "episode_length": episode_length,
                        "alpha": current_alpha,
                        "elapsed_seconds": elapsed,
                    }
                )

                current_alpha = max(alpha_min, min(1.0, current_alpha * alpha_decay))
                obs, info = env.reset()
                previous_lives = info.get("lives")
                previous_lives = int(previous_lives) if previous_lives is not None else None
                episode_raw_return = 0.0
                episode_shaped_return = 0.0
                episode_length = 0

                if episode_index % 10 == 0:
                    recent = episode_rows[-10:]
                    avg_raw = np.mean([row["raw_return"] for row in recent])
                    print(
                        f"[{label}] steps={global_steps:>8}/{total_steps} "
                        f"episodes={episode_index:>5} avg_raw10={avg_raw:>8.2f} "
                        f"alpha={current_alpha:.4f}"
                    )

            if global_steps >= total_steps:
                break

        if buffer.ptr == 0:
            break

        if last_transition_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                _logits, last_v = model(torch.from_numpy(obs).unsqueeze(0).to(device))
                last_value = float(last_v.item())

        target_adv_np, target_returns_np = buffer.compute_gae(
            last_value,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
        )

        obs_t = torch.from_numpy(buffer.obs[: buffer.ptr]).to(device)
        next_obs_t = torch.from_numpy(buffer.next_obs[: buffer.ptr]).to(device)
        actions_t = torch.from_numpy(buffer.actions[: buffer.ptr]).to(device)
        rewards_t = torch.from_numpy(buffer.rewards[: buffer.ptr]).to(device)
        dones_t = torch.from_numpy(buffer.dones[: buffer.ptr]).to(device)
        old_log_probs_t = torch.from_numpy(buffer.log_probs[: buffer.ptr]).to(device)
        target_adv_t = torch.from_numpy(target_adv_np).to(device)
        target_returns_t = torch.from_numpy(target_returns_np).to(device)

        with torch.no_grad():
            _target_logits, target_values = model(obs_t)

        if source_model is None or current_alpha <= 0.0:
            source_adv_t = torch.zeros_like(target_adv_t)
            mixed_adv_t = target_adv_t
            source_value_mean = 0.0
        else:
            source_adv_t = source_td_advantage(
                source_model,
                obs_t,
                next_obs_t,
                rewards_t,
                dones_t,
                ppo_cfg.gamma,
            )
            mixed_adv_t = (1.0 - current_alpha) * target_adv_t + current_alpha * source_adv_t
            with torch.no_grad():
                _source_logits, source_values = source_model(obs_t)
                source_value_mean = float(source_values.mean().cpu())

        losses = ppo_update(
            model,
            optimizer,
            obs_t,
            actions_t,
            old_log_probs_t,
            mixed_adv_t,
            target_returns_t,
            ppo_cfg,
        )

        update_index += 1
        diagnostic_rows.append(
            {
                "update": update_index,
                "env_steps": global_steps,
                "alpha": current_alpha,
                "target_adv_mean": float(target_adv_t.mean().cpu()),
                "target_adv_std": float(target_adv_t.std(unbiased=False).cpu()),
                "source_adv_mean": float(source_adv_t.mean().cpu()),
                "source_adv_std": float(source_adv_t.std(unbiased=False).cpu()),
                "mixed_adv_mean": float(mixed_adv_t.mean().cpu()),
                "mixed_adv_std": float(mixed_adv_t.std(unbiased=False).cpu()),
                "target_value_mean": float(target_values.mean().cpu()),
                "source_value_mean": source_value_mean,
                **losses,
            }
        )

    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "env_id": env_id,
            "obs_dim": env.obs_dim,
            "n_actions": n_actions,
            "action_meanings": env.action_meanings(),
            "encoder_config": asdict(encoder_cfg),
            "ppo_config": asdict(ppo_cfg),
            "reward_config": asdict(reward_cfg),
            "alpha_initial": alpha,
            "alpha_final": current_alpha,
            "steps": global_steps,
        },
        checkpoint_path,
    )

    save_csv(
        output_dir / "episodes.csv",
        episode_rows,
        [
            "episode",
            "env_steps",
            "raw_return",
            "shaped_return",
            "episode_length",
            "alpha",
            "elapsed_seconds",
        ],
    )
    save_csv(
        output_dir / "updates.csv",
        diagnostic_rows,
        [
            "update",
            "env_steps",
            "alpha",
            "target_adv_mean",
            "target_adv_std",
            "source_adv_mean",
            "source_adv_std",
            "mixed_adv_mean",
            "mixed_adv_std",
            "target_value_mean",
            "source_value_mean",
            "policy_loss",
            "value_loss",
            "entropy",
        ],
    )

    plot_single_training_run(episode_rows, output_dir, label)
    env.close()
    return TrainResult(model, episode_rows, diagnostic_rows, checkpoint_path)


@torch.no_grad()
def evaluate_agent(
    *,
    label: str,
    model: ActorCriticMLP,
    env_id: str,
    episodes: int,
    seed: int,
    output_dir: Path,
    encoder_cfg: EncoderConfig,
    object_mode: str,
    frameskip: int,
    repeat_action_probability: float,
    device: torch.device,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    model.eval()
    env = ObjectEnv(
        env_id,
        encoder_cfg,
        object_mode,
        frameskip,
        repeat_action_probability,
    )
    rows: List[Dict[str, Any]] = []

    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        done = False
        raw_return = 0.0
        length = 0
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            logits, _value = model(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, done, _info = env.step(action)
            raw_return += reward
            length += 1
        rows.append(
            {
                "episode": episode + 1,
                "seed": seed + episode,
                "raw_return": raw_return,
                "episode_length": length,
            }
        )

    env.close()
    returns = np.asarray([row["raw_return"] for row in rows], dtype=np.float64)
    lengths = np.asarray([row["episode_length"] for row in rows], dtype=np.float64)
    summary = {
        "label": label,
        "env_id": env_id,
        "episodes": episodes,
        "raw_return_mean": float(returns.mean()),
        "raw_return_std": float(returns.std(ddof=0)),
        "raw_return_min": float(returns.min()),
        "raw_return_max": float(returns.max()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_std": float(lengths.std(ddof=0)),
    }
    save_csv(
        output_dir / "evaluation_episodes.csv",
        rows,
        ["episode", "seed", "raw_return", "episode_length"],
    )
    save_json(output_dir / "evaluation_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_training_run(rows: List[Dict[str, Any]], output_dir: Path, label: str) -> None:
    if not rows:
        return
    steps = np.asarray([row["env_steps"] for row in rows])
    raw = np.asarray([row["raw_return"] for row in rows], dtype=np.float64)
    shaped = np.asarray([row["shaped_return"] for row in rows], dtype=np.float64)
    lengths = np.asarray([row["episode_length"] for row in rows], dtype=np.float64)
    window = min(20, len(rows))

    for values, ylabel, filename in (
        (raw, "Raw episode return", "raw_return_curve.png"),
        (shaped, "Shaped episode return", "shaped_return_curve.png"),
        (lengths, "Episode length", "episode_length_curve.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, values, alpha=0.25, label="Episode")
        ax.plot(steps, moving_average(values, window), label=f"Moving average ({window})")
        ax.set_title(label)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def plot_target_comparison(
    baseline_rows: List[Dict[str, Any]],
    transfer_rows: List[Dict[str, Any]],
    output_dir: Path,
    window: int,
) -> None:
    ensure_dir(output_dir)

    def extract(rows: List[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([row["env_steps"] for row in rows], dtype=np.float64),
            np.asarray([row[key] for row in rows], dtype=np.float64),
        )

    for key, ylabel, filename in (
        ("raw_return", "Raw episode return", "target_raw_return_comparison.png"),
        ("shaped_return", "Shaped episode return", "target_shaped_return_comparison.png"),
        ("episode_length", "Episode length", "target_episode_length_comparison.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for rows, label in ((baseline_rows, "No transfer"), (transfer_rows, "Transfer")):
            x, y = extract(rows, key)
            if y.size == 0:
                continue
            w = max(1, min(window, y.size))
            ax.plot(x, moving_average(y, w), label=f"{label} (MA {w})")
        ax.set_title("Target learning comparison")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)


def plot_transfer_diagnostics(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    steps = np.asarray([row["env_steps"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(steps, [row["target_adv_mean"] for row in rows], label="Target advantage")
    ax.plot(steps, [row["source_adv_mean"] for row in rows], label="Source advantage")
    ax.plot(steps, [row["mixed_adv_mean"] for row in rows], label="Mixed advantage")
    ax.set_title("Transfer advantage means")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean advantage")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "transfer_advantage_means.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(steps, [row["target_adv_std"] for row in rows], label="Target advantage")
    ax.plot(steps, [row["source_adv_std"] for row in rows], label="Source advantage")
    ax.plot(steps, [row["mixed_adv_std"] for row in rows], label="Mixed advantage")
    ax.set_title("Transfer advantage standard deviations")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Advantage standard deviation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "transfer_advantage_std.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(steps, [row["alpha"] for row in rows])
    ax.set_title("Transfer weight schedule")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Alpha")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "transfer_alpha_schedule.png", dpi=170)
    plt.close(fig)


def plot_evaluation_comparison(
    baseline: Dict[str, Any],
    transfer: Dict[str, Any],
    output_dir: Path,
) -> None:
    labels = ["No transfer", "Transfer"]
    means = [baseline["raw_return_mean"], transfer["raw_return_mean"]]
    stds = [baseline["raw_return_std"], transfer["raw_return_std"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean raw evaluation return")
    ax.set_title("Deterministic target evaluation")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "target_evaluation_comparison.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OCAtari source, target baseline, and target transfer agents in one run."
    )
    parser.add_argument("--source_env", default="ALE/SpaceInvaders-v5")
    parser.add_argument("--target_env", default="ALE/Galaxian-v5")
    parser.add_argument("--object_mode", choices=["vision", "ram"], default="ram")
    parser.add_argument("--source_steps", type=int, default=200_000)
    parser.add_argument("--target_steps", type=int, default=200_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha_decay", type=float, default=1.0)
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--repeat_action_probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--moving_average_window", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Fast pipeline test: 30k steps per phase, 3 evaluations.")

    # Object encoder
    parser.add_argument("--stack_size", type=int, default=4)
    parser.add_argument("--max_enemies", type=int, default=12)
    parser.add_argument("--max_projectiles", type=int, default=4)
    parser.add_argument(
        "--slot_strategy",
        choices=["temporal", "distance"],
        default="temporal",
    )
    parser.add_argument("--slot_match_distance", type=float, default=0.20)

    # PPO
    parser.add_argument("--rollout_steps", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--clip_epsilon", type=float, default=0.1)
    parser.add_argument("--value_coefficient", type=float, default=0.5)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)

    # Reward shaping preserved from the previous program
    parser.add_argument("--survival_bonus", type=float, default=0.5)
    parser.add_argument("--death_penalty", type=float, default=-5.0)
    parser.add_argument("--reward_scale", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if not 0.0 < args.alpha_decay <= 1.0:
        raise ValueError("--alpha_decay must be in (0, 1]")
    if not 0.0 <= args.alpha_min <= 1.0:
        raise ValueError("--alpha_min must be in [0, 1]")

    if args.quick:
        args.source_steps = min(args.source_steps, 30_000)
        args.target_steps = min(args.target_steps, 30_000)
        args.eval_episodes = min(args.eval_episodes, 3)
        args.rollout_steps = min(args.rollout_steps, 512)
        args.ppo_epochs = min(args.ppo_epochs, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir or f"ocatari_transfer_results_{timestamp}")
    source_dir = ensure_dir(root / "01_source")
    baseline_dir = ensure_dir(root / "02_target_no_transfer")
    transfer_dir = ensure_dir(root / "03_target_transfer")
    comparison_dir = ensure_dir(root / "04_comparison")

    device = torch.device(args.device)
    encoder_cfg = EncoderConfig(
        max_enemies=args.max_enemies,
        max_projectiles=args.max_projectiles,
        stack_size=args.stack_size,
        slot_strategy=args.slot_strategy,
        slot_match_distance=args.slot_match_distance,
    )
    ppo_cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        rollout_steps=args.rollout_steps,
        hidden_size=args.hidden_size,
    )
    reward_cfg = RewardConfig(
        survival_bonus=args.survival_bonus,
        death_penalty=args.death_penalty,
        reward_scale=args.reward_scale,
    )
    run_cfg = RunConfig(
        source_env=args.source_env,
        target_env=args.target_env,
        object_mode=args.object_mode,
        source_steps=args.source_steps,
        target_steps=args.target_steps,
        eval_episodes=args.eval_episodes,
        alpha=args.alpha,
        alpha_decay=args.alpha_decay,
        alpha_min=args.alpha_min,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        seed=args.seed,
        moving_average_window=args.moving_average_window,
        output_dir=str(root),
        device=str(device),
    )

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ocatari": module_version("ocatari"),
        "gymnasium": module_version("gymnasium"),
        "ale-py": module_version("ale-py"),
        "matplotlib": module_version("matplotlib"),
    }
    save_json(
        root / "config.json",
        {
            "run": asdict(run_cfg),
            "encoder": asdict(encoder_cfg),
            "ppo": asdict(ppo_cfg),
            "reward": asdict(reward_cfg),
            "versions": versions,
        },
    )

    print(f"Output directory: {root.resolve()}")
    print(f"Device: {device}")

    # Validate environment compatibility before the expensive runs.
    source_probe = ObjectEnv(
        args.source_env,
        encoder_cfg,
        args.object_mode,
        args.frameskip,
        args.repeat_action_probability,
    )
    target_probe = ObjectEnv(
        args.target_env,
        encoder_cfg,
        args.object_mode,
        args.frameskip,
        args.repeat_action_probability,
    )
    source_probe.reset(seed=args.seed)
    target_probe.reset(seed=args.seed)
    source_actions = int(source_probe.action_space.n)
    target_actions = int(target_probe.action_space.n)
    compatibility = {
        "source_action_count": source_actions,
        "target_action_count": target_actions,
        "source_action_meanings": source_probe.action_meanings(),
        "target_action_meanings": target_probe.action_meanings(),
        "common_observation_dimension": encoder_cfg.stack_size
        * (5 + encoder_cfg.max_enemies * 5 + encoder_cfg.max_projectiles * 5 + 6),
    }
    save_json(root / "environment_compatibility.json", compatibility)
    source_probe.close()
    target_probe.close()
    if source_actions != target_actions:
        raise ValueError(
            "Source and target action counts differ. This verification script "
            "requires an explicit action mapping before transfer. See "
            "environment_compatibility.json."
        )

    print("\n=== Phase 1/3: source training ===")
    source_result = train_agent(
        label="Source PPO",
        env_id=args.source_env,
        total_steps=args.source_steps,
        seed=args.seed,
        output_dir=source_dir,
        encoder_cfg=encoder_cfg,
        ppo_cfg=ppo_cfg,
        reward_cfg=reward_cfg,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
    )

    # Construct one shared initial target state so the only intended difference
    # between target conditions is the transfer advantage.
    set_global_seed(args.seed + 1)
    initial_target_model = ActorCriticMLP(
        encoder_cfg.obs_dim,
        target_actions,
        ppo_cfg.hidden_size,
    )
    initial_target_state = copy.deepcopy(initial_target_model.state_dict())

    print("\n=== Phase 2/3: target baseline (no transfer) ===")
    baseline_result = train_agent(
        label="Target PPO without transfer",
        env_id=args.target_env,
        total_steps=args.target_steps,
        seed=args.seed + 1,
        output_dir=baseline_dir,
        encoder_cfg=encoder_cfg,
        ppo_cfg=ppo_cfg,
        reward_cfg=reward_cfg,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
        initial_state_dict=initial_target_state,
        source_model=None,
        alpha=0.0,
    )

    print("\n=== Phase 3/3: target transfer ===")
    transfer_result = train_agent(
        label=f"Target PPO with transfer (alpha={args.alpha})",
        env_id=args.target_env,
        total_steps=args.target_steps,
        seed=args.seed + 1,
        output_dir=transfer_dir,
        encoder_cfg=encoder_cfg,
        ppo_cfg=ppo_cfg,
        reward_cfg=reward_cfg,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
        initial_state_dict=initial_target_state,
        source_model=source_result.model,
        alpha=args.alpha,
        alpha_decay=args.alpha_decay,
        alpha_min=args.alpha_min,
    )

    print("\n=== Deterministic evaluation ===")
    source_eval = evaluate_agent(
        label="Source",
        model=source_result.model,
        env_id=args.source_env,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        output_dir=source_dir,
        encoder_cfg=encoder_cfg,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
    )
    baseline_eval = evaluate_agent(
        label="Target no transfer",
        model=baseline_result.model,
        env_id=args.target_env,
        episodes=args.eval_episodes,
        seed=args.seed + 20_000,
        output_dir=baseline_dir,
        encoder_cfg=encoder_cfg,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
    )
    transfer_eval = evaluate_agent(
        label="Target transfer",
        model=transfer_result.model,
        env_id=args.target_env,
        episodes=args.eval_episodes,
        seed=args.seed + 20_000,
        output_dir=transfer_dir,
        encoder_cfg=encoder_cfg,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
    )

    plot_target_comparison(
        baseline_result.episode_rows,
        transfer_result.episode_rows,
        comparison_dir,
        args.moving_average_window,
    )
    plot_transfer_diagnostics(transfer_result.diagnostic_rows, comparison_dir)
    plot_evaluation_comparison(baseline_eval, transfer_eval, comparison_dir)

    summary = {
        "source": source_eval,
        "target_no_transfer": baseline_eval,
        "target_transfer": transfer_eval,
        "transfer_minus_baseline_mean_return": (
            transfer_eval["raw_return_mean"] - baseline_eval["raw_return_mean"]
        ),
        "output_directory": str(root.resolve()),
    }
    save_json(root / "summary.json", summary)

    print("\n=== Completed ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nMain comparison graphs:")
    for filename in (
        "target_raw_return_comparison.png",
        "target_shaped_return_comparison.png",
        "target_episode_length_comparison.png",
        "target_evaluation_comparison.png",
        "transfer_advantage_means.png",
        "transfer_advantage_std.png",
        "transfer_alpha_schedule.png",
    ):
        print(f"  {comparison_dir / filename}")


if __name__ == "__main__":
    main()
