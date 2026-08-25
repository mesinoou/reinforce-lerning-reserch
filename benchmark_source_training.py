#!/usr/bin/env python3
"""Benchmark object-list or pixel PPO throughput.

The benchmark executes the same environment interaction, observation encoding,
policy inference, GAE, and PPO optimization used by the source trainer. It then
converts measured wall-clock throughput into:

- environment steps achievable within user-specified training hours
- wall-clock time required for user-specified target step counts

Periodic evaluation and checkpoint I/O are excluded. A configurable reserve
fraction provides a conservative projection for those real-run overheads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import numpy as np
import torch
import torch.optim as optim

from ocatari_source_ppo import (
    ObjectDiagnostics,
    ObservationStatistics,
    PixelConfig,
    PPOConfig,
    RewardConfig,
    RolloutBuffer,
    count_ale_frames_advanced,
    format_duration,
    maybe_fire_after_reset,
    make_actor_critic,
    make_observation_env,
    model_parameter_count,
    ppo_update,
    reward_components,
    set_global_seed,
)
from ocatari_transfer_full_experiment import (
    EncoderConfig,
    module_version,
    save_json,
)


@dataclass
class EnvironmentState:
    observation: np.ndarray
    info: Dict[str, Any]
    previous_lives: Optional[int]
    episode_length: int = 0
    episodes_completed: int = 0


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def one_training_update(
    *,
    env: Any,
    state: EnvironmentState,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    encoder_config: EncoderConfig,
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    args: argparse.Namespace,
    device: torch.device,
    observation_statistics: Optional[ObservationStatistics],
    object_diagnostics: Optional[ObjectDiagnostics],
) -> Tuple[EnvironmentState, Dict[str, float]]:
    buffer.reset()
    ale_frames = 0
    last_learning_terminal = False

    synchronize(device)
    collection_start = time.perf_counter()
    for _step in range(ppo_config.rollout_steps):
        obs = state.observation
        if observation_statistics is not None:
            observation_statistics.update(
                obs,
                args.stack_size,
                summarize_pixels=args.input_mode == "pixels",
            )
        if object_diagnostics is not None:
            object_diagnostics.update(env)

        observation_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = model(observation_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action_tensor = distribution.sample()
            log_probability = distribution.log_prob(action_tensor)
        action = int(action_tensor.item())

        next_obs, raw_reward, environment_done, next_info = env.step(action)
        frames_advanced = count_ale_frames_advanced(
            state.info,
            next_info,
            args.frameskip,
        )
        ale_frames += frames_advanced
        next_lives_value = next_info.get("lives", state.previous_lives)
        next_lives = (
            int(next_lives_value)
            if next_lives_value is not None
            else None
        )
        life_lost = bool(
            state.previous_lives is not None
            and next_lives is not None
            and next_lives < state.previous_lives
        )
        max_length_reached = (
            state.episode_length + 1 >= args.max_episode_steps
        )
        episode_done = environment_done or max_length_reached
        learning_terminal = episode_done or (
            args.life_loss_terminal and life_lost
        )
        components = reward_components(
            raw_reward,
            life_lost,
            reward_config,
            frames_advanced,
        )
        buffer.add(
            obs,
            action,
            components.total,
            learning_terminal,
            float(log_probability.item()),
            float(value.item()),
            raw_reward=raw_reward,
            score_component=components.score,
            survival_component=components.survival,
            life_loss_component=components.life_loss,
            ale_frames_advanced=frames_advanced,
        )

        state.observation = next_obs
        state.info = next_info
        state.previous_lives = next_lives
        state.episode_length += 1
        last_learning_terminal = learning_terminal

        if episode_done:
            state.episodes_completed += 1
            obs, info = env.reset()
            obs, info = maybe_fire_after_reset(
                env,
                obs,
                info,
                args.auto_fire_reset,
            )
            lives_value = info.get("lives")
            state = EnvironmentState(
                observation=obs,
                info=info,
                previous_lives=(
                    int(lives_value)
                    if lives_value is not None
                    else None
                ),
                episode_length=0,
                episodes_completed=state.episodes_completed,
            )
    synchronize(device)
    collection_seconds = time.perf_counter() - collection_start

    synchronize(device)
    optimization_start = time.perf_counter()
    if last_learning_terminal:
        last_value = 0.0
    else:
        with torch.no_grad():
            _logits, last_value_tensor = model(
                torch.from_numpy(state.observation).unsqueeze(0).to(device)
            )
            last_value = float(last_value_tensor.item())
    advantages, returns = buffer.compute_gae(
        last_value,
        ppo_config.gamma,
        ppo_config.gae_lambda,
    )
    losses = ppo_update(
        model,
        optimizer,
        torch.from_numpy(buffer.obs[: buffer.ptr]).to(device),
        torch.from_numpy(buffer.actions[: buffer.ptr]).to(device),
        torch.from_numpy(buffer.log_probs[: buffer.ptr]).to(device),
        torch.from_numpy(buffer.values[: buffer.ptr]).to(device),
        torch.from_numpy(advantages).to(device),
        torch.from_numpy(returns).to(device),
        ppo_config,
    )
    synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_start

    return state, {
        "environment_steps": float(buffer.ptr),
        "ale_frames": float(ale_frames),
        "collection_seconds": collection_seconds,
        "optimization_seconds": optimization_seconds,
        "total_update_seconds": collection_seconds + optimization_seconds,
        **losses,
    }


def projection_rows(
    steps_per_second: float,
    reserve_fraction: float,
    project_hours: List[float],
    target_steps: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    conservative_sps = steps_per_second * (1.0 - reserve_fraction)
    by_time = []
    for hours in project_hours:
        seconds = hours * 3_600.0
        by_time.append(
            {
                "hours": hours,
                "measured_projection_env_steps": int(
                    math.floor(steps_per_second * seconds)
                ),
                "conservative_projection_env_steps": int(
                    math.floor(conservative_sps * seconds)
                ),
            }
        )
    by_steps = []
    for steps in target_steps:
        measured_seconds = steps / steps_per_second
        conservative_seconds = steps / conservative_sps
        by_steps.append(
            {
                "target_env_steps": steps,
                "measured_estimated_seconds": measured_seconds,
                "measured_estimated_time": format_duration(measured_seconds),
                "conservative_estimated_seconds": conservative_seconds,
                "conservative_estimated_time": format_duration(
                    conservative_seconds
                ),
            }
        )
    return by_time, by_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark object-list or pixel PPO throughput and convert training "
            "time to environment steps."
        )
    )
    parser.add_argument("--env", default="ALE/SpaceInvaders-v5")
    parser.add_argument(
        "--input-mode",
        choices=["objects", "pixels"],
        default="objects",
    )
    parser.add_argument("--object-mode", choices=["ram", "vision"], default="ram")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.10,
        help="Fraction reserved for evaluation/checkpoint overhead.",
    )
    parser.add_argument(
        "--project-hours",
        nargs="+",
        type=float,
        default=[1.0, 6.0, 12.0, 24.0, 48.0],
    )
    parser.add_argument(
        "--target-steps",
        nargs="+",
        type=int,
        default=[500_000, 1_000_000, 5_000_000],
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--pixel-width", type=int, default=84)
    parser.add_argument("--pixel-height", type=int, default=84)
    parser.add_argument("--max-enemies", type=int, default=12)
    parser.add_argument("--max-projectiles", type=int, default=4)
    parser.add_argument(
        "--slot-strategy",
        choices=["temporal", "distance"],
        default="temporal",
    )
    parser.add_argument("--slot-match-distance", type=float, default=0.20)
    parser.add_argument("--max-episode-steps", type=int, default=27_000)
    parser.add_argument(
        "--life-loss-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--auto-fire-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--value-clip-epsilon", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument(
        "--reward-mode",
        choices=["raw", "clipped", "scaled_raw", "scaled_survival", "shaped"],
        default="scaled_raw",
    )
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--survival-reward-per-frame", type=float, default=0.001)
    parser.add_argument("--life-loss-penalty", type=float, default=-1.0)

    args = parser.parse_args()
    if args.quick:
        args.benchmark_seconds = min(args.benchmark_seconds, 2.0)
        args.warmup_updates = 0
        args.rollout_steps = min(args.rollout_steps, 128)
        args.ppo_epochs = min(args.ppo_epochs, 1)
        args.minibatch_size = min(args.minibatch_size, 64)
    if args.benchmark_seconds <= 0.0:
        parser.error("--benchmark-seconds must be positive")
    if args.warmup_updates < 0:
        parser.error("--warmup-updates must be non-negative")
    if not 0.0 <= args.reserve_fraction < 1.0:
        parser.error("--reserve-fraction must be in [0, 1)")
    if args.rollout_steps <= 0 or args.ppo_epochs <= 0 or args.minibatch_size <= 0:
        parser.error("rollout, epoch, and minibatch values must be positive")
    if any(hours <= 0.0 for hours in args.project_hours):
        parser.error("--project-hours values must be positive")
    if any(steps <= 0 for steps in args.target_steps):
        parser.error("--target-steps values must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_global_seed(args.seed, deterministic_torch=False)

    encoder_config = EncoderConfig(
        max_enemies=args.max_enemies,
        max_projectiles=args.max_projectiles,
        stack_size=args.stack_size,
        slot_strategy=args.slot_strategy,
        slot_match_distance=args.slot_match_distance,
    )
    pixel_config = PixelConfig(
        width=args.pixel_width,
        height=args.pixel_height,
        stack_size=args.stack_size,
    )
    ppo_config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        anneal_learning_rate=False,
        clip_epsilon=args.clip_epsilon,
        value_clip_epsilon=args.value_clip_epsilon,
        value_clipping=True,
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
    )
    env = make_observation_env(
        input_mode=args.input_mode,
        env_id=args.env,
        encoder_config=encoder_config,
        object_mode=args.object_mode,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        pixel_config=pixel_config,
    )
    try:
        observation, info = env.reset(seed=args.seed)
        observation, info = maybe_fire_after_reset(
            env,
            observation,
            info,
            args.auto_fire_reset,
        )
        lives_value = info.get("lives")
        state = EnvironmentState(
            observation=observation,
            info=info,
            previous_lives=(
                int(lives_value)
                if lives_value is not None
                else None
            ),
        )
        observation_shape = tuple(
            int(value)
            for value in getattr(env, "obs_shape", (env.obs_dim,))
        )
        model = make_actor_critic(
            input_mode=args.input_mode,
            observation_shape=observation_shape,
            n_actions=int(env.action_space.n),
            hidden_size=ppo_config.hidden_size,
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=ppo_config.learning_rate,
            eps=1e-5,
        )
        buffer = RolloutBuffer(ppo_config.rollout_steps, observation_shape)
        observation_statistics = (
            ObservationStatistics(
                env.obs_dim
                if args.input_mode == "objects"
                else args.stack_size * 5
            )
            if args.include_diagnostics
            else None
        )
        object_diagnostics = (
            ObjectDiagnostics() if args.include_diagnostics else None
        )

        print(
            f"Benchmark input={args.input_mode} device={device} "
            f"rollout={ppo_config.rollout_steps} "
            f"epochs={ppo_config.ppo_epochs} diagnostics={args.include_diagnostics}"
        )
        for warmup_index in range(args.warmup_updates):
            state, _metrics = one_training_update(
                env=env,
                state=state,
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                encoder_config=encoder_config,
                ppo_config=ppo_config,
                reward_config=reward_config,
                args=args,
                device=device,
                observation_statistics=observation_statistics,
                object_diagnostics=object_diagnostics,
            )
            print(
                f"[warmup] {warmup_index + 1}/{args.warmup_updates} complete"
            )

        synchronize(device)
        benchmark_start = time.perf_counter()
        measured_steps = 0
        measured_ale_frames = 0
        collection_seconds = 0.0
        optimization_seconds = 0.0
        updates = 0
        update_rows: List[Dict[str, float]] = []
        while True:
            state, metrics = one_training_update(
                env=env,
                state=state,
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                encoder_config=encoder_config,
                ppo_config=ppo_config,
                reward_config=reward_config,
                args=args,
                device=device,
                observation_statistics=observation_statistics,
                object_diagnostics=object_diagnostics,
            )
            updates += 1
            measured_steps += int(metrics["environment_steps"])
            measured_ale_frames += int(metrics["ale_frames"])
            collection_seconds += metrics["collection_seconds"]
            optimization_seconds += metrics["optimization_seconds"]
            update_rows.append(metrics)
            synchronize(device)
            elapsed = time.perf_counter() - benchmark_start
            print(
                f"[benchmark] updates={updates} steps={measured_steps:,} "
                f"elapsed={format_duration(elapsed)} "
                f"speed={measured_steps / elapsed:.1f} steps/s"
            )
            if elapsed >= args.benchmark_seconds:
                break

        synchronize(device)
        measured_seconds = time.perf_counter() - benchmark_start
        steps_per_second = measured_steps / measured_seconds
        ale_frames_per_second = measured_ale_frames / measured_seconds
        by_time, by_steps = projection_rows(
            steps_per_second,
            args.reserve_fraction,
            args.project_hours,
            args.target_steps,
        )
        result = {
            "benchmark": {
                "requested_seconds": args.benchmark_seconds,
                "measured_seconds": measured_seconds,
                "measured_time": format_duration(measured_seconds),
                "warmup_updates": args.warmup_updates,
                "measured_updates": updates,
                "measured_environment_steps": measured_steps,
                "measured_ale_frames": measured_ale_frames,
                "environment_steps_per_second": steps_per_second,
                "ale_frames_per_second": ale_frames_per_second,
                "collection_seconds": collection_seconds,
                "optimization_seconds": optimization_seconds,
                "collection_time_fraction": (
                    collection_seconds / measured_seconds
                ),
                "optimization_time_fraction": (
                    optimization_seconds / measured_seconds
                ),
                "episodes_completed": state.episodes_completed,
            },
            "projection": {
                "reserve_fraction": args.reserve_fraction,
                "conservative_environment_steps_per_second": (
                    steps_per_second * (1.0 - args.reserve_fraction)
                ),
                "steps_by_training_time": by_time,
                "time_by_target_steps": by_steps,
            },
            "configuration": {
                "arguments": vars(args),
                "input_mode": args.input_mode,
                "model_type": (
                    "cnn" if args.input_mode == "pixels" else "mlp"
                ),
                "encoder": asdict(encoder_config),
                "pixels": asdict(pixel_config),
                "ppo": asdict(ppo_config),
                "reward": asdict(reward_config),
                "observation_dimension": env.obs_dim,
                "observation_shape": list(observation_shape),
                "model_parameter_count": model_parameter_count(model),
                "action_count": int(env.action_space.n),
                "action_meanings": env.action_meanings(),
            },
            "system": {
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "ocatari": module_version("ocatari"),
                "gymnasium": module_version("gymnasium"),
                "ale-py": module_version("ale-py"),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            },
            "notes": [
                "Includes the selected observation representation, policy inference, GAE, PPO optimization, and optional in-memory diagnostics.",
                "Excludes periodic evaluation, checkpoint writes, plotting, and other disk I/O.",
                "Use the conservative projection to reserve time for excluded overhead.",
                "Run this benchmark on the same Linux PC, CUDA device, and PPO configuration as the planned training.",
            ],
        }
        output_path = Path(
            args.output
            or (
                "benchmark_source_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".json"
            )
        )
        save_json(output_path, result)

        print("\nProjected environment steps by training time:")
        for row in by_time:
            print(
                f"  {row['hours']:>6g} h: "
                f"{row['measured_projection_env_steps']:,} measured / "
                f"{row['conservative_projection_env_steps']:,} conservative"
            )
        print("\nEstimated time by target steps:")
        for row in by_steps:
            print(
                f"  {row['target_env_steps']:>10,} steps: "
                f"{row['measured_estimated_time']} measured / "
                f"{row['conservative_estimated_time']} conservative"
            )
        print(f"\nSaved: {output_path.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
