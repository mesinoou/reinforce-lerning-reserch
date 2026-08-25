#!/usr/bin/env python3
"""Render MP4 videos and representative play images from a trained model.

Accepted model files:

- ``model.pt``: final source model
- ``checkpoint_best.pt``: best periodic-evaluation model
- ``checkpoint_latest.pt``: latest resumable model

The renderer reconstructs the object encoder and ActorCriticMLP from checkpoint
metadata, runs fixed-seed episodes, and saves representative frames, a contact
sheet, an MP4 video, a step-level trajectory, and provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from ocatari_source_ppo import maybe_fire_after_reset
from ocatari_transfer_full_experiment import (
    ActorCriticMLP,
    CommonObjectEncoder,
    EncoderConfig,
    ObjectEnv,
    ensure_dir,
    save_csv,
    save_json,
)


TRAJECTORY_FIELDS = [
    "episode",
    "episode_seed",
    "step",
    "action",
    "action_meaning",
    "raw_reward",
    "cumulative_raw_return",
    "value",
    "policy_entropy",
    "selected_action_probability",
    "maximum_action_probability",
    "lives",
    "done",
]


@dataclass
class LoadedModel:
    model: ActorCriticMLP
    payload: Dict[str, Any]
    encoder_config: EncoderConfig
    observation_dimension: int
    runtime_config: Dict[str, Any]
    env_id: str
    object_mode: str
    frameskip: int
    repeat_action_probability: float
    max_episode_steps: int
    auto_fire_reset: bool
    n_actions: int
    action_meanings: List[str]
    source_steps: Optional[int]


@dataclass
class CapturedFrame:
    step: int
    cumulative_raw_return: float
    action_meaning: str
    image: Image.Image


class RepresentativeFrameReservoir:
    """Bound memory while sampling frames across the full episode."""

    def __init__(self, capacity: int, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.first: Optional[CapturedFrame] = None
        self.last: Optional[CapturedFrame] = None
        self.middle: List[CapturedFrame] = []
        self.middle_seen = 0

    def add(self, capture: CapturedFrame) -> None:
        if self.first is None:
            self.first = capture
            self.last = capture
            return
        self.last = capture
        middle_capacity = max(0, self.capacity - 2)
        if middle_capacity == 0:
            return
        self.middle_seen += 1
        if len(self.middle) < middle_capacity:
            self.middle.append(capture)
            return
        replacement_index = self.rng.randrange(self.middle_seen)
        if replacement_index < middle_capacity:
            self.middle[replacement_index] = capture

    def selected(self) -> List[CapturedFrame]:
        candidates: List[CapturedFrame] = []
        if self.capacity == 1:
            if self.last is not None:
                candidates.append(self.last)
        else:
            if self.first is not None:
                candidates.append(self.first)
            candidates.extend(self.middle)
            if self.last is not None:
                candidates.append(self.last)
        unique_by_step = {capture.step: capture for capture in candidates}
        return [
            unique_by_step[step]
            for step in sorted(unique_by_step)
        ][: self.capacity]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_optional_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model_metadata(
    payload: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    runtime = dict(config.get("arguments", {}))
    runtime.update(payload.get("configuration", {}))
    encoder_data = payload.get("encoder_config") or config.get("encoder")
    ppo_data = payload.get("ppo_config") or config.get("ppo")
    if not encoder_data:
        raise ValueError(
            "Encoder configuration is missing from both model and config.json"
        )
    if not ppo_data:
        raise ValueError(
            "PPO configuration is missing from both model and config.json"
        )
    env_id = args.env or payload.get("env_id") or runtime.get("env")
    if not env_id:
        raise ValueError("Environment ID is missing; pass --env explicitly")
    n_actions_value = payload.get("n_actions")
    if n_actions_value is None:
        raise ValueError("n_actions is missing from the model file")
    return {
        "runtime": runtime,
        "encoder_data": encoder_data,
        "ppo_data": ppo_data,
        "env_id": str(env_id),
        "object_mode": args.object_mode or runtime.get("object_mode", "ram"),
        "frameskip": (
            args.frameskip
            if args.frameskip > 0
            else int(runtime.get("frameskip", 4))
        ),
        "repeat_action_probability": (
            args.repeat_action_probability
            if args.repeat_action_probability >= 0.0
            else float(runtime.get("repeat_action_probability", 0.0))
        ),
        "max_episode_steps": (
            args.max_episode_steps
            if args.max_episode_steps > 0
            else int(runtime.get("max_episode_steps", 27_000))
        ),
        "auto_fire_reset": (
            args.auto_fire_reset
            if args.auto_fire_reset is not None
            else bool(runtime.get("auto_fire_reset", False))
        ),
        "n_actions": int(n_actions_value),
        "action_meanings": list(payload.get("action_meanings", [])),
        "source_steps": payload.get(
            "steps",
            payload.get("environment_steps"),
        ),
    }


def load_trained_model(
    model_path: Path,
    config_path: Optional[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> LoadedModel:
    payload = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in payload:
        raise ValueError(f"model_state_dict is missing from {model_path}")
    config = read_optional_config(config_path)
    metadata = resolve_model_metadata(payload, config, args)
    encoder_config = EncoderConfig(**metadata["encoder_data"])
    observation_dimension = CommonObjectEncoder(encoder_config).obs_dim
    hidden_size = int(metadata["ppo_data"].get("hidden_size", 256))
    model = ActorCriticMLP(
        observation_dimension,
        metadata["n_actions"],
        hidden_size,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return LoadedModel(
        model=model,
        payload=payload,
        encoder_config=encoder_config,
        observation_dimension=observation_dimension,
        runtime_config=metadata["runtime"],
        env_id=metadata["env_id"],
        object_mode=metadata["object_mode"],
        frameskip=metadata["frameskip"],
        repeat_action_probability=metadata[
            "repeat_action_probability"
        ],
        max_episode_steps=metadata["max_episode_steps"],
        auto_fire_reset=metadata["auto_fire_reset"],
        n_actions=metadata["n_actions"],
        action_meanings=metadata["action_meanings"],
        source_steps=(
            int(metadata["source_steps"])
            if metadata["source_steps"] is not None
            else None
        ),
    )


def object_color(env: ObjectEnv, category: str) -> Tuple[int, int, int]:
    normalized = category.replace("_", "").lower()
    if normalized in env.encoder.PLAYER_NAMES:
        return (50, 255, 80)
    if env.encoder._is_enemy(normalized):
        return (255, 70, 70)
    if env.encoder._is_projectile(normalized):
        return (255, 230, 40)
    return (60, 210, 255)


def render_current_frame(
    env: ObjectEnv,
    *,
    overlay_objects: bool,
    scale: int,
) -> Image.Image:
    frame_array = np.asarray(env.rgb_frame(), dtype=np.uint8)
    image = Image.fromarray(frame_array, mode="RGB")
    image = image.resize(
        (image.width * scale, image.height * scale),
        resample=Image.Resampling.NEAREST,
    )
    if not overlay_objects:
        return image
    draw = ImageDraw.Draw(image)
    for obj in env.objects:
        try:
            if not bool(obj):
                continue
        except Exception:
            pass
        try:
            x, y, width, height = env.encoder._xywh(obj)
            category = str(
                getattr(obj, "category", obj.__class__.__name__)
            )
            color = object_color(env, category)
            box = (
                int(round(x * scale)),
                int(round(y * scale)),
                int(round((x + width) * scale)),
                int(round((y + height) * scale)),
            )
            draw.rectangle(box, outline=color, width=max(1, scale))
            draw.text(
                (box[0], max(0, box[1] - 10)),
                category,
                fill=color,
            )
        except Exception:
            continue
    return image


def make_contact_sheet(
    captures: Sequence[CapturedFrame],
    *,
    columns: int,
    title: str,
) -> Image.Image:
    if not captures:
        raise ValueError("No frames were captured")
    columns = max(1, min(columns, len(captures)))
    rows = math.ceil(len(captures) / columns)
    frame_width, frame_height = captures[0].image.size
    label_height = 28
    header_height = 56
    tile_width = frame_width
    tile_height = frame_height + label_height
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * tile_height),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill=(255, 255, 255))
    for index, capture in enumerate(captures):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = header_height + row * tile_height
        label = (
            f"step={capture.step:,} score={capture.cumulative_raw_return:g} "
            f"action={capture.action_meaning}"
        )
        draw.text((x + 4, y + 4), label, fill=(235, 235, 235))
        sheet.paste(capture.image, (x, y + label_height))
    return sheet


def action_meaning(meanings: Sequence[str], action: int) -> str:
    if 0 <= action < len(meanings):
        return meanings[action]
    return str(action)


def open_video_writer(path: Path, fps: float) -> Any:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "MP4 output requires imageio and imageio-ffmpeg. "
            "Install requirements_ocatari_transfer.txt, or use "
            "--no-save-video."
        ) from error
    return imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="warning",
    )


@torch.no_grad()
def render_episode(
    *,
    loaded: LoadedModel,
    episode_number: int,
    episode_seed: int,
    policy: str,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)

    env = ObjectEnv(
        loaded.env_id,
        loaded.encoder_config,
        loaded.object_mode,
        loaded.frameskip,
        loaded.repeat_action_probability,
    )
    trajectory_rows: List[Dict[str, Any]] = []
    reservoir = RepresentativeFrameReservoir(
        args.max_representative_frames,
        episode_seed,
    )
    episode_prefix = f"episode_{episode_number:03d}"
    video_path = output_dir / f"{episode_prefix}.mp4"
    video_fps = (
        args.video_fps
        if args.video_fps > 0.0
        else 60.0 / loaded.frameskip
    )
    video_writer = None
    try:
        obs, info = env.reset(seed=episode_seed)
        obs, info = maybe_fire_after_reset(
            env,
            obs,
            info,
            loaded.auto_fire_reset,
        )
        env_action_meanings = env.action_meanings()
        if int(env.action_space.n) != loaded.n_actions:
            raise ValueError(
                "Action-space mismatch: "
                f"model={loaded.n_actions}, environment={env.action_space.n}"
            )
        initial_frame = render_current_frame(
            env,
            overlay_objects=args.overlay_objects,
            scale=args.scale,
        )
        reservoir.add(
            CapturedFrame(
                step=0,
                cumulative_raw_return=0.0,
                action_meaning="RESET",
                image=initial_frame,
            )
        )
        if args.save_video:
            video_writer = open_video_writer(video_path, video_fps)
            video_writer.append_data(np.asarray(initial_frame))

        cumulative_raw_return = 0.0
        step = 0
        done = False
        last_action_meaning = "RESET"
        while not done and step < loaded.max_episode_steps:
            observation_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            logits, value = loaded.model(observation_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            probabilities = distribution.probs.squeeze(0)
            if policy == "deterministic":
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(distribution.sample().item())
            selected_probability = float(probabilities[action].cpu())
            maximum_probability = float(probabilities.max().cpu())
            entropy = float(distribution.entropy().item())
            last_action_meaning = action_meaning(
                env_action_meanings,
                action,
            )

            obs, raw_reward, done, info = env.step(action)
            step += 1
            cumulative_raw_return += raw_reward
            trajectory_rows.append(
                {
                    "episode": episode_number,
                    "episode_seed": episode_seed,
                    "step": step,
                    "action": action,
                    "action_meaning": last_action_meaning,
                    "raw_reward": raw_reward,
                    "cumulative_raw_return": cumulative_raw_return,
                    "value": float(value.item()),
                    "policy_entropy": entropy,
                    "selected_action_probability": selected_probability,
                    "maximum_action_probability": maximum_probability,
                    "lives": info.get("lives"),
                    "done": done,
                }
            )
            should_capture_representative = (
                step % args.capture_every == 0 or done
            )
            current_frame = None
            if args.save_video or should_capture_representative:
                current_frame = render_current_frame(
                    env,
                    overlay_objects=args.overlay_objects,
                    scale=args.scale,
                )
            if video_writer is not None:
                video_writer.append_data(np.asarray(current_frame))
            if should_capture_representative:
                reservoir.add(
                    CapturedFrame(
                        step=step,
                        cumulative_raw_return=cumulative_raw_return,
                        action_meaning=last_action_meaning,
                        image=current_frame,
                    )
                )

        if reservoir.last is None or reservoir.last.step != step:
            reservoir.add(
                CapturedFrame(
                    step=step,
                    cumulative_raw_return=cumulative_raw_return,
                    action_meaning=last_action_meaning,
                    image=render_current_frame(
                        env,
                        overlay_objects=args.overlay_objects,
                        scale=args.scale,
                    ),
                )
            )
        captures = reservoir.selected()
        frames_dir = ensure_dir(output_dir / f"{episode_prefix}_frames")
        frame_paths: List[str] = []
        if args.save_representative_frames:
            for capture in captures:
                frame_path = frames_dir / f"step_{capture.step:06d}.png"
                capture.image.save(frame_path)
                frame_paths.append(str(frame_path.resolve()))

        final_frame_path = output_dir / f"{episode_prefix}_final.png"
        if reservoir.last is None:
            raise RuntimeError("Final frame was not captured")
        reservoir.last.image.save(final_frame_path)

        contact_sheet_path = output_dir / f"{episode_prefix}_contact_sheet.png"
        sheet = make_contact_sheet(
            captures,
            columns=args.columns,
            title=(
                f"{loaded.env_id} | {policy} | seed={episode_seed} | "
                f"return={cumulative_raw_return:g} | steps={step:,}"
            ),
        )
        sheet.save(contact_sheet_path)
        trajectory_path = output_dir / f"{episode_prefix}_trajectory.csv"
        save_csv(
            trajectory_path,
            trajectory_rows,
            TRAJECTORY_FIELDS,
        )
        return {
            "episode": episode_number,
            "seed": episode_seed,
            "policy": policy,
            "raw_return": cumulative_raw_return,
            "episode_length": step,
            "environment_terminated": done,
            "renderer_step_limit_reached": (
                not done and step >= loaded.max_episode_steps
            ),
            "representative_frame_count": len(captures),
            "representative_frames": frame_paths,
            "video": (
                str(video_path.resolve())
                if args.save_video
                else None
            ),
            "video_fps": video_fps if args.save_video else None,
            "final_frame": str(final_frame_path.resolve()),
            "contact_sheet": str(contact_sheet_path.resolve()),
            "trajectory": str(trajectory_path.resolve()),
        }
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render MP4 videos and play images from a trained "
            "OCAtari object PPO model."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="Defaults to config.json next to the model.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument(
        "--policy",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--capture-every", type=int, default=4)
    parser.add_argument("--max-representative-frames", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.0,
        help=(
            "MP4 frames per second. The default (0) uses 60 / frameskip."
        ),
    )
    parser.add_argument(
        "--overlay-objects",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-representative-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Optional overrides. Sentinel values mean "use saved configuration".
    parser.add_argument("--env", default="")
    parser.add_argument("--object-mode", choices=["ram", "vision"], default="")
    parser.add_argument("--frameskip", type=int, default=0)
    parser.add_argument("--repeat-action-probability", type=float, default=-1.0)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument(
        "--auto-fire-reset",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.capture_every <= 0:
        parser.error("--capture-every must be positive")
    if args.max_representative_frames <= 0:
        parser.error("--max-representative-frames must be positive")
    if args.columns <= 0 or args.scale <= 0:
        parser.error("--columns and --scale must be positive")
    if args.video_fps < 0.0:
        parser.error("--video-fps must be non-negative")
    if args.repeat_action_probability > 1.0:
        parser.error("--repeat-action-probability must be <= 1")
    return args


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    config_path = (
        Path(args.config).resolve()
        if args.config
        else model_path.parent / "config.json"
    )
    if not config_path.exists():
        config_path = None
    output_dir = ensure_dir(
        Path(
            args.output_dir
            or (
                model_path.parent
                / (
                    "playback_"
                    + model_path.stem
                    + "_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            )
        )
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    loaded = load_trained_model(
        model_path,
        config_path,
        args,
        device,
    )
    if loaded.object_mode != "ram":
        print(
            "Warning: loaded object mode is not REM/ram.",
            file=sys.stderr,
        )

    print(f"Model: {model_path}")
    print(f"Environment: {loaded.env_id}")
    print(
        f"Object mode: {loaded.object_mode}, "
        f"observation dimension: {loaded.observation_dimension}"
    )
    print(
        f"Policy: {args.policy}, episodes: {args.episodes}, "
        f"output: {output_dir.resolve()}"
    )
    episode_summaries = []
    for episode_index in range(args.episodes):
        episode_summary = render_episode(
            loaded=loaded,
            episode_number=episode_index + 1,
            episode_seed=args.seed + episode_index,
            policy=args.policy,
            output_dir=output_dir,
            args=args,
            device=device,
        )
        episode_summaries.append(episode_summary)
        print(
            f"[episode {episode_index + 1}] "
            f"return={episode_summary['raw_return']:.2f} "
            f"steps={episode_summary['episode_length']:,} "
            f"video={episode_summary['video']} "
            f"sheet={episode_summary['contact_sheet']}"
        )

    returns = np.asarray(
        [summary["raw_return"] for summary in episode_summaries],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [summary["episode_length"] for summary in episode_summaries],
        dtype=np.float64,
    )
    summary = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "config_path": str(config_path) if config_path is not None else None,
        "source_training_steps": loaded.source_steps,
        "environment": loaded.env_id,
        "object_mode": loaded.object_mode,
        "frameskip": loaded.frameskip,
        "repeat_action_probability": loaded.repeat_action_probability,
        "observation_dimension": loaded.observation_dimension,
        "action_count": loaded.n_actions,
        "saved_action_meanings": loaded.action_meanings,
        "policy": args.policy,
        "base_seed": args.seed,
        "episodes": args.episodes,
        "raw_return_mean": float(returns.mean()),
        "raw_return_std": float(returns.std(ddof=0)),
        "episode_length_mean": float(lengths.mean()),
        "overlay_objects": args.overlay_objects,
        "capture_every_environment_steps": args.capture_every,
        "save_video": args.save_video,
        "video_fps": (
            episode_summaries[0]["video_fps"]
            if episode_summaries
            else None
        ),
        "episode_results": episode_summaries,
    }
    summary_path = output_dir / "playback_summary.json"
    save_json(summary_path, summary)
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
