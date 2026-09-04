"""Frozen source-critic, raw Advantage mixing (no new transfer algorithm)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ENVIRONMENT_KEYS = (
    "object_mode", "frameskip", "repeat_action_probability",
    "life_loss_terminal", "auto_fire_reset", "max_episode_steps",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def read_source_checkpoint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Source model does not exist: {path}; specify model.pt, not its directory")
    # Only use trusted local research checkpoints: training checkpoints contain
    # NumPy/Python RNG objects that cannot be loaded with weights_only=True.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Expected an OCAtari model/checkpoint with model_state_dict")
    config_path = path.parent / "config.json"
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = dict(config.get("arguments", {}))
    runtime.update(payload.get("configuration", {}))
    result = {
        "payload": payload,
        "runtime": runtime,
        "encoder": payload.get("encoder_config") or config.get("encoder"),
        "pixels": payload.get("pixel_config") or config.get("pixels") or {},
        "ppo": payload.get("ppo_config") or config.get("ppo"),
        "reward": payload.get("reward_config") or config.get("reward"),
        "input_mode": payload.get("input_mode", runtime.get("input_mode", "objects")),
        "path": str(path),
        "sha256": file_sha256(path),
    }
    if any(not result[key] for key in ("encoder", "ppo", "reward")):
        raise ValueError("Source encoder/PPO/reward metadata missing; keep its original config.json beside the model")
    missing = [key for key in ENVIRONMENT_KEYS if key not in runtime]
    if missing:
        raise ValueError(f"Source runtime settings missing: {missing}; keep the source config.json beside model.pt")
    return result


def source_argument_defaults(source: dict[str, Any]) -> dict[str, Any]:
    """Inherit source preprocessing, reward and PPO, not source run identity."""
    defaults = {key: source["runtime"][key] for key in ENVIRONMENT_KEYS}
    defaults.update({
        key: value for key, value in source["encoder"].items()
        if key not in ("screen_width", "screen_height")
    })
    defaults.update(source["ppo"])
    defaults["target_kl"] = defaults.get("target_kl") or 0.0
    defaults.update({
        ("reward_mode" if key == "mode" else key): value
        for key, value in source["reward"].items()
    })
    defaults["input_mode"] = source["input_mode"]
    for key in ("width", "height"):
        if key in source["pixels"]:
            defaults[f"pixel_{key}"] = source["pixels"][key]
    return defaults


def validate_source_settings(source, args, encoder, pixels, reward) -> None:
    mismatches = []
    if source["input_mode"] != args.input_mode:
        mismatches.append("input_mode")
    if source["encoder"] != encoder:
        mismatches.append("encoder_config")
    if args.input_mode == "pixels" and source["pixels"] != pixels:
        mismatches.append("pixel_config")
    # Keep the same reward units for this raw-mixing experiment.
    if source["reward"] != reward:
        mismatches.append("reward_config")
    if float(source["ppo"]["gamma"]) != args.gamma:
        mismatches.append("gamma")
    for key in ENVIRONMENT_KEYS:
        if key == "max_episode_steps":
            continue  # May be shortened for a local execution smoke test.
        if source["runtime"][key] != getattr(args, key):
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"Source/target configuration mismatch: {mismatches}. Use ocatari_target_transfer.py to inherit source settings.")


def protect_output_directory(output: Path, source_path: Path, resume: str) -> None:
    output = output.resolve()
    source_path = source_path.resolve()
    if output == source_path.parent or output in source_path.parents:
        raise ValueError("Target output must not be the source directory or its ancestor")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"Output already contains files: {output}; use a new directory or --resume")
    if resume and Path(resume).resolve().parent != output:
        raise ValueError("Resume output must be the directory containing the target checkpoint")


@torch.no_grad()
def teacher_td_advantage(teacher, observations, last_observation, rewards, dones, gamma, batch_size=256):
    """Use next rollout state; reset observations are masked at terminals.

    Only one additional final state is needed, avoiding a duplicate pixel
    rollout buffer. Life-loss masks are the same masks used by target GAE.
    """
    values = torch.cat([
        teacher(batch)[1] for batch in observations.split(batch_size)
    ])
    final_value = teacher(last_observation.unsqueeze(0))[1]
    next_values = torch.cat((values[1:], final_value))
    advantages = rewards + gamma * next_values * (1.0 - dones) - values
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Non-finite source TD Advantage")
    return advantages, values


def mix_advantages(target, source, alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        return target  # Exact no-transfer path; do not evaluate the teacher.
    if source is None or source.shape != target.shape:
        raise ValueError("Source and target Advantages must have identical shapes")
    mixed = (1.0 - alpha) * target + alpha * source.detach()
    if not torch.isfinite(mixed).all():
        raise FloatingPointError("Non-finite mixed Advantage")
    return mixed


def advantage_metrics(target, source, mixed, alpha):
    metrics = {
        "transfer_alpha": alpha,
        "source_advantage_mean": None,
        "source_advantage_std": None,
        "mixed_advantage_mean": float(mixed.mean()),
        "mixed_advantage_std": float(mixed.std(unbiased=False)),
        "source_target_sign_agreement": None,
        "source_target_correlation": None,
    }
    if source is not None:
        metrics["source_advantage_mean"] = float(source.mean())
        metrics["source_advantage_std"] = float(source.std(unbiased=False))
        metrics["source_target_sign_agreement"] = float((target.sign() == source.sign()).float().mean())
        t, s = target - target.mean(), source - source.mean()
        denominator = t.norm() * s.norm()
        if float(denominator) > 1e-12:
            metrics["source_target_correlation"] = float(((t * s).sum() / denominator).clamp(-1.0, 1.0))
    return metrics
