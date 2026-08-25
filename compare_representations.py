#!/usr/bin/env python3
"""Create paired object-list versus pixel PPO comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np


SHARED_CONDITION_KEYS = [
    "env",
    "object_mode",
    "total_steps",
    "deterministic_torch",
    "frameskip",
    "repeat_action_probability",
    "stack_size",
    "pixel_width",
    "pixel_height",
    "max_enemies",
    "max_projectiles",
    "slot_strategy",
    "slot_match_distance",
    "reward_mode",
    "reward_scale",
    "survival_reward_per_frame",
    "life_loss_penalty",
    "rollout_steps",
    "gamma",
    "gae_lambda",
    "learning_rate",
    "anneal_learning_rate",
    "clip_epsilon",
    "value_clip_epsilon",
    "value_clipping",
    "value_coefficient",
    "entropy_coefficient",
    "max_grad_norm",
    "ppo_epochs",
    "minibatch_size",
    "hidden_size",
    "target_kl",
    "eval_interval",
    "eval_episodes",
    "random_eval_episodes",
    "eval_seed",
    "checkpoint_interval",
    "max_episode_steps",
    "life_loss_terminal",
    "auto_fire_reset",
]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_csv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def last_evaluation(
    rows: Sequence[Dict[str, str]],
    stage: str,
    policy: str,
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if row["stage"] == stage and row["policy"] == policy
    ]
    if not matches:
        raise ValueError(
            f"Missing evaluation: stage={stage}, policy={policy}"
        )
    return matches[-1]


def policy_curve(
    rows: Sequence[Dict[str, str]],
    policy: str,
) -> Dict[int, float]:
    curve: Dict[int, float] = {}
    for row in rows:
        if row["policy"] != policy or row["stage"] == "random_baseline":
            continue
        curve[int(row["env_steps"])] = float(row["raw_return_mean"])
    return curve


def normalized_auc(curve: Dict[int, float]) -> float:
    if len(curve) < 2:
        return float("nan")
    steps = np.asarray(sorted(curve), dtype=np.float64)
    returns = np.asarray([curve[int(step)] for step in steps], dtype=np.float64)
    span = steps[-1] - steps[0]
    if span <= 0.0:
        return float("nan")
    return float(np.trapz(returns, steps) / span)


def load_runs(root: Path, expected_mode: str) -> Dict[int, Dict[str, Any]]:
    directories = sorted(
        path
        for path in root.glob("seed_*")
        if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(f"No seed directories found under {root}")
    runs: Dict[int, Dict[str, Any]] = {}
    for directory in directories:
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
        evaluations = read_csv(directory / "periodic_evaluation.csv")
        arguments = config["arguments"]
        seed = int(arguments["seed"])
        input_mode = str(
            config.get(
                "input_mode",
                arguments.get("input_mode", "objects"),
            )
        )
        if input_mode != expected_mode:
            raise ValueError(
                f"Expected {expected_mode} run, found {input_mode}: {directory}"
            )
        if seed in runs:
            raise ValueError(f"Duplicate seed {seed} under {root}")
        curve = policy_curve(evaluations, "stochastic")
        runs[seed] = {
            "directory": directory,
            "config": config,
            "arguments": arguments,
            "summary": summary,
            "evaluations": evaluations,
            "curve": curve,
            "auc": normalized_auc(curve),
            "random_return": float(
                last_evaluation(
                    evaluations,
                    "random_baseline",
                    "random",
                )["raw_return_mean"]
            ),
            "initial_return": float(
                last_evaluation(
                    evaluations,
                    "initial",
                    "stochastic",
                )["raw_return_mean"]
            ),
            "final_return": float(
                last_evaluation(
                    evaluations,
                    "final",
                    "stochastic",
                )["raw_return_mean"]
            ),
        }
    return runs


def validate_pair(
    seed: int,
    objects: Dict[str, Any],
    pixels: Dict[str, Any],
) -> Dict[str, Any]:
    differences = {}
    for key in SHARED_CONDITION_KEYS:
        object_value = objects["arguments"].get(key)
        pixel_value = pixels["arguments"].get(key)
        if object_value != pixel_value:
            differences[key] = {
                "objects": object_value,
                "pixels": pixel_value,
            }
    if differences:
        raise ValueError(
            f"Seed {seed} does not use matched conditions: {differences}"
        )
    object_steps = set(objects["curve"])
    pixel_steps = set(pixels["curve"])
    if object_steps != pixel_steps:
        raise ValueError(
            f"Seed {seed} evaluation steps differ: "
            f"objects={sorted(object_steps)}, pixels={sorted(pixel_steps)}"
        )
    return {
        key: objects["arguments"].get(key)
        for key in SHARED_CONDITION_KEYS
    }


def sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def aggregate_curve_rows(
    runs_by_mode: Dict[str, Dict[int, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for mode, runs in runs_by_mode.items():
        for run in runs.values():
            for step, raw_return in run["curve"].items():
                values[(mode, step)].append(raw_return)
    rows: List[Dict[str, Any]] = []
    for (mode, step), raw_values in sorted(values.items()):
        array = np.asarray(raw_values, dtype=np.float64)
        rows.append(
            {
                "input_mode": mode,
                "env_steps": step,
                "seed_count": int(array.size),
                "raw_return_mean": float(array.mean()),
                "raw_return_std": float(array.std(ddof=0)),
                "raw_return_sem": sem(array),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matched object-list and pixel PPO runs."
    )
    parser.add_argument(
        "--objects-root",
        default="representation_comparison_runs/objects",
    )
    parser.add_argument(
        "--pixels-root",
        default="representation_comparison_runs/pixels",
    )
    parser.add_argument(
        "--output-dir",
        default="representation_comparison_runs/comparison",
    )
    args = parser.parse_args()

    object_runs = load_runs(Path(args.objects_root), "objects")
    pixel_runs = load_runs(Path(args.pixels_root), "pixels")
    if set(object_runs) != set(pixel_runs):
        raise ValueError(
            "Object and pixel seed sets differ: "
            f"objects={sorted(object_runs)}, pixels={sorted(pixel_runs)}"
        )
    seeds = sorted(object_runs)
    shared_conditions = None
    per_seed_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        conditions = validate_pair(seed, object_runs[seed], pixel_runs[seed])
        if shared_conditions is None:
            shared_conditions = conditions
        elif conditions != shared_conditions:
            raise ValueError("Shared conditions differ across seeds")
        object_run = object_runs[seed]
        pixel_run = pixel_runs[seed]
        per_seed_rows.append(
            {
                "seed": seed,
                "objects_final_raw_return": object_run["final_return"],
                "pixels_final_raw_return": pixel_run["final_return"],
                "objects_minus_pixels_final": (
                    object_run["final_return"] - pixel_run["final_return"]
                ),
                "objects_stochastic_auc": object_run["auc"],
                "pixels_stochastic_auc": pixel_run["auc"],
                "objects_minus_pixels_auc": (
                    object_run["auc"] - pixel_run["auc"]
                ),
                "objects_steps_per_second": object_run["summary"].get(
                    "steps_per_second"
                ),
                "pixels_steps_per_second": pixel_run["summary"].get(
                    "steps_per_second"
                ),
                "objects_parameter_count": object_run["summary"].get(
                    "model_parameter_count"
                ),
                "pixels_parameter_count": pixel_run["summary"].get(
                    "model_parameter_count"
                ),
                "objects_run_directory": str(
                    object_run["directory"].resolve()
                ),
                "pixels_run_directory": str(pixel_run["directory"].resolve()),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_fields = list(per_seed_rows[0])
    write_csv(
        output_dir / "representation_per_seed.csv",
        per_seed_rows,
        per_seed_fields,
    )
    curve_rows = aggregate_curve_rows(
        {"objects": object_runs, "pixels": pixel_runs}
    )
    write_csv(
        output_dir / "representation_curve.csv",
        curve_rows,
        [
            "input_mode",
            "env_steps",
            "seed_count",
            "raw_return_mean",
            "raw_return_std",
            "raw_return_sem",
        ],
    )

    final_differences = np.asarray(
        [row["objects_minus_pixels_final"] for row in per_seed_rows],
        dtype=np.float64,
    )
    auc_differences = np.asarray(
        [row["objects_minus_pixels_auc"] for row in per_seed_rows],
        dtype=np.float64,
    )
    summary = {
        "comparison": "objects_minus_pixels",
        "seed_count": len(seeds),
        "seeds": seeds,
        "shared_conditions": shared_conditions,
        "paired_final_raw_return_difference_mean": float(
            final_differences.mean()
        ),
        "paired_final_raw_return_difference_std": float(
            final_differences.std(ddof=0)
        ),
        "paired_final_raw_return_difference_sem": sem(final_differences),
        "paired_auc_difference_mean": float(auc_differences.mean()),
        "paired_auc_difference_std": float(auc_differences.std(ddof=0)),
        "paired_auc_difference_sem": sem(auc_differences),
        "seeds_objects_above_pixels_final": int(
            np.sum(final_differences > 0.0)
        ),
        "seeds_objects_above_pixels_auc": int(np.sum(auc_differences > 0.0)),
        "claim_boundary": (
            "The objects condition uses OCAtari RAM extraction and therefore "
            "measures an oracle object representation without perception error."
        ),
        "interpretation": (
            "Positive paired differences favor objects. With three seeds these "
            "statistics are descriptive; inspect curves and variance before "
            "making a strong claim."
        ),
    }
    write_json(output_dir / "representation_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode, color in (("objects", "tab:blue"), ("pixels", "tab:orange")):
        selected = [row for row in curve_rows if row["input_mode"] == mode]
        steps = np.asarray([row["env_steps"] for row in selected])
        means = np.asarray([row["raw_return_mean"] for row in selected])
        errors = np.asarray([row["raw_return_sem"] for row in selected])
        axes[0].plot(steps, means, marker="o", color=color, label=mode)
        axes[0].fill_between(
            steps,
            means - errors,
            means + errors,
            color=color,
            alpha=0.20,
        )
    axes[0].set_title("Fixed-seed stochastic evaluation")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Mean raw return")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for row in per_seed_rows:
        axes[1].plot(
            [0, 1],
            [
                row["pixels_final_raw_return"],
                row["objects_final_raw_return"],
            ],
            marker="o",
            alpha=0.75,
            label=f"seed {row['seed']}",
        )
    axes[1].set_xticks([0, 1], ["pixels", "objects"])
    axes[1].set_ylabel("Final stochastic raw return")
    axes[1].set_title("Paired final evaluation")
    axes[1].grid(True, axis="y", alpha=0.25)
    if len(seeds) <= 8:
        axes[1].legend()
    fig.suptitle("Space Invaders representation comparison")
    fig.tight_layout()
    fig.savefig(
        output_dir / "representation_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
