#!/usr/bin/env python3
"""Aggregate multiple source-only OCAtari PPO seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np


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
    fields: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def last_evaluation(
    rows: List[Dict[str, str]],
    stage: str,
    policy: str,
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if row["stage"] == stage and row["policy"] == policy
    ]
    if not matches:
        raise ValueError(f"Missing evaluation row: stage={stage}, policy={policy}")
    return matches[-1]


def curve_auc(rows: List[Dict[str, str]], policy: str) -> float:
    by_step: Dict[int, float] = {}
    for row in rows:
        if row["policy"] != policy or row["stage"] == "random_baseline":
            continue
        by_step[int(row["env_steps"])] = float(row["raw_return_mean"])
    if len(by_step) < 2:
        return float("nan")
    steps = np.asarray(sorted(by_step), dtype=np.float64)
    values = np.asarray([by_step[int(step)] for step in steps], dtype=np.float64)
    span = steps[-1] - steps[0]
    if span <= 0.0:
        return float("nan")
    return float(np.trapz(values, steps) / span)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate source-only PPO results across seeds."
    )
    parser.add_argument("runs", nargs="*", help="Seed result directories.")
    parser.add_argument("--root", default="source_baseline_runs")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    run_directories = [Path(value) for value in args.runs]
    if not run_directories:
        run_directories = sorted(
            path
            for path in Path(args.root).glob("seed_*")
            if path.is_dir()
        )
    if not run_directories:
        parser.error("No seed result directories were found")

    output_dir = Path(args.output_dir or (Path(args.root) / "aggregate"))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, Any]] = []
    curves: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    for run_directory in run_directories:
        summary_path = run_directory / "summary.json"
        evaluation_path = run_directory / "periodic_evaluation.csv"
        config_path = run_directory / "config.json"
        for required in (summary_path, evaluation_path, config_path):
            if not required.exists():
                raise FileNotFoundError(f"Required result file is missing: {required}")
        summary = read_json(summary_path)
        config = read_json(config_path)
        evaluations = read_csv(evaluation_path)
        seed = int(config["arguments"]["seed"])
        input_mode = str(
            config.get(
                "input_mode",
                config["arguments"].get("input_mode", "objects"),
            )
        )
        random_row = last_evaluation(
            evaluations,
            "random_baseline",
            "random",
        )
        initial_row = last_evaluation(
            evaluations,
            "initial",
            "stochastic",
        )
        final_row = last_evaluation(
            evaluations,
            "final",
            "stochastic",
        )
        random_return = float(random_row["raw_return_mean"])
        initial_return = float(initial_row["raw_return_mean"])
        final_return = float(final_row["raw_return_mean"])
        per_seed_rows.append(
            {
                "seed": seed,
                "input_mode": input_mode,
                "run_directory": str(run_directory.resolve()),
                "environment_steps": int(summary["environment_steps"]),
                "random_raw_return": random_return,
                "initial_stochastic_raw_return": initial_return,
                "final_stochastic_raw_return": final_return,
                "final_minus_random": final_return - random_return,
                "final_minus_initial": final_return - initial_return,
                "stochastic_evaluation_auc": curve_auc(
                    evaluations,
                    "stochastic",
                ),
                "training_last_20_percent_raw_return": summary.get(
                    "training_raw_return_last_20_percent_mean",
                    float("nan"),
                ),
                "steps_per_second": summary.get(
                    "steps_per_second",
                    float("nan"),
                ),
                "model_parameter_count": summary.get(
                    "model_parameter_count",
                    config.get("model_parameter_count"),
                ),
                "player_detection_rate": summary["object_diagnostics"][
                    "player_detection_rate"
                ],
                "enemy_detection_rate": summary["object_diagnostics"][
                    "enemy_detection_rate"
                ],
            }
        )
        per_run_curve: Dict[Tuple[str, int], float] = {}
        for row in evaluations:
            if row["policy"] not in {"deterministic", "stochastic"}:
                continue
            key = (row["policy"], int(row["env_steps"]))
            per_run_curve[key] = float(row["raw_return_mean"])
        for key, value in per_run_curve.items():
            curves[key].append(value)

    per_seed_rows.sort(key=lambda row: row["seed"])
    input_modes = {row["input_mode"] for row in per_seed_rows}
    if len(input_modes) != 1:
        raise ValueError(
            f"Runs from multiple input modes cannot be aggregated: {input_modes}"
        )
    curve_rows: List[Dict[str, Any]] = []
    for (policy, env_steps), values in sorted(
        curves.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        array = np.asarray(values, dtype=np.float64)
        curve_rows.append(
            {
                "policy": policy,
                "env_steps": env_steps,
                "seed_count": len(values),
                "raw_return_mean": float(array.mean()),
                "raw_return_std": float(array.std(ddof=0)),
                "raw_return_sem": float(
                    array.std(ddof=1) / math.sqrt(len(array))
                )
                if len(array) > 1
                else 0.0,
            }
        )

    write_csv(
        output_dir / "per_seed_summary.csv",
        per_seed_rows,
        [
            "seed",
            "input_mode",
            "run_directory",
            "environment_steps",
            "random_raw_return",
            "initial_stochastic_raw_return",
            "final_stochastic_raw_return",
            "final_minus_random",
            "final_minus_initial",
            "stochastic_evaluation_auc",
            "training_last_20_percent_raw_return",
            "steps_per_second",
            "model_parameter_count",
            "player_detection_rate",
            "enemy_detection_rate",
        ],
    )
    write_csv(
        output_dir / "evaluation_curve_aggregate.csv",
        curve_rows,
        [
            "policy",
            "env_steps",
            "seed_count",
            "raw_return_mean",
            "raw_return_std",
            "raw_return_sem",
        ],
    )

    random_values = np.asarray(
        [row["random_raw_return"] for row in per_seed_rows],
        dtype=np.float64,
    )
    initial_values = np.asarray(
        [row["initial_stochastic_raw_return"] for row in per_seed_rows],
        dtype=np.float64,
    )
    final_values = np.asarray(
        [row["final_stochastic_raw_return"] for row in per_seed_rows],
        dtype=np.float64,
    )
    seed_count = len(per_seed_rows)
    wins_over_random = int(np.sum(final_values > random_values))
    wins_over_initial = int(np.sum(final_values > initial_values))
    required_wins = math.ceil(2 * seed_count / 3)
    aggregate_summary = {
        "input_mode": per_seed_rows[0]["input_mode"],
        "seed_count": seed_count,
        "seeds": [row["seed"] for row in per_seed_rows],
        "random_raw_return_mean": float(random_values.mean()),
        "initial_stochastic_raw_return_mean": float(initial_values.mean()),
        "final_stochastic_raw_return_mean": float(final_values.mean()),
        "final_stochastic_raw_return_std": float(final_values.std(ddof=0)),
        "final_minus_random_mean": float(
            (final_values - random_values).mean()
        ),
        "final_minus_initial_mean": float(
            (final_values - initial_values).mean()
        ),
        "seeds_final_above_random": wins_over_random,
        "seeds_final_above_initial": wins_over_initial,
        "minimum_seed_count_met": seed_count >= 3,
        "minimum_directional_criteria_met": bool(
            seed_count >= 3
            and final_values.mean() > random_values.mean()
            and final_values.mean() > initial_values.mean()
            and wins_over_random >= required_wins
            and wins_over_initial >= required_wins
        ),
        "interpretation": (
            "The automatic criterion is a screening rule, not a statistical "
            "proof. Inspect the periodic curve and PPO diagnostics before "
            "declaring the source baseline established."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", aggregate_summary)

    fig, ax = plt.subplots(figsize=(9, 5))
    for policy in ("deterministic", "stochastic"):
        selected = [row for row in curve_rows if row["policy"] == policy]
        if not selected:
            continue
        ax.errorbar(
            [row["env_steps"] for row in selected],
            [row["raw_return_mean"] for row in selected],
            yerr=[row["raw_return_sem"] for row in selected],
            marker="o",
            capsize=3,
            label=f"{policy} (mean ± SEM)",
        )
    ax.axhline(
        random_values.mean(),
        color="black",
        linestyle="--",
        label="random baseline mean",
    )
    ax.set_title(
        "Space Invaders source PPO across seeds "
        f"({aggregate_summary['input_mode']})"
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation raw return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_curve_across_seeds.png", dpi=180)
    plt.close(fig)

    print(json.dumps(aggregate_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
