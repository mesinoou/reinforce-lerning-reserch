#!/usr/bin/env python3
"""Compare matched target seeds against alpha=0 using fixed-seed evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_representations import (
    SHARED_CONDITION_KEYS, last_evaluation, policy_curve, read_csv, read_json,
    write_csv, write_json,
)
import matplotlib.pyplot as plt
import numpy as np


def validate_pair(baseline, transfer):
    keys = SHARED_CONDITION_KEYS + [
        "input_mode", "seed", "source_checkpoint_sha256", "survival_bonus", "death_penalty",
    ]
    a, b = baseline["arguments"], transfer["arguments"]
    different = [key for key in keys if a.get(key) != b.get(key)]
    if different:
        raise ValueError(f"Target conditions differ beyond alpha: {different}")
    if not a.get("source_checkpoint_sha256"):
        raise ValueError("Source identity is missing")
    initial = baseline.get("initial_target_sha256")
    if not initial or initial != transfer.get("initial_target_sha256"):
        raise ValueError("Target initial parameters are not identical")


def curve_auc(curve):
    steps = sorted(curve)
    if len(steps) < 2 or steps[-1] <= steps[0]:
        raise ValueError("At least two distinct evaluation steps are required")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate([curve[s] for s in steps], steps) / (steps[-1] - steps[0]))


def compare(root: Path, output: Path, policy: str):
    runs = {}
    for folder in sorted(root.glob("alpha_*/seed_*")):
        if not (folder / "summary.json").is_file():
            raise ValueError(f"Run has not completed: {folder}")
        config = read_json(folder / "config.json")
        args = config["arguments"]
        alpha, seed = float(args["transfer_alpha"]), int(args["seed"])
        key = (alpha, seed)
        if key in runs:
            raise ValueError(f"Duplicate alpha/seed: {key}")
        evaluations = read_csv(folder / "periodic_evaluation.csv")
        curve = policy_curve(evaluations, policy)
        final = last_evaluation(evaluations, "final", policy)
        if max(curve) != args["total_steps"] or int(final["env_steps"]) != args["total_steps"]:
            raise ValueError(f"Incomplete learning budget: {folder}")
        runs[key] = (config, curve, float(final["raw_return_mean"]))
    alphas = sorted({alpha for alpha, _ in runs})
    if not alphas or alphas[0] != 0.0 or len(alphas) < 2:
        raise ValueError("Require alpha=0 and at least one transfer condition")
    seeds = sorted(seed for alpha, seed in runs if alpha == 0.0)
    grid = sorted(runs[(0.0, seeds[0])][1])
    reference = runs[(0.0, seeds[0])][0]["arguments"]
    for alpha in alphas:
        if sorted(seed for a, seed in runs if a == alpha) != seeds:
            raise ValueError("Conditions must contain the same target seeds")
        for seed in seeds:
            config, curve, _ = runs[(alpha, seed)]
            validate_pair(runs[(0.0, seed)][0], config)
            if any(config["arguments"].get(key) != reference.get(key) for key in SHARED_CONDITION_KEYS + ["input_mode", "source_checkpoint_sha256"]):
                raise ValueError("Across-seed aggregation requires identical settings and the same fixed teacher")
            if sorted(curve) != grid:
                raise ValueError("Evaluation grids differ; do not compare unequal budgets/grids")

    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for alpha in alphas:
        matrix = np.asarray([[runs[(alpha, s)][1][step] for step in grid] for s in seeds])
        mean = matrix.mean(axis=0)
        label = f"alpha={alpha:g}" + (" (no transfer)" if alpha == 0.0 else "")
        ax.plot(grid, mean, marker="o", label=label)
        if len(seeds) > 1:
            sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            ax.fill_between(grid, mean - sem, mean + sem, alpha=0.15)
        for seed in seeds:
            _, curve, final = runs[(alpha, seed)]
            _, baseline_curve, baseline_final = runs[(0.0, seed)]
            rows.append({
                "alpha": alpha, "seed": seed, "policy": policy,
                "final_raw_return": final, "normalized_auc": curve_auc(curve),
                "final_minus_baseline": final - baseline_final,
                "auc_minus_baseline": curve_auc(curve) - curve_auc(baseline_curve),
            })
    ax.set_title(f"{reference['env']}: Advantage transfer ({policy}, {len(seeds)} target seeds)")
    ax.set_xlabel("Target environment steps")
    ax.set_ylabel("Evaluation raw return" + (" (mean +/- SEM)" if len(seeds) > 1 else ""))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"transfer_evaluation_{policy}.png", dpi=180)
    plt.close(fig)
    write_csv(output / f"paired_metrics_{policy}.csv", rows, list(rows[0]))
    write_json(output / f"comparison_{policy}.json", {
        "policy": policy, "target_seeds": seeds, "alphas": alphas,
        "source_sha256": reference["source_checkpoint_sha256"],
        "source_checkpoint": reference["source_checkpoint"],
        "conditional_on_one_fixed_source": True,
        "note": "Single source model held fixed. One target seed is exploratory; SEM is across target seeds, not evaluation episodes.",
        "paired_metrics": rows,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy", choices=["deterministic", "stochastic"], default="stochastic")
    args = parser.parse_args()
    print(compare(args.root, args.output_dir or args.root / "comparison", args.policy).resolve())


if __name__ == "__main__":
    main()
