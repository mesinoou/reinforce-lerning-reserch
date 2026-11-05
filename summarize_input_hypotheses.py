#!/usr/bin/env python3
"""Validate a planned experiment and summarize paired evaluation AUCs."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics

PAIRS = [("objects_256", "objects_768", "capacity"),
         ("objects_256", "pixels_256", "representation_reference"),
         ("enemy_control", "enemy_extended", "enemy_information"),
         ("shield_control", "shield_boxes", "shield_geometry")]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_auc(curve, interval=None):
    xs = sorted(curve)
    if len(xs) < 2 or xs[-1] <= xs[0]:
        raise ValueError("AUC needs at least two different evaluation steps")
    if interval is not None:
        start, end = interval
        if not xs[0] <= start < end <= xs[-1]:
            return None

        def interpolate(step):
            if step in curve:
                return curve[step]
            for left, right in zip(xs, xs[1:]):
                if left < step < right:
                    weight = (step - left) / (right - left)
                    return curve[left] + weight * (curve[right] - curve[left])
            raise ValueError(f"Missing evaluation bounds for step {step}")

        curve = {step: interpolate(step) for step in
                 [start, *(step for step in xs if start < step < end), end]}
        xs = sorted(curve)
    return sum((curve[a]+curve[b])*.5*(b-a) for a,b in zip(xs,xs[1:]))/(xs[-1]-xs[0])


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "std_across_seeds": statistics.stdev(values) if len(values)>1 else None,
            "values": values}


def load_results(root, partial=False):
    plan = json.loads((root/"experiment_plan.json").read_text(encoding="utf-8"))
    runs = {}
    grid = None
    for job in plan["jobs"]:
        directory = root/job["id"]
        marker = json.loads((directory/"job.json").read_text(encoding="utf-8"))
        if marker["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError(f"Plan mismatch: {directory}")
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        actual = config["arguments"]
        mismatches = {key: [expected, actual.get(key)] for key, expected in job["arguments"].items()
                      if actual.get(key) != expected}
        if mismatches:
            raise ValueError(f"Run setting mismatch: {directory}: {mismatches}")
        if actual.get("input_mode") == "object_channels" and (
                config.get("object_channel_spec") != plan.get("object_channel_spec")):
            raise ValueError(f"Object-channel specification mismatch: {directory}")
        summary = {} if partial else json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        total = job["arguments"]["total_steps"]
        if not partial and summary["environment_steps"] != total:
            raise ValueError(f"Incomplete training: {directory}")
        evaluations = read_csv(directory/"periodic_evaluation.csv")
        curve = {}
        final_found = False
        for row in evaluations:
            if row["policy"] != "stochastic" or row["stage"] == "random_baseline":
                continue
            step, value = int(row["env_steps"]), float(row["raw_return_mean"])
            if not math.isfinite(value) or step < 0 or step > total:
                raise ValueError(f"Invalid evaluation: {directory}")
            if int(float(row["episodes"])) != actual["eval_episodes"]:
                raise ValueError(f"Evaluation episode count mismatch: {directory}")
            curve[step] = value
            final_found |= row["stage"] == "final" and step == total
        if not curve or min(curve) != 0 or (not partial and (max(curve) != total or not final_found)):
            raise ValueError(f"Missing initial/final stochastic evaluation: {directory}")
        if grid is None:
            grid = sorted(curve)
        elif partial:
            grid = sorted(set(grid).intersection(curve))
        elif sorted(curve) != grid:
            raise ValueError(f"Evaluation grids differ: {directory}")
        updates = [] if partial else read_csv(directory/"updates.csv")
        late = [row for row in updates if float(row["env_steps"]) >= .8*total]
        ev = [float(row["explained_variance"]) for row in late]
        finite_ev = [value for value in ev if math.isfinite(value)]
        runs[(job["condition"],job["seed"])] = {
            "condition": job["condition"], "seed": job["seed"], "curve": curve,
            "model_parameters": config["model_parameter_count"],
            "observation_dimension": config["observation_dimension"],
            "initial_model_sha256": config.get("initial_target_sha256"),
            "resumed": bool(actual.get("resume")),
            "late_explained_variance_mean": statistics.mean(finite_ev) if finite_ev else None,
            "late_explained_variance_nonfinite_rows": len(ev)-len(finite_ev),
            "input_audit": summary.get("object_diagnostics", {}).get("input_audit", {}),
        }
        if "evaluation_intervals" in plan:
            runs[(job["condition"], job["seed"])]["steps_per_second"] = summary.get("steps_per_second")
    if not grid or len(grid) < 2:
        raise ValueError("Comparison needs at least two evaluation steps common to all planned runs")
    score_key = "last_common_raw_return" if partial else "final_raw_return"
    for run in runs.values():
        curve = {step: run["curve"][step] for step in grid}
        run.update(curve=curve, auc=normalized_auc(curve))
        run[score_key] = curve[grid[-1]]
        for key, interval in plan.get("evaluation_intervals", {}).items():
            run[key] = normalized_auc(curve, interval)
    return plan, runs, grid


def summarize(root, plot=True, partial=False):
    plan, runs, grid = load_results(root, partial=partial)
    output = root/("comparison_partial" if partial else "comparison")
    output.mkdir(exist_ok=True)
    score_key = "last_common_raw_return" if partial else "final_raw_return"
    interval_keys = list(plan.get("evaluation_intervals", {}))
    rows = [{key: value for key,value in run.items() if key not in ("curve","input_audit")}
            for run in runs.values()]
    write_csv(output/"per_seed.csv", rows)
    pairs = []
    paired_rows = []
    for control, changed, name in plan.get("comparison_pairs", PAIRS):
        if control not in plan["conditions"] or changed not in plan["conditions"]:
            continue
        differences = []
        finals = []
        interval_differences = {key: [] for key in interval_keys}
        for seed in plan["seeds"]:
            a,b = runs[(control,seed)],runs[(changed,seed)]
            if name in ("enemy_information", "shield_geometry"):
                if (a["model_parameters"] != b["model_parameters"]
                        or a["observation_dimension"] != b["observation_dimension"]):
                    raise ValueError(f"Information pair has different dimensions/capacity: {name}")
                if (not a["initial_model_sha256"]
                        or a["initial_model_sha256"] != b["initial_model_sha256"]):
                    raise ValueError(f"Information pair has different initial weights: {name}")
            differences.append(b["auc"]-a["auc"])
            finals.append(b[score_key]-a[score_key])
            row = {"comparison": name, "seed": seed, "control": control,
                   "changed": changed, "auc_difference": differences[-1],
                   f"{score_key}_difference": finals[-1]}
            for key in interval_keys:
                delta = b[key] - a[key] if a[key] is not None and b[key] is not None else None
                row[f"{key}_difference"] = delta
                if delta is not None:
                    interval_differences[key].append(delta)
            paired_rows.append(row)
        pairs.append({"comparison": name, "control": control, "changed": changed,
                      "paired_auc_difference": describe(differences),
                      ("paired_last_common_return_difference" if partial else "paired_final_return_difference"):
                          describe(finals),
                      **{f"paired_{key}_difference": describe(values) if values else None
                         for key, values in interval_differences.items()}})
    write_csv(output/"paired_differences.csv", paired_rows)
    curve_rows = []
    condition_metrics = {}
    for condition in plan["conditions"]:
        selected = [runs[(condition,seed)] for seed in plan["seeds"]]
        condition_metrics[condition] = {
            "auc": describe([r["auc"] for r in selected]),
            score_key: describe([r[score_key] for r in selected])}
        for key in interval_keys:
            values = [r[key] for r in selected if r[key] is not None]
            condition_metrics[condition][key] = describe(values) if values else None
        for step in grid:
            values = [r["curve"][step] for r in selected]
            desc = describe(values)
            curve_rows.append({"condition": condition, "env_steps": step, "seed_count":len(values),
                               "raw_return_mean":desc["mean"],"raw_return_std":desc["std_across_seeds"]})
    write_csv(output/"evaluation_curve.csv", curve_rows)
    result = {"plan_sha256": plan["plan_sha256"], "suite":plan["suite"],
              "auc_interval": [grid[0],grid[-1]],
              "auc_definition": "trapezoidal integral of stochastic fixed-seed raw-return evaluations / interval length; no training-episode smoothing",
              "conditions":condition_metrics,"paired_comparisons":pairs,
              "input_audits": {f"{key[0]}/seed_{key[1]}": run["input_audit"] for key,run in runs.items()},
              "limitations": ["Seed count is small; differences are descriptive, without significance claims.",
                              "Critic metrics use different on-policy state/return distributions.",
                              "Shield boxes do not encode interior damage.",
                              "Concurrent wall-clock measurements include shared CPU/I/O contention."]}
    if interval_keys:
        result.update(partial=partial, evaluation_intervals=plan["evaluation_intervals"],
                      primary_metric=plan["primary_metric"])
    (output/"comparison_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    score_label = "共通評価点の得点平均" if partial else "最終評価得点平均"
    scope = f"集計区間 {grid[0]:,}〜{grid[-1]:,} 環境ステップ" + ("（途中結果）" if partial else "")
    lines = ["# 入力仮説検証の結果", "", f"suite: {plan['suite']} / seeds: {plan['seeds']}", "", scope,
             ("主指標は0〜200万stepの正規化AUC。短い試走では算出しない。" if interval_keys else
              "主指標は独立した評価走行のraw returnによる正規化AUC。"),
             "差は変更条件 − 対照条件。seed数が少ないため、有意差や原因の確定とは扱わない。", "",
             f"| 条件 | 区間全体の評価AUC平均 | {score_label} |", "| --- | ---: | ---: |"]
    for condition, metrics in condition_metrics.items():
        lines.append(f"| {condition} | {metrics['auc']['mean']:.2f} | {metrics[score_key]['mean']:.2f} |")
    lines += ["", "| 比較 | 対応seedのAUC差平均 | seedごとのAUC差 |", "| --- | ---: | --- |"]
    for pair in pairs:
        delta = pair["paired_auc_difference"]
        lines.append(f"| {pair['comparison']} | {delta['mean']:.2f} | {', '.join(f'{v:.2f}' for v in delta['values'])} |")
    if interval_keys:
        for pair in pairs:
            for key in interval_keys:
                delta = pair[f"paired_{key}_difference"]
                if delta:
                    left, right = plan["evaluation_intervals"][key]
                    lines.append(f"\n{pair['comparison']} の {left:,}〜{right:,} stepの区間平均差は {delta['mean']:+.2f}。")
    else:
        lines += ["", "容量比較の改善は容量・最適化の制約に整合するが、CNNとの構造差までは解消しない。",
                  "追加情報とゼロ対照の差は、その情報を使う効果の検証に用いる。改善しない場合も欠落情報の無関係を証明しない。",
                  "盾比較の対象は存在と外形のみ。内部の損傷形状は未表現。"]
    lines += ["", "詳細：comparison_summary.json、per_seed.csv、paired_differences.csv、evaluation_curve.csv。"]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if plot:
        os.environ.setdefault("MPLBACKEND","Agg")
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(10,5.5),layout="constrained")
        for condition in plan["conditions"]:
            selected = [row for row in curve_rows if row["condition"]==condition]
            ax.plot([r["env_steps"]/1e6 for r in selected], [r["raw_return_mean"] for r in selected],
                    label=condition, linewidth=1.6)
        ax.set(xlabel="Training environment steps [millions]", ylabel="Evaluation raw return",
               title=f"Space Invaders | {plan['suite']} | mean of {len(plan['seeds'])} seeds"
                     + (" | partial" if partial else ""))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9,frameon=False)
        fig.savefig(output/"evaluation_comparison.png",dpi=180)
        plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--no-plot",action="store_true")
    parser.add_argument("--partial", action="store_true",
                        help="Compare the evaluation steps common to all runs in comparison_partial/.")
    args=parser.parse_args()
    try:
        result=summarize(args.root, not args.no_plot, partial=args.partial)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result["paired_comparisons"],indent=2))


if __name__=="__main__":
    main()
