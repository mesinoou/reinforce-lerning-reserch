#!/usr/bin/env python3
"""Space Invaders full-object comparison with live per-GPU progress and ETA.

Reuses the existing PPO and GPU scheduler without changing their source files.
Only the parent process polls JSON progress; worker output stays in console.log.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

from object_channels import object_channel_spec


CONDITIONS = {
    "common_256": {"input_mode": "objects", "hidden_size": 256,
                   "representation": "common"},
    "all_objects_256": {"input_mode": "objects", "hidden_size": 256,
                        "representation": "all_objects"},
    "pixels_256": {"input_mode": "pixels", "hidden_size": 256,
                   "representation": "pixels"},
    "object_channels_256": {"input_mode": "object_channels", "hidden_size": 256,
                            "representation": "object_channels"},
}
DEFAULT_CONDITIONS = ["common_256", "all_objects_256", "pixels_256"]
NOTE = (
    "全物体比較では入力次元・物体選択・並べ方・パラメータ数が変わる。"
    "表現全体の比較であり、情報追加だけの因果効果とは扱わない。"
    "HUDと盾内部の損傷形状は含まない。"
)


def full_object_layout(spec):
    if not spec or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                       for n in spec.values()):
        raise ValueError("Invalid OCAtari full-object specification")
    categories, names = [], []
    for name, count in spec.items():
        categories.extend([name.replace("_", "").lower()] * count)
        names.extend(f"{name}[{i}]" for i in range(count))
    return categories, names


def make_full_object_encoder(common_encoder_class, spec, np):
    categories, _ = full_object_layout(spec)

    class FullObjectEncoder(common_encoder_class):
        @property
        def frame_dim(self):
            return len(categories) * 5

        def encode_frame(self, objects):
            objects = list(objects)
            if len(objects) != len(categories):
                raise ValueError(
                    f"RAM slot count mismatch: {len(objects)} vs {len(categories)}")
            result = np.zeros((len(categories), 5), dtype=np.float32)
            for index, (obj, expected) in enumerate(zip(objects, categories)):
                if not self._visible(obj):
                    continue
                actual = self._category(obj)
                if actual != expected:
                    raise ValueError(
                        f"RAM slot {index}: expected {expected}, found {actual}")
                if actual == "player":
                    self.last_player_center = self._center(obj)
                result[index] = self._object_feature(
                    obj, self.last_player_center, relative=False)
            return result.reshape(-1)

    return FullObjectEncoder


def train_worker(argv):
    # Import GPU/Atari dependencies only inside isolated worker processes.
    import numpy as np
    import torch
    import ocatari_source_ppo as base
    from ocatari.ram import spaceinvaders
    from ocatari_transfer_full_experiment import CommonObjectEncoder

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--representation", choices=["common", "all_objects", "pixels", "object_channels"],
                        required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    extra, remaining = parser.parse_known_args(argv)
    expected_mode = (extra.representation if extra.representation in {"pixels", "object_channels"}
                     else "objects")
    args = base.parse_args(remaining, defaults={"input_mode": expected_mode})
    if args.input_mode != expected_mode:
        raise ValueError("Input mode and representation disagree")
    if args.env != "ALE/SpaceInvaders-v5" or args.object_mode != "ram":
        raise ValueError("This experiment requires SpaceInvaders RAM mode")
    if args.source_checkpoint or args.transfer_alpha:
        raise ValueError("This experiment does not use transfer")
    if extra.torch_threads < 1:
        raise ValueError("torch-threads must be positive")

    spec = dict(spaceinvaders.MAX_NB_OBJECTS)
    categories, slot_names = full_object_layout(spec)
    args.representation = extra.representation
    args.torch_threads = extra.torch_threads
    uses_ram_schema = extra.representation in {"all_objects", "object_channels"}
    args.full_object_spec = spec if uses_ram_schema else {}
    args.full_object_ram_sha256 = (
        hashlib.sha256(Path(spaceinvaders.__file__).read_bytes()).hexdigest()
        if uses_ram_schema else "")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False).get(
            "configuration", {})
        for key in ("representation", "input_mode", "hidden_size", "total_steps",
                    "eval_interval", "eval_episodes", "eval_seed",
                    "full_object_spec", "full_object_ram_sha256"):
            if saved.get(key) != getattr(args, key):
                raise ValueError(f"Resume setting differs: {key}")

    torch.set_num_threads(extra.torch_threads)
    torch.set_num_interop_threads(1)
    encoder_class = make_full_object_encoder(CommonObjectEncoder, spec, np)
    original_factory, original_names = base.make_observation_env, base.feature_names

    def factory(**kwargs):
        env = original_factory(**kwargs)
        if kwargs["input_mode"] == "objects":
            env.encoder = encoder_class(kwargs["encoder_config"])
            env.obs_dim = env.encoder.obs_dim
        return env

    def names(config):
        return [f"frame[{frame}].{slot}.{field}"
                for frame in range(config.stack_size) for slot in slot_names
                for field in ("present", "cx", "cy", "w", "h")]

    if extra.representation == "all_objects":
        base.make_observation_env, base.feature_names = factory, names
        print(f"Full-object schema: {spec}; "
              f"dimension={len(categories) * 5 * args.stack_size}", flush=True)
    try:
        base.train(args)
    finally:
        base.make_observation_env, base.feature_names = original_factory, original_names


def read_json(path):
    # The scheduler uses atomic writes. Also tolerate startup and external copies.
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def finite_number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def duration(seconds):
    seconds = finite_number(seconds)
    if seconds is None or seconds < 0:
        return "--:--:--"
    days, seconds = divmod(int(seconds), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    text = f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{days}d {text}" if days else text


def epoch(iso_time):
    try:
        return datetime.fromisoformat(iso_time.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None


def progress_lines(root, plan, status, elapsed, wall_time=None):
    """Format an observation of the scheduler, without altering training state."""
    wall_time = time.time() if wall_time is None else wall_time
    states = status.get("jobs", {})
    counts = {name: sum(row.get("status") == name for row in states.values())
              for name in ("completed", "running", "queued", "failed", "interrupted")}
    lines = [f"[overall] done={counts['completed']}/{len(plan['jobs'])} "
             f"running={counts['running']} queued={counts['queued']} "
             f"failed={counts['failed']} interrupted={counts['interrupted']} "
             f"elapsed={duration(elapsed)}"]
    jobs = {job["id"]: job for job in plan["jobs"]}
    for job_id, state in states.items():
        if state.get("status") != "running" or job_id not in jobs:
            continue
        job = jobs[job_id]
        total = job["arguments"]["total_steps"]
        progress_path = Path(root) / job_id / "progress.json"
        row = read_json(progress_path)
        prefix = f"[progress] gpu={state.get('gpu')} {job_id}"
        started = epoch(state.get("started_at"))
        process_elapsed = max(0, wall_time - started) if started is not None else None
        stamp = epoch(row.get("timestamp")) if row else None
        # Ignore the previous process's progress until a resumed worker writes.
        fresh = row is not None and (started is None or
                                     (stamp is not None and stamp + 1 >= started))
        if not fresh or row.get("total_env_steps") != total:
            lines.append(f"{prefix} initializing/resuming/initial-evaluation "
                         f"elapsed={duration(process_elapsed)} eta=--:--:--")
            continue
        steps = finite_number(row.get("env_steps"))
        if steps is None or steps < 0 or steps > total:
            lines.append(f"{prefix} waiting for valid progress data")
            continue
        steps = int(steps)
        speed = finite_number(row.get("average_steps_per_second"))
        speed_text = f"{speed:.1f}" if speed is not None and speed >= 0 else "--"
        base = (f"{prefix} {100 * steps / total:6.2f}% "
                f"steps={steps:,}/{total:,} speed={speed_text} step/s "
                f"elapsed={duration(process_elapsed)}")
        if steps == total:
            lines.append(base + " training-finished; evaluation/saving in progress")
            continue
        eta = row.get("eta_seconds")
        finish = row.get("estimated_finish_time") or "pending"
        age = max(0, wall_time - stamp) if stamp is not None else None
        recent = finite_number(row.get("recent_raw_return_mean"))
        score = f" train100={recent:.2f}" if recent is not None else ""
        lines.append(base + f" eta~{duration(eta)} finish~{finish}" + score
                     + f" updated={duration(age)} ago")
    return lines


class TerminalProgress:
    """Read progress.json while the original main-thread scheduler owns signals."""
    def __init__(self, root, plan, interval=30.0, emit=print):
        self.root, self.plan, self.interval, self.emit = Path(root), plan, interval, emit
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None

    def report(self):
        status = read_json(self.root / "scheduler_status.json")
        if status:
            self.emit("\n".join(progress_lines(
                self.root, self.plan, status, time.monotonic() - self.started)), flush=True)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.report()

    def __enter__(self):
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._loop, name="terminal-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.report()


def build_plan(argv):
    import run_input_hypotheses as runner
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                        default=DEFAULT_CONDITIONS)
    parser.add_argument("--status-interval", type=float, default=30.0)
    extra, remaining = parser.parse_known_args(argv)
    if not math.isfinite(extra.status_interval) or extra.status_interval <= 0:
        parser.error("--status-interval must be finite and positive")
    if len(set(extra.conditions)) != len(extra.conditions):
        parser.error("--conditions must be unique")
    if not any(arg == "--suite" or arg.startswith("--suite=") for arg in remaining):
        remaining = ["--suite", "all_objects"] + remaining
    original_conditions, original_suites = dict(runner.CONDITIONS), dict(runner.SUITES)
    try:
        runner.CONDITIONS.update(CONDITIONS)
        runner.SUITES["all_objects"] = extra.conditions
        args = runner.parse_args(remaining)
        if args.suite != "all_objects":
            raise ValueError("Use --suite all_objects")
        plan = runner.make_plan(args)
    finally:
        runner.CONDITIONS.clear()
        runner.CONDITIONS.update(original_conditions)
        runner.SUITES.clear()
        runner.SUITES.update(original_suites)
    for arguments in [plan["common_arguments"]] + [job["arguments"] for job in plan["jobs"]]:
        for key in ("expanded_enemies", "shield_slots", "audit_interval"):
            arguments.pop(key, None)
    if "object_channels_256" in plan["conditions"]:
        common = plan["common_arguments"]
        plan["object_channel_spec"] = object_channel_spec(
            common["pixel_width"], common["pixel_height"], common["stack_size"])
        plan["comparison_pairs"] = [["pixels_256", "object_channels_256", "object_channels_vs_pixels"]]
        plan["evaluation_intervals"] = {
            "early_auc": [0, 2_000_000],
            "late_mean": [10_000_000, 14_000_000],
        }
        plan["primary_metric"] = (
            "Normalized AUC of fixed-seed stochastic raw returns, 0–2,000,000 steps"
            if not args.smoke else "Smoke test; learning effects are not evaluated"
        )
        plan["secondary_metrics"] = ["auc", "late_mean", "final_raw_return", "steps_per_second"]
    # JSON is already written at the existing trainer's 10-update/60-second
    # cadence. The display interval does not change the training configuration.
    plan.pop("plan_sha256")
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return args, extra, plan


def run_experiment(argv):
    import run_input_hypotheses as runner
    import summarize_input_hypotheses as summary
    args, extra, plan = build_plan(argv)
    root = Path(args.output_root).resolve()
    gpus = [None] if args.cpu else args.gpus
    if args.dry_run:
        print(json.dumps(dict(plan, worker_gpus=gpus, output_root=str(root)), indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with runner.lock_root(root):
        devices = runner.preflight(args.python, gpus, args.torch_threads, args.cpu)
        print(json.dumps({"devices": devices}, indent=2), flush=True)
        print(f"[monitor] every {extra.status_interval:g}s; ETA is an estimate for "
              "remaining training, with past periodic-evaluation overhead. "
              "Initial/final evaluation and saving can take extra time.", flush=True)
        with TerminalProgress(root, plan, extra.status_interval):
            success = runner.run_plan(
                plan, root, gpus, args.python, resume=args.resume, cpu=args.cpu,
                trainer=Path(__file__).resolve())
        if not success:
            print(f"Some jobs failed. Inspect {root / 'scheduler_status.json'} "
                  "and each run's console.log.", file=sys.stderr)
            return 1
        if "object_channel_spec" in plan:
            result = summary.summarize(root)
            print(json.dumps(result["paired_comparisons"], indent=2))
            return 0
        original_pairs = summary.PAIRS
        summary.PAIRS = [("common_256", "all_objects_256", "full_objects"),
                         ("all_objects_256", "pixels_256", "pixel_reference")]
        try:
            result = summary.summarize(root)
        finally:
            summary.PAIRS = original_pairs
        result["limitations"].append(NOTE)
        output = root / "comparison"
        (output / "comparison_summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report = output / "RESULTS.md"
        text = report.read_text(encoding="utf-8").split(
            "容量比較の改善は容量・最適化の制約")[0]
        report.write_text(text + NOTE + "\n", encoding="utf-8")
        print(json.dumps(result["paired_comparisons"], indent=2))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(arg == "--representation" or arg.startswith("--representation=") for arg in argv):
        train_worker(argv)
        return 0
    return run_experiment(argv)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped child processes; rerun the same command with --resume", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
