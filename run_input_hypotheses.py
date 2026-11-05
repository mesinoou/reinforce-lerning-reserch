#!/usr/bin/env python3
"""One isolated source-training process per GPU; dynamically schedule seeds."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "objects_256": {"input_mode": "objects", "hidden_size": 256, "feature_mode": "baseline"},
    "objects_768": {"input_mode": "objects", "hidden_size": 768, "feature_mode": "baseline"},
    "pixels_256": {"input_mode": "pixels", "hidden_size": 256, "feature_mode": "baseline"},
    "enemy_control": {"input_mode": "objects", "hidden_size": 256, "feature_mode": "enemy_control"},
    "enemy_extended": {"input_mode": "objects", "hidden_size": 256, "feature_mode": "enemy_extended"},
    "shield_control": {"input_mode": "objects", "hidden_size": 256, "feature_mode": "shield_control"},
    "shield_boxes": {"input_mode": "objects", "hidden_size": 256, "feature_mode": "shield_boxes"},
}
SUITES = {"capacity": ["objects_256", "objects_768", "pixels_256"],
          "enemies": ["enemy_control", "enemy_extended"],
          "shields": ["shield_control", "shield_boxes"], "all": list(CONDITIONS)}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)


def code_identity():
    files = sorted(ROOT.glob("*.py"))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "file_sha256": hashes}


def make_plan(args):
    common = {"env": "ALE/SpaceInvaders-v5", "object_mode": "ram",
              "total_steps": args.total_steps, "frameskip": 4,
              "repeat_action_probability": 0.0, "stack_size": 4,
              "max_enemies": 12, "max_projectiles": 4,
              "slot_strategy": "temporal", "slot_match_distance": 0.20,
              "pixel_width": 84, "pixel_height": 84,
              "reward_mode": "scaled_raw", "reward_scale": 0.1,
              "rollout_steps": 1024, "minibatch_size": 256, "ppo_epochs": 4,
              "learning_rate": 2.5e-4, "gamma": 0.99, "gae_lambda": 0.95,
              "clip_epsilon": 0.1, "value_clip_epsilon": 0.1,
              "value_coefficient": 0.5, "entropy_coefficient": 0.01,
              "max_grad_norm": 0.5, "target_kl": 0.03,
              "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
              "random_eval_episodes": args.random_eval_episodes, "eval_seed": 100000,
              "checkpoint_interval": args.checkpoint_interval,
              "max_episode_steps": 27000, "object_sample_frames": 20,
              "expanded_enemies": 40, "shield_slots": 3,
              "torch_threads": args.torch_threads, "audit_interval": args.audit_interval,
              "anneal_learning_rate": True, "value_clipping": True,
              "life_loss_terminal": True, "auto_fire_reset": False,
              "deterministic_torch": args.deterministic_torch}
    if args.smoke:
        common.update(total_steps=2048, eval_interval=1024, checkpoint_interval=1024,
                      eval_episodes=1, random_eval_episodes=1, max_episode_steps=2000)
    jobs = []
    # Rotate conditions between seeds to avoid a permanent condition/GPU pairing.
    names = SUITES[args.suite]
    for index, seed in enumerate(args.seeds):
        rotated = names[index % len(names):]+names[:index % len(names)]
        for name in rotated:
            jobs.append({"id": f"{name}/seed_{seed}", "condition": name, "seed": seed,
                         "arguments": dict(common, **CONDITIONS[name], seed=seed)})
    plan = {"format_version": 1, "suite": args.suite, "seeds": args.seeds,
            "primary_metric": "Normalized AUC of fixed-seed stochastic evaluation, from 0 to total_steps",
            "secondary_metrics": ["final_stochastic_raw_return", "critic_explained_variance", "input_audit"],
            "common_arguments": common, "conditions": {name: CONDITIONS[name] for name in names},
            "code": code_identity(), "jobs": jobs}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def command_for(job, directory, python, device, resume=False, trainer=None):
    command = [python, "-u", str(trainer or ROOT/"train_input_hypothesis.py")]
    for key, value in job["arguments"].items():
        flag = "--"+key.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-"+key.replace("_", "-"))
        else:
            command.extend([flag, str(value)])
    command += ["--device", device, "--output-dir", str(directory)]
    if resume:
        command += ["--resume", str(directory/"checkpoint_latest.pt")]
    return command


def worker_env(gpu, threads):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    return env


def preflight(python, gpus, threads, cpu=False):
    code = ("import json,torch; import ocatari.core; import gymnasium; "
            "print(json.dumps({'cuda':torch.cuda.is_available(), 'count':torch.cuda.device_count(), "
            "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))")
    info = []
    for gpu in gpus:
        result = subprocess.run([python, "-c", code], env=worker_env(gpu, threads),
                                cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Dependency/device preflight failed for {gpu}:\n{result.stderr}")
        row = json.loads(result.stdout.strip().splitlines()[-1])
        if not cpu and (not row["cuda"] or row["count"] != 1):
            raise RuntimeError(f"GPU {gpu} is not exclusively visible as cuda:0: {row}")
        info.append(dict(row, gpu=gpu))
    return info


def is_complete(directory, total_steps):
    try:
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        with (directory/"periodic_evaluation.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        return (int(summary["environment_steps"]) == total_steps
                and any(row["stage"] == "final" and row["policy"] == "stochastic"
                        and int(row["env_steps"]) == total_steps for row in rows))
    except (OSError, ValueError, KeyError):
        return False


@contextmanager
def lock_root(root):
    # Kernel releases the lock on crash; no stale lock-file removal is needed.
    import fcntl
    with (root/"scheduler.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another scheduler is using {root}") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def run_plan(plan, root, gpus, python, resume=False, cpu=False, trainer=None, poll_seconds=1):
    saved = root/"experiment_plan.json"
    if saved.exists():
        prior = json.loads(saved.read_text(encoding="utf-8"))
        if prior["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Existing plan differs (settings/code/seeds). Use a new output root.")
        if not resume:
            raise ValueError("Output root already has a plan; use --resume or a new root.")
    else:
        save_json(saved, plan)
    status = {job["id"]: {"status": "queued"} for job in plan["jobs"]}
    queue = []
    # Validate all run directories BEFORE launching any subprocess.
    for job in plan["jobs"]:
        directory = root/job["id"]
        if directory.exists() and any(directory.iterdir()):
            if not resume:
                raise ValueError(f"Nonempty run directory: {directory}")
            marker = directory/"job.json"
            if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")).get("plan_sha256") != plan["plan_sha256"]:
                raise ValueError(f"Unmatched or unmarked run directory: {directory}")
            if is_complete(directory, job["arguments"]["total_steps"]):
                status[job["id"]] = {"status": "completed", "skipped_on_resume": True}
                continue
            if (directory/"config.json").exists() and not (directory/"checkpoint_latest.pt").exists():
                raise ValueError(f"Partial run without checkpoint: {directory}; use a new output root.")
        queue.append(job)
    active = {}
    status_path = root/"scheduler_status.json"

    def persist():
        save_json(status_path, {"updated_at": now(), "jobs": status})

    def stop_children():
        def send_signal(process, signum):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        for process, log, job in active.values():
            if process.poll() is None:
                send_signal(process, signal.SIGTERM)
        deadline = time.monotonic()+10
        while any(p.poll() is None for p, _, _ in active.values()) and time.monotonic() < deadline:
            time.sleep(.1)
        for process, log, job in active.values():
            if process.poll() is None:
                send_signal(process, signal.SIGKILL)
            process.wait()
            log.close()
            status[job["id"]].update(status="interrupted", finished_at=now())
        persist()

    def terminated(_signum, _frame):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, terminated)
    persist()
    try:
        while queue or active:
            for gpu in gpus:
                if gpu in active or not queue:
                    continue
                job = queue.pop(0)
                if trainer is None and code_identity() != plan["code"]:
                    raise ValueError("Experiment code changed while scheduling. Stop and use a new output root.")
                directory = root/job["id"]
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint = directory/"checkpoint_latest.pt"
                command = command_for(job, directory, python, "cpu" if cpu else "cuda:0",
                                      resume=resume and checkpoint.exists(), trainer=trainer)
                save_json(directory/"job.json", dict(job, plan_sha256=plan["plan_sha256"],
                                                   assigned_gpu=gpu, command=command))
                log = (directory/"console.log").open("a", encoding="utf-8")
                log.write(f"\n=== scheduler start {now()}, GPU={gpu} ===\n")
                log.flush()
                try:
                    process = subprocess.Popen(command, cwd=ROOT,
                        env=worker_env(gpu, job["arguments"]["torch_threads"]),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    log.close()
                    status[job["id"]] = {"status": "failed", "error": str(exc)}
                    persist()
                    continue
                active[gpu] = (process, log, job)
                status[job["id"]] = {"status": "running", "gpu": gpu, "pid": process.pid,
                                     "started_at": now()}
                print(f"[start] gpu={gpu} {job['id']}", flush=True)
                persist()
            for gpu, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                okay = code == 0 and is_complete(root/job["id"], job["arguments"]["total_steps"])
                status[job["id"]].update(status="completed" if okay else "failed",
                                          returncode=code, finished_at=now())
                del active[gpu]
                print(f"[{'done' if okay else 'failed'}] gpu={gpu} {job['id']} exit={code}", flush=True)
                persist()
            if queue or active:
                time.sleep(poll_seconds)
    except BaseException:
        stop_children()
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    return all(row["status"] == "completed" for row in status.values())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="capacity")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                        help="Physical GPU indices/UUIDs, comma separated; inherits CUDA_VISIBLE_DEVICES by default")
    parser.add_argument("--total-steps", type=int, default=2000000)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--random-eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--audit-interval", type=int, default=32)
    parser.add_argument("--output-root", default="input_hypothesis_runs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Single CPU worker for smoke tests")
    parser.add_argument("--deterministic-torch", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    for key in ("total_steps", "eval_interval", "eval_episodes", "random_eval_episodes",
                "checkpoint_interval", "torch_threads", "audit_interval"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    args.gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not args.cpu and (not args.gpus or len(args.gpus) != len(set(args.gpus))):
        parser.error("Provide unique GPU indices/UUIDs")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    plan = make_plan(args)
    gpus = [None] if args.cpu else args.gpus
    if args.dry_run:
        print(json.dumps(dict(plan, worker_gpus=gpus, output_root=str(root)), indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with lock_root(root):
        info = preflight(args.python, gpus, args.torch_threads, args.cpu)
        print(json.dumps({"devices": info}, indent=2), flush=True)
        success = run_plan(plan, root, gpus, args.python, args.resume, args.cpu)
        if not success:
            print("Some jobs failed; inspect scheduler_status.json and each console.log", file=sys.stderr)
            return 1
        return subprocess.call([args.python, str(ROOT/"summarize_input_hypotheses.py"),
                                "--root", str(root)], cwd=ROOT)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped child processes; rerun the same command with --resume", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
