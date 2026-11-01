"""Phase 6: run many (reward, seed) trainings one after another, and evaluate each.

Every run is a separate process (train, then evaluate), so a crash in one run
does not kill the others. The experiment can be resumed: a run that already
has eval.json is skipped, and a run with model.zip but no eval.json is only
evaluated. Progress goes to <root>/experiment.log and to stdout.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rocket_land.config import ROOT, RUNS_DIR, TOTAL_TIMESTEPS
from rocket_land.rewards import REWARDS


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate every (reward, seed) pair.")
    p.add_argument("--reward", nargs="+", required=True, help=f"names from {list(REWARDS)}, or 'all'")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--episodes", type=int, default=20, help="evaluation episodes per run")
    p.add_argument("--force", action="store_true", help="redo runs that are already done")
    p.add_argument("--root", type=Path, default=RUNS_DIR, help="runs go to <root>/<reward>/seed<k>")
    args = p.parse_args()
    if "all" in args.reward:
        args.reward = list(REWARDS)
    for name in args.reward:
        if name not in REWARDS:
            p.error(f"unknown reward {name!r}, choose from {list(REWARDS)} or 'all'")
    return args


def plan_run(run_dir: Path, force: bool) -> str:
    # What to do for one run: "skip", "evaluate" (model but no eval) or "train"
    if force:
        return "train"
    if (run_dir / "eval.json").exists():
        return "skip"
    if (run_dir / "model.zip").exists():
        return "evaluate"
    return "train"


class Log:
    """Timestamped lines to stdout and to a file, flushed on every line."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def __call__(self, text):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def run_step(cmd, log_path):
    # Run one module in its own process; its output goes to log_path
    with open(log_path, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def format_summary(s):
    return (f"mean {s['mean']:.1f}  landed {s['landed_rate']:.0%}  hover {s['hover_rate']:.0%}  "
            f"grounded_timeout {s['grounded_timeout_rate']:.0%}  farming {s['farming_rate']:.0%}")


def load_summary(run_dir):
    # None if there is no (readable) eval.json yet
    try:
        return json.loads((run_dir / "eval.json").read_text())
    except (OSError, ValueError):
        return None


def do_run(reward, seed, run_dir, plan, args, log):
    # Returns None on success, or a short reason for the failure
    run_dir.mkdir(parents=True, exist_ok=True)
    module = [sys.executable, "-m"]
    if plan == "train":
        # Drop an old result first, so a crash can't leave a stale eval.json behind
        (run_dir / "eval.json").unlink(missing_ok=True)
        log(f"{reward} seed{seed}: training {args.steps} steps")
        t0 = time.perf_counter()
        code = run_step(
            module + ["rocket_land.train", "--reward", reward, "--seed", str(seed),
                      "--steps", str(args.steps), "--out", str(run_dir)],
            run_dir / "train.log",
        )
        log(f"{reward} seed{seed}: training {'done' if code == 0 else 'FAILED'} "
            f"(exit {code}, {time.perf_counter() - t0:.0f} s)")
        if code != 0:
            return f"train exit {code}"
    log(f"{reward} seed{seed}: evaluating {args.episodes} episodes")
    t0 = time.perf_counter()
    code = run_step(
        module + ["rocket_land.evaluate", "--run", str(run_dir), "--episodes", str(args.episodes),
                  "--record", "--gif", str(run_dir / "clip.gif")],
        run_dir / "evaluate.log",
    )
    log(f"{reward} seed{seed}: evaluation {'done' if code == 0 else 'FAILED'} "
        f"(exit {code}, {time.perf_counter() - t0:.0f} s)")
    if code != 0:
        return f"evaluate exit {code}"
    summary = load_summary(run_dir)
    if summary is None:
        return "eval.json missing after evaluate"
    log(f"{reward} seed{seed}: {format_summary(summary)}")
    return None


def print_table(rows, log):
    log("Final table")
    log(f"{'reward':<14}{'seed':>5}  {'status':<10}{'time':>7}{'mean':>8}{'landed':>8}"
        f"{'hover':>7}{'grounded':>9}{'farming':>8}")
    for r in rows:
        s = r["summary"]
        scores = (f"{s['mean']:>8.1f}{s['landed_rate']:>8.0%}{s['hover_rate']:>7.0%}"
                  f"{s['grounded_timeout_rate']:>9.0%}{s['farming_rate']:>8.0%}") if s else ""
        log(f"{r['reward']:<14}{r['seed']:>5}  {r['status']:<10}{r['seconds']:>6.0f}s{scores}")


def main():
    args = parse_args()
    root = args.root.resolve()
    log = Log(root / "experiment.log")
    log(f"Experiment start: rewards {args.reward}, seeds {args.seeds}, steps {args.steps}, "
        f"episodes {args.episodes}, force {args.force}, root {root}")

    start = time.perf_counter()
    rows = []
    failures = []
    for reward in args.reward:
        for seed in args.seeds:
            run_dir = root / reward / f"seed{seed}"
            plan = plan_run(run_dir, args.force)
            t0 = time.perf_counter()
            if plan == "skip":
                log(f"{reward} seed{seed}: skip (eval.json exists)")
                status = "skipped"
            else:
                if plan == "evaluate":
                    log(f"{reward} seed{seed}: model.zip found but no eval.json, evaluating only")
                reason = do_run(reward, seed, run_dir, plan, args, log)
                status = "ok" if reason is None else "FAILED"
                if reason:
                    failures.append(f"{reward} seed{seed}: {reason}")
            rows.append({"reward": reward, "seed": seed, "status": status,
                         "seconds": time.perf_counter() - t0, "summary": load_summary(run_dir)})

    log(f"Experiment end: {len(rows)} runs, {len(failures)} failed, "
        f"{(time.perf_counter() - start) / 60:.1f} min")
    print_table(rows, log)
    if failures:
        log("Failures:")
        for f in failures:
            log(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
