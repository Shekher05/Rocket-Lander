"""Phase 7: turn the eval.json files of all runs into results/RESULTS.md.

Every number comes from runs/<reward>/seed<k>/eval.json. If any of the 12 expected
runs is missing or unreadable, the report fails with a list of what is missing
and writes nothing (an old RESULTS.md is left as it was).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from rocket_land.config import ROOT, RUNS_DIR, TIME_LIMIT

# Only these runs are reported; anything else in runs/ (_dryrun, _smoke, ...) is ignored
REWARD_ORDER = ("builtin", "time_fee", "hover_hack", "potential_fix")
SEEDS = (0, 1, 2)

PASS_SCORE = 200  # same bar as evaluate.py: a run passes at mean score >= 200

# GIF file -> (reward, seed) of the run whose episode 0 it shows.
# builtin: seed 0 (the baseline run; its GIF was recorded on 2026-10-30).
# hover_hack: seed 2, the clip the owner watched and confirmed does not land.
# potential_fix: rule fixed in advance, the seed with the median mean score of the
#   three (the middle seed, neither the best nor the worst), not the best-looking clip.
GIFS = {
    "builtin": ("builtin", 0),
    "hover_hack": ("hover_hack", 2),
    "potential_fix": ("potential_fix", 2),
}

# Columns of the main table, after "reward" and "seed": (header, eval.json key, scale)
RATE_COLUMNS = (
    ("landed %", "landed_rate"),
    ("crashed %", "crash_rate"),
    ("hover %", "hover_rate"),
    ("grounded_timeout %", "grounded_timeout_rate"),
    ("farming %", "farming_rate"),
    (f"at {TIME_LIMIT}-frame limit %", "at_time_limit_rate"),
)


class MissingRunsError(Exception):
    """One or more expected runs have no readable eval.json."""


def load_runs(runs_dir: Path) -> dict:
    # {(reward, seed): eval.json contents}; raises if any expected run is missing
    runs = {}
    problems = []
    for reward in REWARD_ORDER:
        for seed in SEEDS:
            path = Path(runs_dir) / reward / f"seed{seed}" / "eval.json"
            try:
                runs[(reward, seed)] = json.loads(path.read_text())
            except (OSError, ValueError) as e:
                problems.append(f"{reward}/seed{seed}: {type(e).__name__} reading {path}")
    if problems:
        raise MissingRunsError(
            f"{len(problems)} of {len(REWARD_ORDER) * len(SEEDS)} expected runs are missing or unreadable:\n  "
            + "\n  ".join(problems)
        )
    # The table is only comparable if every run was evaluated the same way
    settings = {json.dumps(r["settings"], sort_keys=True) for r in runs.values()}
    if len(settings) != 1:
        raise ValueError(f"runs were evaluated with different settings: {sorted(settings)}")
    return runs


def mean_std(values):
    # Sample std (n - 1) across seeds
    return statistics.fmean(values), statistics.stdev(values)


def pm(values, scale=1.0):
    m, s = mean_std(values)
    return f"{m * scale:.1f} ± {s * scale:.1f}"


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def results_table(runs):
    headers = ["reward", "seed", "mean score", "std within run"]
    headers += [h for h, _ in RATE_COLUMNS] + ["mean episode length"]
    lines = [row(headers), row(["---"] * len(headers))]
    for reward in REWARD_ORDER:
        group = [runs[(reward, s)] for s in SEEDS]
        for seed, r in zip(SEEDS, group):
            lines.append(row(
                [reward, seed, f"{r['mean']:.1f}", f"{r['std']:.1f}"]
                + [f"{r[key] * 100:.1f}" for _, key in RATE_COLUMNS]
                + [f"{r['mean_ep_len']:.1f}"]
            ))
        lines.append(row(
            [reward, "**mean ± std**", pm([r["mean"] for r in group]), "–"]
            + [pm([r[key] for r in group], 100) for _, key in RATE_COLUMNS]
            + [pm([r["mean_ep_len"] for r in group])]
        ))
    return "\n".join(lines)


def counts(r):
    # Outcome counts for one run, straight from its episodes
    outcomes = [e["outcome"] for e in r["episodes"]]
    return {o: outcomes.count(o) for o in ("landed", "crashed", "hover", "grounded_timeout", "other")}


def pass_bar_lines(runs):
    lines = []
    for reward in REWARD_ORDER:
        passing = [s for s in SEEDS if runs[(reward, s)]["mean"] >= PASS_SCORE]
        which = ", ".join(map(str, passing)) if passing else "none"
        lines.append(f"- {reward}: {len(passing)} of {len(SEEDS)} seeds with mean score ≥ {PASS_SCORE} (seeds: {which})")
    return lines


def below_bar_lines(runs):
    # Spell out every run under the bar, so a pass by the 2-of-3 rule is not read as 3 of 3
    lines = []
    for reward in REWARD_ORDER:
        for s in SEEDS:
            r = runs[(reward, s)]
            if r["mean"] < PASS_SCORE:
                c = counts(r)
                detail = ", ".join(f"{n} {o}" for o, n in c.items() if n)
                lines.append(
                    f"- {reward} seed {s}: mean {r['mean']:.1f}, farming {r['farming_rate'] * 100:.1f}%, "
                    f"outcomes of {r['n_episodes']} episodes: {detail}"
                )
    return lines


def farming_range_lines(runs):
    lines = []
    for reward in REWARD_ORDER:
        by_seed = ", ".join(f"seed {s}: {runs[(reward, s)]['farming_rate'] * 100:.1f}%" for s in SEEDS)
        lines.append(f"- {reward}: {by_seed}")
    return lines


def clips_table(runs):
    lines = [
        row(["file", "source run", "env seed", "outcome", "score", "steps"]),
        row(["---"] * 6),
    ]
    for name, (reward, seed) in GIFS.items():
        r = runs[(reward, seed)]
        ep = r["episodes"][0]
        env_seed = r["settings"]["seed_base"]  # episode i resets with seed_base + i, so episode 0 is seed_base
        lines.append(row([
            f"{name}.gif", f"{reward} seed {seed}", env_seed, ep["outcome"], f"{ep['score']:.1f}", ep["length"],
        ]))
    return "\n".join(lines)


def build_report(runs) -> str:
    any_run = next(iter(runs.values()))
    st = any_run["settings"]
    n = any_run["n_episodes"]
    seed_base = st["seed_base"]
    return f"""# Results

Generated by `python -m rocket_land.report` from `runs/<reward>/seed<k>/eval.json`. Do not edit by hand.

## How the numbers were made

- {len(REWARD_ORDER)} rewards x {len(SEEDS)} training seeds = {len(runs)} runs. Each run was evaluated on {n} episodes ({st['env']}, env seeds {seed_base} to {seed_base + n - 1}), {'deterministic' if st['deterministic'] else 'stochastic'} policy, with the plain built-in reward and no reward wrapper. So the score is always the built-in score, even for `hover_hack` and `potential_fix`.
- The `std within run` column is the population std of the {n} episode scores of that one run (as written by `evaluate.py`).
- The `mean ± std` rows use the **sample std (n - 1)** across the {len(SEEDS)} seeds, applied to each column (rates are in percentage points). With only {len(SEEDS)} seeds this is a rough spread, not a confidence interval.
- landed / crashed / hover / grounded_timeout are the outcome labels of `rocket_land/outcomes.py`. `farming %` = hover % + grounded_timeout % (a run that times out in the air or on the pad). `at {TIME_LIMIT}-frame limit %` counts episodes that used all {TIME_LIMIT} frames, whatever their label. Episodes can be labelled landed and still be very long, so read the mean episode length next to farming.

## Table

{results_table(runs)}

## Pass bar (mean score ≥ {PASS_SCORE})

{chr(10).join(pass_bar_lines(runs))}

Runs under the bar:

{chr(10).join(below_bar_lines(runs)) or "- none"}

## Farming rate by seed

{chr(10).join(farming_range_lines(runs))}

Seeds can behave very differently under the same reward. Compare rows, not only the summary rows.

## Clips

Each GIF is episode 0 of one run: env seed {seed_base}, deterministic policy. An episode-0 clip shows one episode of one seed. It does not represent the other seeds, or the other episodes of that run.

{clips_table(runs)}

- `builtin.gif`: made on 2026-10-30 by `evaluate.py --gif` from the `builtin` seed 0 model (that run has no `clip.gif` of its own). Not re-recorded.
- `hover_hack.gif`: copied from `runs/hover_hack/seed2/clip.gif`, the clip that was watched and confirmed not to land.
- `potential_fix.gif`: copied from `runs/potential_fix/seed2/clip.gif`. Chosen by a rule, not by how the clip looks: take the seed whose mean score is the median of the three seeds, so it is neither the best nor the worst run. Its episode 0 outcome is in the table above. The other seeds of this reward, including the weakest one, are not shown by this clip.
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Write results/RESULTS.md from the eval.json files.")
    p.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder holding <reward>/seed<k>/eval.json")
    p.add_argument("--out", type=Path, default=ROOT / "results", help="folder to write RESULTS.md into")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        text = build_report(load_runs(args.runs))
    except (MissingRunsError, ValueError) as e:
        sys.exit(f"report failed, RESULTS.md not written:\n{e}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "RESULTS.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path} ({len(REWARD_ORDER) * len(SEEDS)} runs + {len(REWARD_ORDER)} summary rows)")


if __name__ == "__main__":
    main()
