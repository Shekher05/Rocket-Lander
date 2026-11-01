import json
import statistics

import pytest

from rocket_land.outcomes import summarize
from rocket_land.report import (
    GIFS,
    REWARD_ORDER,
    SEEDS,
    MissingRunsError,
    build_report,
    load_runs,
    main,
)


def write_run(root, reward, seed, scores=(100.0,) * 8, outcomes=None, lengths=None):
    # A fake eval.json shaped like the one evaluate.py writes (same summarize())
    outcomes = outcomes or ["landed"] * len(scores)
    lengths = lengths or [300] * len(scores)
    eps = [{"score": s, "length": n, "outcome": o} for s, n, o in zip(scores, lengths, outcomes)]
    data = {
        **summarize(eps),
        "settings": {"episodes": len(eps), "deterministic": True, "seed_base": 10_000, "env": "LunarLander-v3"},
        "episodes": eps,
    }
    run_dir = root / reward / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval.json").write_text(json.dumps(data))


def write_all(root, mean_by_reward=None):
    mean_by_reward = mean_by_reward or {}
    for reward in REWARD_ORDER:
        for seed in SEEDS:
            m = mean_by_reward.get(reward, 250.0) + 10.0 * seed
            write_run(root, reward, seed, scores=(m - 5, m + 5) * 4)


def table_rows(text):
    # Markdown table lines of the main results table (start with "| builtin" etc.)
    return [line for line in text.splitlines() if any(line.startswith(f"| {r} ") for r in REWARD_ORDER)]


def test_missing_runs_fail_loudly_and_name_every_one(tmp_path):
    write_all(tmp_path)
    (tmp_path / "hover_hack" / "seed1" / "eval.json").unlink()
    (tmp_path / "potential_fix" / "seed2" / "eval.json").unlink()
    with pytest.raises(MissingRunsError) as e:
        load_runs(tmp_path)
    assert "hover_hack/seed1" in str(e.value)
    assert "potential_fix/seed2" in str(e.value)


def test_empty_root_fails(tmp_path):
    with pytest.raises(MissingRunsError):
        load_runs(tmp_path)


def test_unreadable_eval_json_fails(tmp_path):
    write_all(tmp_path)
    (tmp_path / "builtin" / "seed0" / "eval.json").write_text("{not json")
    with pytest.raises(MissingRunsError):
        load_runs(tmp_path)


def test_dryrun_and_smoke_are_ignored(tmp_path):
    write_all(tmp_path)
    for name in ("_dryrun", "_smoke"):
        write_run(tmp_path, name, 0, scores=(-999.0, -999.0))
    runs = load_runs(tmp_path)
    assert sorted(runs) == sorted((r, s) for r in REWARD_ORDER for s in SEEDS)
    assert "-999" not in build_report(runs)


def test_inconsistent_settings_fail(tmp_path):
    write_all(tmp_path)
    path = tmp_path / "builtin" / "seed1" / "eval.json"
    data = json.loads(path.read_text())
    data["settings"]["deterministic"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="settings"):
        load_runs(tmp_path)


def test_12_runs_plus_4_summary_rows(tmp_path):
    write_all(tmp_path)
    text = build_report(load_runs(tmp_path))
    rows = table_rows(text)
    assert len(rows) == 16
    summary = [r for r in rows if "mean ± std" in r]
    assert len(summary) == 4
    assert len([r for r in rows if "mean ± std" not in r]) == 12


def test_summary_uses_sample_std_across_seeds(tmp_path):
    # Seed means 100, 200, 300: sample std is 100, population std would be 81.6
    for seed, m in zip(SEEDS, (100.0, 200.0, 300.0)):
        write_run(tmp_path, "builtin", seed, scores=(m,) * 8)
    for reward in REWARD_ORDER[1:]:
        for seed in SEEDS:
            write_run(tmp_path, reward, seed)
    text = build_report(load_runs(tmp_path))
    row = [r for r in table_rows(text) if r.startswith("| builtin") and "mean ± std" in r][0]
    assert "200.0 ± 100.0" in row
    assert "81.6" not in row
    assert "sample std" in text  # the report says which std it used


def test_numbers_come_from_eval_json(tmp_path):
    write_all(tmp_path)
    # 4 landed, 1 crashed, 1 hover, 2 grounded_timeout out of 8 episodes
    outcomes = ["landed"] * 4 + ["crashed", "hover", "grounded_timeout", "grounded_timeout"]
    lengths = [300] * 4 + [100, 1000, 1000, 1000]
    write_run_over = tmp_path / "time_fee" / "seed1" / "eval.json"
    write_run_over.unlink()
    write_run(tmp_path, "time_fee", 1, scores=[10.0] * 8, outcomes=outcomes, lengths=lengths)
    text = build_report(load_runs(tmp_path))
    row = [r for r in table_rows(text) if r.startswith("| time_fee | 1 ")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    # reward, seed, mean, std, landed %, crashed %, hover %, grounded %, farming %, at limit %, mean len
    assert cells[2] == "10.0"
    assert cells[4] == "50.0"  # landed
    assert cells[5] == "12.5"  # crashed
    assert cells[6] == "12.5"  # hover
    assert cells[7] == "25.0"  # grounded_timeout
    assert cells[8] == "37.5"  # farming = hover + grounded
    assert cells[9] == "37.5"  # at the 1000 frame limit
    assert cells[10] == str(statistics.fmean(lengths).__round__(1))


def test_pass_bar_counts_seeds(tmp_path):
    write_all(tmp_path, {"hover_hack": 50.0, "potential_fix": 190.0})
    # potential_fix means are 190, 200, 210 -> 2 of 3 at or above 200
    text = build_report(load_runs(tmp_path))
    assert "potential_fix: 2 of 3 seeds" in text
    assert "hover_hack: 0 of 3 seeds" in text
    assert "builtin: 3 of 3 seeds" in text


def test_clip_section_reads_episode_0_from_eval_json(tmp_path):
    write_all(tmp_path)
    path = tmp_path / "hover_hack" / "seed2" / "eval.json"
    path.unlink()
    write_run(tmp_path, "hover_hack", 2, scores=[-120.9] + [5.0] * 7, outcomes=["hover"] + ["landed"] * 7, lengths=[1000] + [300] * 7)
    text = build_report(load_runs(tmp_path))
    gif_reward, gif_seed = GIFS["hover_hack"]
    assert (gif_reward, gif_seed) == ("hover_hack", 2)
    line = [l for l in text.splitlines() if l.startswith("| hover_hack.gif")][0]
    assert "seed 2" in line
    assert "10000" in line  # env seed of episode 0
    assert "hover" in line
    assert "-120.9" in line
    assert "1000" in line


def test_failed_report_does_not_touch_existing_file(tmp_path):
    runs = tmp_path / "runs"
    out = tmp_path / "out"
    out.mkdir()
    (out / "RESULTS.md").write_text("old")
    write_all(runs)
    (runs / "builtin" / "seed2" / "eval.json").unlink()
    with pytest.raises(SystemExit):
        main(["--runs", str(runs), "--out", str(out)])
    assert (out / "RESULTS.md").read_text() == "old"


def test_main_writes_results_md(tmp_path):
    runs = tmp_path / "runs"
    out = tmp_path / "out"
    write_all(runs)
    main(["--runs", str(runs), "--out", str(out)])
    text = (out / "RESULTS.md").read_text()
    assert text.startswith("# Results")
    assert len(table_rows(text)) == 16
