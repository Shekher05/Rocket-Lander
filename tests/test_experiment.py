from rocket_land.experiment import plan_run


def test_empty_dir_trains(tmp_path):
    assert plan_run(tmp_path, False) == "train"


def test_missing_dir_trains(tmp_path):
    assert plan_run(tmp_path / "nope", False) == "train"


def test_model_without_eval_only_evaluates(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"x")
    assert plan_run(tmp_path, False) == "evaluate"


def test_eval_json_skips(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"x")
    (tmp_path / "eval.json").write_text("{}")
    assert plan_run(tmp_path, False) == "skip"


def test_eval_without_model_still_skips(tmp_path):
    (tmp_path / "eval.json").write_text("{}")
    assert plan_run(tmp_path, False) == "skip"


def test_force_always_trains(tmp_path):
    assert plan_run(tmp_path, True) == "train"
    (tmp_path / "model.zip").write_bytes(b"x")
    assert plan_run(tmp_path, True) == "train"
    (tmp_path / "eval.json").write_text("{}")
    assert plan_run(tmp_path, True) == "train"
