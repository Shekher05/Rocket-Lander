from types import SimpleNamespace

import pytest
from stable_baselines3.common.logger import configure

from rocket_land.train import ShapedReturnCallback


def make_callback():
    cb = ShapedReturnCallback()
    # cb.logger is read from cb.model.logger, so a stub model is enough
    cb.model = SimpleNamespace(logger=configure(None, []))
    return cb


def step(cb, infos):
    cb.locals = {"infos": infos}
    assert cb._on_step() is True


def test_collects_only_finished_episodes():
    cb = make_callback()
    step(cb, [{"shaped_return": -10.0}, {}, {"shaped_return": 30.0}])
    step(cb, [{}, {}])
    assert list(cb.returns) == [-10.0, 30.0]


def test_rollout_end_logs_mean_over_all_steps():
    cb = make_callback()
    step(cb, [{"shaped_return": 10.0}, {}])
    step(cb, [{}, {"shaped_return": 20.0}])
    step(cb, [{"shaped_return": 60.0}, {}])
    cb._on_rollout_end()
    assert cb.logger.name_to_value["custom/shaped_ep_return"] == pytest.approx(30.0)


def test_keeps_last_100_episodes():
    cb = make_callback()
    for i in range(150):
        step(cb, [{"shaped_return": float(i)}])
    assert len(cb.returns) == 100
    cb._on_rollout_end()
    assert cb.logger.name_to_value["custom/shaped_ep_return"] == pytest.approx(sum(range(50, 150)) / 100)


def test_rollout_end_logs_nothing_when_empty():
    cb = make_callback()
    cb._on_rollout_end()
    assert "custom/shaped_ep_return" not in cb.logger.name_to_value
