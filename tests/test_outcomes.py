import pytest

from rocket_land.outcomes import classify, summarize


def obs(left=0.0, right=0.0):
    # Last observation: only the leg contacts matter here, so keep the lander
    # well above the ground (y=1.0) to stay clear of the ground-level fallback
    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, left, right]


def test_landed():
    assert classify(True, False, 100.0, obs(1.0, 1.0)) == "landed"


def test_crashed():
    assert classify(True, False, -100.0, obs()) == "crashed"


def test_other():
    # Terminated but neither +100 nor -100 (e.g. flew out of bounds)
    assert classify(True, False, 3.0, obs()) == "other"
    # Still running
    assert classify(False, False, 1.0, obs()) == "other"


def test_truncated_both_legs_is_grounded_timeout():
    assert classify(False, True, -0.5, obs(1.0, 1.0)) == "grounded_timeout"


def test_truncated_left_only_is_hover():
    assert classify(False, True, -0.5, obs(left=1.0)) == "hover"


def test_truncated_right_only_is_hover():
    assert classify(False, True, -0.5, obs(right=1.0)) == "hover"


def test_truncated_no_legs_is_hover():
    assert classify(False, True, -0.5, obs()) == "hover"


def test_truncated_with_normal_reward_is_hover():
    assert classify(False, True, 1.7, obs()) == "hover"


def full_obs(x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.0, left=0.0, right=0.0):
    # Last observation with position and velocity too
    return [x, y, vx, vy, angle, 0.0, left, right]


def test_truncated_leg_flicker_at_rest_is_grounded_timeout():
    # Real baseline episode: resting level on the pad, one leg sensor flickered
    last = full_obs(x=-0.112, y=-0.001, vx=0.004, vy=0.0, angle=-0.002, left=1.0)
    assert classify(False, True, -0.5, last) == "grounded_timeout"


def test_truncated_one_leg_high_up_still_is_hover():
    assert classify(False, True, -0.5, full_obs(y=0.5, left=1.0)) == "hover"


def test_truncated_no_legs_low_but_moving_is_hover():
    # Sliding or bouncing near the ground: speed 0.3
    assert classify(False, True, -0.5, full_obs(y=0.01, vx=0.3)) == "hover"


def test_truncated_no_legs_low_and_still_is_grounded_timeout():
    assert classify(False, True, -0.5, full_obs(y=0.01)) == "grounded_timeout"


def test_summarize_known_numbers():
    episodes = [
        {"score": 200.0, "length": 100, "outcome": "landed"},
        {"score": -100.0, "length": 50, "outcome": "crashed"},
        {"score": 0.0, "length": 1000, "outcome": "hover"},
        {"score": 100.0, "length": 150, "outcome": "landed"},
    ]
    s = summarize(episodes)
    assert s["n_episodes"] == 4
    assert s["mean"] == pytest.approx(50.0)
    assert s["std"] == pytest.approx(111.803398875)  # population std
    assert s["min"] == -100.0
    assert s["max"] == 200.0
    assert s["landed_rate"] == 0.5
    assert s["crash_rate"] == 0.25
    assert s["hover_rate"] == 0.25
    assert s["grounded_timeout_rate"] == 0.0
    assert s["other_rate"] == 0.0
    assert s["mean_ep_len"] == pytest.approx(325.0)


def test_summarize_hover_and_grounded_timeout():
    episodes = [
        {"score": 200.0, "length": 100, "outcome": "landed"},
        {"score": 0.0, "length": 1000, "outcome": "hover"},
        {"score": 10.0, "length": 1000, "outcome": "grounded_timeout"},
        {"score": 20.0, "length": 1000, "outcome": "grounded_timeout"},
    ]
    s = summarize(episodes)
    assert s["hover_rate"] == 0.25
    assert s["grounded_timeout_rate"] == 0.5
    assert s["landed_rate"] == 0.25
