import gymnasium
import numpy as np
import pytest

from rocket_land import rewards
from rocket_land.config import FEE, GAMMA, HACK_BONUS, OBS_NAMES
from rocket_land.rewards import (
    REWARDS,
    HoverHack,
    PotentialFix,
    TimeFee,
    phi,
    upright_and_slow,
)

ALL_WRAPPERS = [TimeFee, HoverHack, PotentialFix]


def make_obs(angle=0.0, vx=0.0, vy=0.0):
    obs = np.zeros(len(OBS_NAMES), dtype=np.float32)
    obs[OBS_NAMES.index("angle")] = angle
    obs[OBS_NAMES.index("vx")] = vx
    obs[OBS_NAMES.index("vy")] = vy
    return obs


GOOD = make_obs()  # upright and slow
TILTED = make_obs(angle=0.5)
FAST = make_obs(vy=-0.8)


class ScriptedEnv(gymnasium.Env):
    """Replays a list of (obs, reward, terminated, truncated), starting from obs0."""

    observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (len(OBS_NAMES),), np.float32)
    action_space = gymnasium.spaces.Discrete(4)

    def __init__(self, obs0, script):
        self.obs0 = obs0
        self.script = script

    def reset(self, *, seed=None, options=None):
        self.i = 0
        return self.obs0, {}

    def step(self, action):
        obs, reward, terminated, truncated = self.script[self.i]
        self.i += 1
        return obs, reward, terminated, truncated, {}


def run(wrapper, obs0, script):
    # Play the whole script through the wrapper; return the list of step infos and rewards
    env = wrapper(ScriptedEnv(obs0, script))
    env.reset()
    infos, rewards_out = [], []
    for _ in script:
        _, r, _, _, info = env.step(0)
        infos.append(info)
        rewards_out.append(r)
    return infos, rewards_out


def test_registry_names():
    assert set(REWARDS) == {"builtin", "time_fee", "hover_hack", "potential_fix"}
    assert REWARDS["builtin"] is None


def test_upright_and_slow():
    assert upright_and_slow(GOOD)
    assert not upright_and_slow(TILTED)
    assert not upright_and_slow(FAST)
    assert not upright_and_slow(make_obs(angle=-0.5))  # tilted the other way


def test_time_fee_subtracts_fee_every_step():
    script = [(GOOD, 1.0, False, False), (TILTED, -2.0, False, False), (FAST, 0.5, False, True)]
    infos, rs = run(TimeFee, GOOD, script)
    for (_, base, _, _), info, r in zip(script, infos, rs):
        assert info["bonus"] == pytest.approx(-FEE)
        assert r == pytest.approx(base - FEE)


def test_hover_hack_pays_only_on_upright_and_slow_frames():
    script = [(GOOD, 0.0, False, False), (TILTED, 0.0, False, False), (FAST, 0.0, False, False),
              (GOOD, 0.0, False, True)]
    infos, _ = run(HoverHack, GOOD, script)
    assert [i["bonus"] for i in infos] == [HACK_BONUS, 0.0, 0.0, HACK_BONUS]


@pytest.mark.parametrize("wrapper", ALL_WRAPPERS)
def test_base_reward_is_the_inner_reward(wrapper):
    script = [(GOOD, 1.5, False, False), (TILTED, -3.0, False, False), (FAST, 100.0, True, False)]
    infos, _ = run(wrapper, GOOD, script)
    assert [i["base_reward"] for i in infos] == [1.5, -3.0, 100.0]


@pytest.mark.parametrize("wrapper", ALL_WRAPPERS)
def test_shaped_return_only_on_last_step_and_equals_sum(wrapper):
    script = [(GOOD, 1.0, False, False), (TILTED, -2.0, False, False), (GOOD, 0.5, False, False),
              (FAST, -100.0, True, False)]
    infos, rs = run(wrapper, GOOD, script)
    assert all("shaped_return" not in i for i in infos[:-1])
    assert infos[-1]["shaped_return"] == pytest.approx(sum(rs))


def test_shaped_return_resets_each_episode():
    script = [(GOOD, 1.0, False, True)]
    env = TimeFee(ScriptedEnv(GOOD, script))
    for _ in range(2):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["shaped_return"] == pytest.approx(1.0 - FEE)


def test_potential_fix_telescopes(monkeypatch):
    # With the fee off, each bonus is GAMMA*phi(next) - phi(prev). Discounted by
    # GAMMA**t, the middle terms cancel and only -phi(s0) is left (phi after a real
    # terminal is 0). So whatever path the lander takes, the total discounted bonus
    # is fixed by where it started. Looping in and out of the good state, or
    # sitting in it, gains nothing. That is why this reward can't be farmed.
    monkeypatch.setattr(rewards, "FEE", 0.0)
    path = [GOOD, TILTED, GOOD, GOOD, FAST, GOOD, TILTED, FAST]
    script = [(o, 0.0, False, False) for o in path[:-1]] + [(path[-1], -100.0, True, False)]
    for obs0 in (GOOD, TILTED):
        infos, _ = run(PotentialFix, obs0, script)
        discounted = sum(GAMMA**t * i["bonus"] for t, i in enumerate(infos))
        assert discounted == pytest.approx(-phi(obs0), abs=1e-6)


def test_loitering_hover_hack_pays_potential_fix_does_not():
    # 1000 frames of sitting upright and slow, then the time limit
    script = [(GOOD, 0.0, False, False)] * 999 + [(GOOD, 0.0, False, True)]
    hack, _ = run(HoverHack, GOOD, script)
    fix, _ = run(PotentialFix, GOOD, script)
    hack_total = sum(i["bonus"] for i in hack)
    fix_total = sum(i["bonus"] for i in fix)
    print(f"\nloitering 1000 frames: hover_hack bonus = {hack_total:.2f}, potential_fix bonus = {fix_total:.2f}")

    assert hack_total == pytest.approx(1000 * HACK_BONUS)
    # potential_fix only pays the fee plus a small discount leak (GAMMA - 1) * phi
    # per frame, both costs. It never goes positive and stays small.
    assert fix_total <= 0
    assert fix_total >= -(1000 * FEE + 1000 * (1 - GAMMA) * phi(GOOD) + 1e-6)


class MultiEpisodeEnv(ScriptedEnv):
    """Like ScriptedEnv, but each reset() starts the next (obs0, script) episode."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.n_resets = 0

    def reset(self, *, seed=None, options=None):
        self.obs0, self.script = self.episodes[self.n_resets]
        self.n_resets += 1
        return super().reset(seed=seed, options=options)


def test_potential_fix_terminal_in_good_state_has_zero_phi_next():
    # Terminating while upright and slow: phi_next must be 0 (a real end), so the
    # last bonus is -phi(prev) - FEE. Using phi(obs) there would give GAMMA*K - K - FEE.
    script = [(GOOD, 0.0, False, False), (GOOD, 0.0, False, False), (GOOD, 0.0, True, False)]
    infos, _ = run(PotentialFix, GOOD, script)
    assert infos[-1]["bonus"] == pytest.approx(-phi(GOOD) - FEE)


def test_potential_fix_telescopes_to_minus_phi_start_when_terminal_in_good_state(monkeypatch):
    monkeypatch.setattr(rewards, "FEE", 0.0)
    script = [(GOOD, 0.0, False, False), (TILTED, 0.0, False, False), (GOOD, 0.0, False, False),
              (GOOD, 0.0, True, False)]
    infos, _ = run(PotentialFix, GOOD, script)
    discounted = sum(GAMMA**t * i["bonus"] for t, i in enumerate(infos))
    assert discounted == pytest.approx(-phi(GOOD), abs=1e-6)


def test_potential_fix_prev_obs_resets_between_episodes():
    # Episode 1 ends in GOOD (phi = K). Episode 2 starts in TILTED (phi = 0), so its
    # first bonus is GAMMA*0 - 0 - FEE. A stale prev_obs would add -K.
    ep1 = (GOOD, [(GOOD, 0.0, False, True)])
    ep2 = (TILTED, [(TILTED, 0.0, False, False)])
    env = PotentialFix(MultiEpisodeEnv([ep1, ep2]))
    env.reset()
    env.step(0)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["bonus"] == pytest.approx(GAMMA * phi(TILTED) - phi(TILTED) - FEE)
    assert info["bonus"] == pytest.approx(-FEE)
