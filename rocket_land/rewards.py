"""Reward variants, by name. A value of None means the plain built-in reward.

This project shows reward hacking. The built-in reward pays for improving a
"position score" (minus fuel, +/-100 at the end). We add small extras on top:

- time_fee: a small cost per frame. It should still land.
- hover_hack: +1 for every frame the lander is upright and slow. A deliberately
  bad, state-based bonus. Hovering for 1000 frames earns about +1000, while fuel
  costs at most about 300, so the agent may learn to hover instead of land.
- potential_fix: the same upright-and-slow signal, but paid as a *change*
  (potential-based shaping, Ng, Harada & Russell 1999), plus the per-frame fee.
  It cannot be farmed by loitering.

Each wrapper adds a bonus to the built-in reward. The bonus and the original
reward are both put in `info`, so we can always see what was added.
"""

import math

import gymnasium

from rocket_land.config import (
    FEE,
    GAMMA,
    HACK_BONUS,
    OBS_NAMES,
    POTENTIAL_K,
    SLOW_MAX_SPEED,
    UPRIGHT_MAX_ANGLE,
)

ANGLE = OBS_NAMES.index("angle")
VX = OBS_NAMES.index("vx")
VY = OBS_NAMES.index("vy")


def upright_and_slow(obs):
    # Tilted less than UPRIGHT_MAX_ANGLE and moving slower than SLOW_MAX_SPEED
    return abs(obs[ANGLE]) < UPRIGHT_MAX_ANGLE and math.hypot(obs[VX], obs[VY]) < SLOW_MAX_SPEED


class ShapedReward(gymnasium.Wrapper):
    """Adds self.bonus(...) to the built-in reward. Subclasses define bonus()."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.shaped_return = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self.bonus(self.prev_obs, obs, terminated, truncated)
        shaped = reward + bonus

        info["base_reward"] = reward  # what the built-in reward paid
        info["bonus"] = bonus  # what we added on top
        self.shaped_return += shaped
        if terminated or truncated:
            info["shaped_return"] = self.shaped_return  # the whole episode, for logging

        self.prev_obs = obs
        return obs, shaped, terminated, truncated, info

    def bonus(self, prev_obs, obs, terminated, truncated):
        raise NotImplementedError


class TimeFee(ShapedReward):
    # Pays for: nothing, it only charges FEE every frame.
    # Exploitable by: ending the episode early, because every extra frame costs.
    #   The fee is small (at most -30 per 1000 frames), so landing still wins.
    def bonus(self, prev_obs, obs, terminated, truncated):
        return -FEE


class HoverHack(ShapedReward):
    # Pays for: every frame that is upright and slow, wherever the lander is.
    # Exploitable by: hovering. 1000 frames earn about +1000, more than a landing.
    def bonus(self, prev_obs, obs, terminated, truncated):
        return HACK_BONUS if upright_and_slow(obs) else 0.0


def phi(obs):
    # The potential: POTENTIAL_K in an upright-and-slow state, else 0
    return POTENTIAL_K * float(upright_and_slow(obs))


class PotentialFix(ShapedReward):
    # Pays for: moving into an upright-and-slow state (the change in potential),
    #   and charges FEE every frame.
    # Exploitable by: not really. Gains and losses cancel out over an episode
    #   (the sum telescopes), so staying in the state pays nothing extra.
    def bonus(self, prev_obs, obs, terminated, truncated):
        # A real end (landed or crashed) has potential 0 afterwards. Truncation is
        # only the time limit, not a true terminal, so the state still has its
        # potential there. Zeroing it would punish the agent for the clock.
        phi_next = 0.0 if terminated else phi(obs)
        return GAMMA * phi_next - phi(prev_obs) - FEE


REWARDS = {
    "builtin": None,
    "time_fee": TimeFee,
    "hover_hack": HoverHack,
    "potential_fix": PotentialFix,
}
