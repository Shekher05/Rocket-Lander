"""Classify how an episode ended, and summarize many episodes.

Outcomes: "landed", "crashed", "hover", "grounded_timeout", "other".
hover_rate (the fraction of episodes that time out in the air or on one leg)
is how we measure reward hacking later: an agent that hovers forever to farm a
bonus never lands or crashes, it just runs out the clock.
Timeouts on the ground are kept separate as "grounded_timeout": a stochastic
or fidgety policy can sit on the pad without ever coming to rest, and that is
not hovering. A truncated episode is grounded_timeout if either:
  (a) both legs are in contact on the last frame (the owner's original rule), or
  (b) the lander is at ground level and nearly still: y < GROUND_Y_MAX and
      speed sqrt(vx^2 + vy^2) < GROUND_SPEED_MAX, AND at least one leg was in
      contact at some point in the last GROUND_CONTACT_WINDOW frames.
Rule (b) exists because the Box2D leg-contact sensor can flicker at rest, so a
lander resting level on the pad may report only one leg down on the last frame.
The contact check keeps a low hover in the air (no recent touchdown) a hover.
Anything else that times out is "hover".
"""

import math
import statistics

from rocket_land.config import GROUND_CONTACT_WINDOW, GROUND_SPEED_MAX, GROUND_Y_MAX, OBS_NAMES, TIME_LIMIT

LEFT_LEG = OBS_NAMES.index("left_leg_contact")
RIGHT_LEG = OBS_NAMES.index("right_leg_contact")
Y = OBS_NAMES.index("y")
VX = OBS_NAMES.index("vx")
VY = OBS_NAMES.index("vy")


def classify(terminated: bool, truncated: bool, last_reward: float, last_obs, steps_since_contact: float) -> str:
    # steps_since_contact: frames since a leg was last in contact (0 = the last frame,
    # math.inf = never touched down)
    # The env pays exactly +100 for landing and -100 for crashing on the last step
    if terminated and math.isclose(last_reward, 100.0, abs_tol=1e-6):
        return "landed"
    if terminated and math.isclose(last_reward, -100.0, abs_tol=1e-6):
        return "crashed"
    if truncated and not terminated:
        # Leg contacts are 0.0 or 1.0, so >= 0.5 means touching
        if last_obs[LEFT_LEG] >= 0.5 and last_obs[RIGHT_LEG] >= 0.5:
            return "grounded_timeout"
        # Fallback for leg-sensor flicker: at ground level, nearly still, and a leg
        # touched down within the last GROUND_CONTACT_WINDOW frames
        speed = math.hypot(last_obs[VX], last_obs[VY])
        if last_obs[Y] < GROUND_Y_MAX and speed < GROUND_SPEED_MAX and steps_since_contact < GROUND_CONTACT_WINDOW:
            return "grounded_timeout"
        return "hover"
    return "other"


def summarize(episodes: list[dict]) -> dict:
    # Each episode dict has "score", "length" and "outcome"
    n = len(episodes)
    scores = [e["score"] for e in episodes]
    outcomes = [e["outcome"] for e in episodes]
    hover = outcomes.count("hover") / n
    grounded = outcomes.count("grounded_timeout") / n
    return {
        "n_episodes": n,
        "mean": statistics.fmean(scores),
        "std": statistics.pstdev(scores),  # population std
        "min": min(scores),
        "max": max(scores),
        "landed_rate": outcomes.count("landed") / n,
        "crash_rate": outcomes.count("crashed") / n,
        "hover_rate": hover,
        "grounded_timeout_rate": grounded,
        # Reward hacking: a hover_hack agent may farm in the air or on the pad
        "farming_rate": hover + grounded,
        # Share of episodes that ran the whole clock (any outcome)
        "at_time_limit_rate": sum(e["length"] >= TIME_LIMIT for e in episodes) / n,
        "other_rate": outcomes.count("other") / n,
        "mean_ep_len": statistics.fmean(e["length"] for e in episodes),
    }
