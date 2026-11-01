"""Shared constants for the project. Later phases add more here."""

from pathlib import Path

# Gymnasium environment we train on
ENV_ID = "LunarLander-v3"

# What each of the 8 observation values means, in order
OBS_NAMES = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)

# What each of the 4 discrete actions does, by index
ACTION_NAMES = ("noop", "fire_left", "fire_main", "fire_right")

# Training setup
N_ENVS = 16  # parallel copies of the env
TOTAL_TIMESTEPS = 1_000_000

# LunarLander truncates an episode at this many frames
TIME_LIMIT = 1000

# Evaluation episode i resets with seed EVAL_SEED_BASE + i
EVAL_SEED_BASE = 10_000

# PPO settings: the RL Zoo / Hugging Face course defaults for LunarLander
PPO_KWARGS = dict(
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
)

# The shaping discount must equal PPO's gamma
GAMMA = PPO_KWARGS["gamma"]

# Reward shaping (Phase 5, see rewards.py)
FEE = 0.03  # per-frame time fee; the owner's choice. At most -30 per 1000-frame episode
HACK_BONUS = 1.0  # hover_hack: reward per upright-and-slow frame
POTENTIAL_K = 10.0  # potential_fix: size of the potential for an upright-and-slow state

# "Upright and slow" for both hover_hack and potential_fix
UPRIGHT_MAX_ANGLE = 0.2  # radians
SLOW_MAX_SPEED = 0.3  # observation units, speed = hypot(vx, vy)

# Outcome classification fallback (observation units). The Box2D leg-contact
# sensors can flicker at rest, so a lander sitting level on the pad may report
# only one leg down. A truncated episode that ends below GROUND_Y_MAX and slower
# than GROUND_SPEED_MAX is treated as grounded even if the legs disagree, but only
# if a leg touched down within the last GROUND_CONTACT_WINDOW frames (so a low
# hover in the air still counts as a hover).
GROUND_Y_MAX = 0.05
GROUND_SPEED_MAX = 0.05
GROUND_CONTACT_WINDOW = 50  # frames, counting the final one

# Project root (the folder above this package) and where training output goes
ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
