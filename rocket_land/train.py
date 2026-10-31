"""Phase 3: train a PPO agent on LunarLander and save the model + config."""

import argparse
import json
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from rocket_land.config import (
    ENV_ID,
    FEE,
    HACK_BONUS,
    N_ENVS,
    POTENTIAL_K,
    PPO_KWARGS,
    ROOT,
    RUNS_DIR,
    SLOW_MAX_SPEED,
    TOTAL_TIMESTEPS,
    UPRIGHT_MAX_ANGLE,
)
from rocket_land.rewards import REWARDS


def parse_args():
    p = argparse.ArgumentParser(description="Train a PPO agent.")
    p.add_argument("--reward", choices=list(REWARDS), default="builtin")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=N_ENVS)
    p.add_argument("--out", type=Path, default=None, help="output dir (default runs/<reward>/seed<seed>)")
    return p.parse_args()


def git_hash():
    # None if there is no git, or no commits yet
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def wrapper_chain(env):
    # Class names from the outermost wrapper to the innermost env
    names = []
    while True:
        names.append(type(env).__name__)
        if not hasattr(env, "env"):
            break
        env = env.env
    return " -> ".join(names)


def make_env(reward, seed, n_envs, monitor_dir):
    # Monitor sits directly around the env (it logs the built-in reward);
    # the reward wrapper, if any, goes outside it
    return make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        wrapper_class=REWARDS[reward],
    )


class ShapedReturnCallback(BaseCallback):
    """Logs the mean shaped return of the last 100 episodes, like rollout/ep_rew_mean."""

    def __init__(self):
        super().__init__()
        self.returns = deque(maxlen=100)

    def _on_step(self):
        # Collect every finished episode's shaped return
        self.returns.extend(info["shaped_return"] for info in self.locals["infos"] if "shaped_return" in info)
        return True

    def _on_rollout_end(self):
        # Once per rollout, just before SB3 dumps the logs
        if self.returns:
            self.logger.record("custom/shaped_ep_return", sum(self.returns) / len(self.returns))


def main():
    args = parse_args()
    out = args.out or RUNS_DIR / args.reward / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    vec_env = make_env(args.reward, args.seed, args.n_envs, str(out / "monitor"))

    # TRD check 3.2: Monitor should sit directly around the LunarLander env,
    # with any reward wrapper outside it
    print("Wrapper chain:", wrapper_chain(vec_env.envs[0]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(out / "tb"),
        **PPO_KWARGS,
    )

    start = time.perf_counter()
    callback = None if args.reward == "builtin" else ShapedReturnCallback()
    model.learn(total_timesteps=args.steps, callback=callback)
    elapsed = time.perf_counter() - start

    model.save(out / "model.zip")
    config = {
        "reward": args.reward,
        "seed": args.seed,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "ppo_kwargs": PPO_KWARGS,
        "reward_constants": {
            "FEE": FEE,
            "HACK_BONUS": HACK_BONUS,
            "POTENTIAL_K": POTENTIAL_K,
            "UPRIGHT_MAX_ANGLE": UPRIGHT_MAX_ANGLE,
            "SLOW_MAX_SPEED": SLOW_MAX_SPEED,
        },
        "elapsed_seconds": elapsed,
        "versions": {
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
        "git_hash": git_hash(),
        "timestamp": datetime.now().isoformat(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Saved model:  {out / 'model.zip'}")
    print(f"Saved config: {out / 'config.json'}")
    print(f"Monitor logs: {out / 'monitor'}")
    print(f"TensorBoard:  {out / 'tb'}")
    print(f"Elapsed: {elapsed:.1f} s")

    vec_env.close()


if __name__ == "__main__":
    main()
