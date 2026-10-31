"""Phase 2: watch a random agent play LunarLander and inspect what it sees."""

import argparse

import gymnasium as gym

from rocket_land.config import ENV_ID, OBS_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Run random episodes and print observations.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-every", type=int, default=20, help="print observation every N frames")
    p.add_argument("--no-render", action="store_true", help="do not open a game window")
    return p.parse_args()


def main():
    args = parse_args()

    # "human" opens a window; None runs headless
    render_mode = None if args.no_render else "human"
    env = gym.make(ENV_ID, render_mode=render_mode)
    env.action_space.seed(args.seed)

    leg_values_seen = set()  # every distinct leg-contact value we observe

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        reward = 0.0
        frames = 0
        left_touched = right_touched = False
        end_reason = "running"
        done = False

        while not done:
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, _ = env.step(action)
            frames += 1
            total_reward += reward

            # Track the two leg-contact values (last two entries of obs)
            leg_values_seen.update((float(obs[6]), float(obs[7])))
            left_touched |= obs[6] == 1.0
            right_touched |= obs[7] == 1.0

            if frames % args.print_every == 0:
                values = "  ".join(f"{n}={v:.3f}" for n, v in zip(OBS_NAMES, obs))
                print(f"ep {ep + 1} frame {frames}: {values}")

            done = terminated or truncated
            end_reason = "terminated" if terminated else "truncated"

        print(
            f"Episode {ep + 1}: frames={frames} total_reward={total_reward:.2f} "
            f"last_reward={reward:.2f} end={end_reason} "
            f"left_leg_touched={left_touched} right_leg_touched={right_touched}"
        )

    only_binary = leg_values_seen <= {0.0, 1.0}
    print(f"Leg contact values seen: {sorted(leg_values_seen)} -> only 0.0 or 1.0: {only_binary}")

    env.close()


if __name__ == "__main__":
    main()
