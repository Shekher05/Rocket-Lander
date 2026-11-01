"""Phase 3: evaluate a trained model on the plain LunarLander env."""

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from rocket_land.config import ENV_ID, EVAL_SEED_BASE
from rocket_land.outcomes import LEFT_LEG, RIGHT_LEG, classify, summarize

PASS_SCORE = 200  # LunarLander counts as solved at a mean score of 200
RECORD_EPISODES = 3  # --record saves the first few episodes
GIF_FPS = 25
GIF_MAX_MB = 5


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    p.add_argument("--run", type=Path, required=True, help="run dir containing model.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the best one")
    p.add_argument("--record", action="store_true", help=f"save MP4s of the first {RECORD_EPISODES} episodes to <run>/video/")
    p.add_argument("--watch", action="store_true", help="show a live window (cannot be used with --record)")
    p.add_argument("--gif", type=Path, default=None, help="save a GIF of the first episode to this path")
    args = p.parse_args()
    if args.watch and args.record:
        p.error("--watch cannot be combined with --record")
    if args.watch and args.gif:
        p.error("--watch cannot be combined with --gif")
    return args


def save_gif(frames, path):
    # Keep every 2nd frame and halve the size to keep the file small
    frames = [f[::2, ::2] for f in frames[::2]]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / GIF_FPS, loop=0)  # loop=0 loops forever
    mb = path.stat().st_size / 1e6
    h, w = frames[0].shape[:2]
    print(f"Saved GIF: {path} ({mb:.2f} MB, {len(frames)} frames, {w}x{h})")
    if mb > GIF_MAX_MB:
        print(f"WARNING: GIF is over {GIF_MAX_MB} MB")


def main():
    args = parse_args()
    model = PPO.load(args.run / "model.zip")

    # Always the plain env with NO reward wrapper, so a custom bonus can't inflate the score
    if args.watch:
        render_mode = "human"
    elif args.record or args.gif:
        render_mode = "rgb_array"
    else:
        render_mode = None
    env = gym.make(ENV_ID, render_mode=render_mode)
    if args.record:
        video_dir = args.run / "video"
        env = RecordVideo(env, str(video_dir), episode_trigger=lambda i: i < RECORD_EPISODES, name_prefix="eval")

    gif_frames = []  # frames of episode 0, only if --gif

    episodes = []
    for i in range(args.episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        score = 0.0
        length = 0
        steps_since_contact = math.inf  # frames since any leg last touched
        done = False
        if args.gif and i == 0:
            gif_frames.append(env.render())
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            length += 1
            steps_since_contact = 0 if max(obs[LEFT_LEG], obs[RIGHT_LEG]) >= 0.5 else steps_since_contact + 1
            done = terminated or truncated
            if args.gif and i == 0:
                gif_frames.append(env.render())
        outcome = classify(terminated, truncated, reward, obs, steps_since_contact)  # reward and obs are the last step's
        episodes.append({"score": score, "length": length, "outcome": outcome})
    env.close()
    if args.gif:
        save_gif(gif_frames, args.gif)

    summary = summarize(episodes)
    result = {
        **summary,
        "settings": {
            "episodes": args.episodes,
            "deterministic": not args.stochastic,
            "seed_base": EVAL_SEED_BASE,
            "env": ENV_ID,
        },
        "episodes": episodes,
    }
    (args.run / "eval.json").write_text(json.dumps(result, indent=2))

    print(f"Evaluated {summary['n_episodes']} episodes ({'stochastic' if args.stochastic else 'deterministic'})")
    print(f"  score: mean {summary['mean']:.1f}  std {summary['std']:.1f}  min {summary['min']:.1f}  max {summary['max']:.1f}")
    print(f"  landed {summary['landed_rate']:.0%}  crashed {summary['crash_rate']:.0%}  "
          f"hover {summary['hover_rate']:.0%}  grounded_timeout {summary['grounded_timeout_rate']:.0%}  "
          f"other {summary['other_rate']:.0%}")
    print(f"  farming {summary['farming_rate']:.0%} (hover + grounded_timeout)  "
          f"at time limit {summary['at_time_limit_rate']:.0%}")
    print(f"  mean episode length: {summary['mean_ep_len']:.0f}")
    print(f"Saved: {args.run / 'eval.json'}")
    if args.record:
        print(f"Videos: {video_dir}")
    print(f"{'PASS' if summary['mean'] >= PASS_SCORE else 'FAIL'}: mean >= {PASS_SCORE}")


if __name__ == "__main__":
    main()
