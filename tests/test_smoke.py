import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from rocket_land.config import ENV_ID, EVAL_SEED_BASE, PPO_KWARGS
from rocket_land.rewards import REWARDS
from rocket_land.train import ShapedReturnCallback, make_env, wrapper_chain


@pytest.mark.slow
@pytest.mark.parametrize("reward", list(REWARDS))
def test_train_save_load_evaluate(reward, tmp_path):
    # Tiny PPO run to check the whole pipeline works for this reward
    vec_env = make_env(reward, seed=0, n_envs=2, monitor_dir=str(tmp_path / "monitor"))
    kwargs = {**PPO_KWARGS, "n_steps": 64, "batch_size": 64}
    model = PPO("MlpPolicy", vec_env, seed=0, **kwargs)
    callback = None if reward == "builtin" else ShapedReturnCallback()
    model.learn(total_timesteps=256, callback=callback)
    model.save(tmp_path / "model.zip")
    vec_env.close()

    # Reload and run one episode on the plain env (no reward wrapper)
    model = PPO.load(tmp_path / "model.zip")
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=EVAL_SEED_BASE)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


@pytest.mark.parametrize("reward", list(REWARDS))
def test_wrapper_order_reward_wrapper_outside_monitor(reward, tmp_path):
    # Monitor must sit directly around the env, with the reward wrapper outside it
    vec_env = make_env(reward, seed=0, n_envs=1, monitor_dir=str(tmp_path / "monitor"))
    try:
        chain = wrapper_chain(vec_env.envs[0])
    finally:
        vec_env.close()
    wrapper_class = REWARDS[reward]
    expected = "Monitor" if wrapper_class is None else f"{wrapper_class.__name__} -> Monitor"
    assert chain.startswith(expected), chain
