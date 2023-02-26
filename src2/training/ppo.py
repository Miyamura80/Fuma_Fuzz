from environment import make_environment

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import tqdm

def train_ppo(rng, config, model, params, mle_log, neptune_client):

    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    
    # The algorithms require a vectorized environment to run
    env, env_params = make_environment(config.env_name, **config.env_kwargs)
    env = DummyVecEnv([env])
    obs = env.reset()

    total_steps = 0
    log_steps, log_return = [], []
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)
    for step in t:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        total_steps += 1 
        # env.render()
    print("Training finished!")