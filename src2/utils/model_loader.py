from environment import make_environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


def get_model(config):
    """Instantiate a model according to obs shape of environment."""
    # The algorithms require a vectorized environment to run
    env, env_params = make_environment(config.env_name, **config.env_kwargs)
    check_env(env)

    if config.train_type == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10_000)

    return model

