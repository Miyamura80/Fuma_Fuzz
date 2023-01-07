from evosax import NetworkMapper
import gymnax
import distrax
import jax
import jax.numpy as jnp
from models import CategoricalSeparateMLP, GaussianSeparateMLP
from environment import make_environment

def get_model(rng, config, speed=False):
    """Instantiate a model according to obs shape of environment."""
    # Get number of desired output units
    env, env_params = make_environment(config.env_name, **config.env_kwargs)

    # Instantiate model class (flax-based)
    if config.train_type == "ES":
        model = NetworkMapper[config.network_name](
            **config.network_config, num_output_units=env.num_actions
        )
    elif config.train_type == "PPO":
        if config.network_name == "Categorical-MLP":
            model = CategoricalSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )
        elif config.network_name == "Gaussian-MLP":
            model = GaussianSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )

    # Only use feedforward MLP in speed evaluations!
    if speed and config.network_name == "LSTM":
        model = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            hidden_activation="relu",
            output_activation="categorical"
            if config.env_name != "PointRobot-misc"
            else "identity",
            num_output_units=env.num_actions,
        )

    # Initialize the network based on the observation shape
    obs_shape = env.observation_space(env_params).shape
    if config.network_name != "LSTM" or speed:
        params = model.init(rng, jnp.zeros(obs_shape), rng=rng)
    else:
        params = model.init(
            rng, jnp.zeros(obs_shape), model.initialize_carry(), rng=rng
        )
    return model, params