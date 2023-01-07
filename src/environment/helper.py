from environment.breakout_miniatari import MinBreakout
from environment.py_evm import PyEVM

def make_environment(env_id: str, **env_kwargs):

    if env_id == "Breakout-MinAtar":
        env = MinBreakout(**env_kwargs)
    elif env_id == "PyEVM":
        env = PyEVM(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    # Create a jax PRNG key for random seed control
    return env, env.default_params

    
