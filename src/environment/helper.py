from environment.breakout_miniatari import MinBreakout
from environment.py_evm_env import PyEVM_Env
from environment.mountaincar_cont import ContinuousMountainCar

def make_environment(env_id: str, **env_kwargs):

    if env_id == "Breakout-MinAtar":
        env = MinBreakout(**env_kwargs)
    elif env_id == "PyEVM":
        env = PyEVM_Env(**env_kwargs)
    elif env_id == "MountainCarContinuous-v0":
        env = ContinuousMountainCar(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    # Create a jax PRNG key for random seed control
    return env, env.default_params

    
