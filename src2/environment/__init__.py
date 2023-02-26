#!/usr/bin/env python3
from environment.pyevm_env import PyEVM_Env

def make_environment(env_id: str, **env_kwargs):

    if env_id == "PyEVM":
        env = PyEVM_Env(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params

    
