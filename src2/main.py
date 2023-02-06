import configparser
import argparse
import os.path as osp
from utils import get_bytecode, get_model, get_config, save_pkl_object, load_model_from_pkl
from training import train_ppo
import jax
from jax.config import config

config.update("jax_enable_x64", True)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Default Configuration
config = configparser.ConfigParser()
config.read("config.ini")



# CLI configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--source", help="Name of EVM code to execute located inside of \`./data\`. If provided .sol file, will compile via solc first.", required=True
)

# Solc Compiler Arguments
parser.add_argument(
    "-r",
    "--recompile",
    help="Recompile if source is solidity file",
    type=str2bool,
    default=True,
)
parser.add_argument(
    "-o",
    "--optimize",
    help="Carry out bytecode optimization on compilation",
    type=str2bool,
    default=True,
)
parser.add_argument("-v", "--version", help="The solc version we will use.", default="0.7.0")

# RL arguments
parser.add_argument(
    "-c", "--config", help="Name of config file (.yaml) to use for RL model located inside of \`./config\`. ", required=True
)
parser.add_argument("--seed", help="The NN random seed", type=int)
parser.add_argument("-m", "--model", help="The model pickle to load", type=str)

# Training arguments
parser.add_argument("--lr", help="Learning rate.", default=0.001, type=float)
parser.add_argument("--epochs", help="Number of epochs.", default=500, type=int)
parser.add_argument("-b", "--batch_size", help="Training Batch size.", default=32, type=int)


# Model specific arguments
parser.add_argument(
    "--num_layers", help="Number of layers in the RL model.", default=6, type=int
)



args = parser.parse_args()


root_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), ".."))


# compiled_json = get_bytecode(args, root_dir)

# for k,v in bytecode.items():
#     print(k)
# env = EVM(args)

config = get_config(args, root_dir)
config = config.train_config


rng = jax.random.PRNGKey(0)
rng, rng_init = jax.random.split(rng)
# Setup the model architecture
model, params = get_model(
    rng,
    config
)
# model, params = load_model_from_pkl(config, f"agents/experiment/PPO_001.pkl")

from environment import make_environment
rng = jax.random.PRNGKey(0)
rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
env, env_params = make_environment(config.env_name, **config.env_kwargs)

obs, state = env.reset(key_reset, env_params)
action = env.action_space(env_params).sample(key_policy)
n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
print("done with exp")

mle_log = None
# Log and store the results.
log_steps, log_return, network_ckpt = train_ppo(
    rng, config, model, params, mle_log
)

# data_to_store = {
#     "log_steps": log_steps,
#     "log_return": log_return,
#     "network": network_ckpt,
#     "train_config": config,
# }


# log_ext = "002"

# save_pkl_object(
#     data_to_store,
#     f"agents/experiment/PPO_{log_ext}.pkl",
# )
