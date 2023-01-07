import configparser
import argparse
import os.path as osp
from utils import get_bytecode, get_model, get_config
from environment import EVM

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
parser.add_argument("-m", "--model", help="The RL architecture we will use.", default="PPO")
parser.add_argument("-e", "--environment", help="The RL training environment the agent uses.", default="EVM")

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


bytecode = get_bytecode(args, root_dir)

# for k,v in bytecode.items():
#     print(k)
# env = EVM(args)

config = get_config(args, root_dir)
print(config)


# model = get_model(
#     args,
# )


# run_sc_model_gc(
#     model,
#     train_graphs,
#     valid_graphs,
#     lr=args.lr,
#     batch_size=args.batch_size,
#     epochs=args.epochs,
# )


