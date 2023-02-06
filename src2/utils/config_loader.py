import re
import yaml
from dotmap import DotMap
from typing import Dict, Any
import os.path as osp

def load_yaml(config_fname: str) -> dict:
    """Load in YAML config file."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(config_fname) as file:
        yaml_config = yaml.load(file, Loader=loader)
    return yaml_config

def get_config(root_dir, config_fname, seed_id=0, lrate=None):
    """Load training configuration and random seed of experiment."""

    

def get_config(args, root_dir) -> Dict[str, Any]:
    config_folders = args.config.split("/")
    # If yaml file provided as config
    if args.config.endswith(".yaml"):
        config_file_path = (osp.join(root_dir, "config", args.config) if len(config_folders) == 1
            else osp.join(root_dir, config_folders))
        
        lrate = args.lr if args.lr != None else None
        seed_id = args.seed if args.seed != None else 0

        config = load_yaml(config_file_path)
        config["train_config"]["seed_id"] = seed_id
        if lrate is not None:
            if "lr_begin" in config["train_config"].keys():
                config["train_config"]["lr_begin"] = lrate
                config["train_config"]["lr_end"] = lrate
            else:
                try:
                    config["train_config"]["opt_params"]["lrate_init"] = lrate
                except Exception:
                    pass
        return DotMap(config)
    else:
        raise ValueError("Please provide a .yaml type for --config")
