# from utils import get_model

def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    import pickle
    from pathlib import Path

    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")

def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj

# def load_model_from_pkl(config, agent_path):
#     rng = jax.random.PRNGKey(0)
#     model, _ = get_model(rng, config)

#     params = load_pkl_object(agent_path)["network"]
#     return model, params