import os.path as osp
import json
import solcx
from typing import Dict, Any

def get_bytecode(args, root_dir) -> Dict[str, Any]:
    source_folders = args.source.split("/")
    # If solidity file provided as source
    if args.source.endswith(".sol"):
        source_code_path = (osp.join(root_dir, "data", "solidity", args.source) if len(source_folders) == 1
            else osp.join(root_dir, source_folders))
        # Get the prefix name of the file: e.g. for fallback.sol -> fallback
        source_prefix = source_folders[-1].split(".")[0]
        precompiled_solidity_path = osp.join(root_dir, "data", "compiled_solidity", f"{source_prefix}.json")

        if not osp.exists(precompiled_solidity_path) or args.recompile:
            compiled_json = solcx.compile_files(
                [source_code_path],
                output_values=["abi", "bin-runtime"],
                solc_version=args.version,
                optimize=args.optimize
            )
            compiled_json = compiled_json[list(compiled_json.keys())[0]]
            with open(precompiled_solidity_path, "w") as f:
                f.write(json.dumps(compiled_json))
        else:
            with open(precompiled_solidity_path, "r") as f:
                compiled_json = json.load(f)
                compiled_json = compiled_json[list(compiled_json.keys())[0]]

        return compiled_json
    else:
        raise ValueError("Please provide a .sol type for source")