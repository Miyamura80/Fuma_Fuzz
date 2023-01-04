import os.path as osp

def get_bytecode(args, root_dir):
    source_folders = args.source.split("/")
    # If solidity file provided as source
    if args.source.endswith(".sol"):
        source_code_path = (osp.join(root_dir, "data", "solidity", args.source) if len(source_folders) == 1
            else osp.join(root_dir, source_folders))
        precompiled_solidity_root = osp.join(source_code_path, "compiled_solidity")
        # TODO: Compile with solc
        bytecode = ""
        pass
    else:
        source_code_path = (osp.join(root_dir, "data", "bytecode", args.source) if len(source_folders) == 1 
            else osp.join(root_dir, source_folders))

        with open(source_code_path, "rb") as f:
            bytecode = f.read()
    return bytecode
