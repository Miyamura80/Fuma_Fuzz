import os.path as osp
import json
import solcx
import glob
from typing import Dict, Any, List

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
            solcx.install_solc(args.version)
            compiled_json = solcx.compile_files(
                [source_code_path],
                output_values=["abi", "bin-runtime"],
                solc_version=args.version,
                optimize=args.optimize
            )
            for key, value in compiled_json.items():
                with open(precompiled_solidity_path, "w") as f:
                    f.write(json.dumps(value))
        else:
            with open(precompiled_solidity_path, "r") as f:
                compiled_json = json.load(f)

        # compiled_jsons = []
        # for _key, value in compiled_json.items():
        #     compiled_jsons.append(value)

        return compiled_json
    else:
        raise ValueError("Please provide a .sol type for source")

# e.g. if args.source = "data/{IDENTIFIER}/" 
# It will compile the jsons to "data/compiled_solidity/{IDENTIFIER}"
def compile_dir(args, root_dir) -> List[Dict[str, Any]]:
    source_folders = args.source.split("/")
    source_folders = source_folders if source_folders[-1]!="" else source_folders[:-1]
    source_folder_prefix = source_folders[-1]
    source_dir_path = (osp.join(root_dir, *source_folders))


    source_files = glob.glob(source_dir_path+"/*.sol")

    compiled_jsons = []
    
    for source_file_path in source_files:

        source_prefix, extention = osp.splitext(osp.basename(source_file_path))
        precompiled_solidity_path = osp.join(root_dir, "data", "compiled_solidity", f"{source_folder_prefix}",f"{source_prefix}.json")

        if not osp.exists(precompiled_solidity_path) or args.recompile:
            try:
                compiled_json = solcx.compile_files(
                    [source_file_path], # BACKLOG: edit by doing it by feeding all files here
                    output_values=["abi", "bin-runtime"],
                    solc_version=args.version,
                    optimize=args.optimize
                )
            except Exception as e:
                if e.return_code==0:
                    continue
                elif e.return_code==1:
                    with open(source_file_path, "r") as f:
                        lines = f.readlines()
                    j = next(i for i,v in enumerate(lines) if v.startswith("pragma"))

                    lines[j] = f"pragma solidity {args.version};\n"
                    with open(source_file_path, "w") as f:
                        f.writelines(lines)
                    compiled_json = solcx.compile_files(
                        [source_file_path], # BACKLOG: edit by doing it by feeding all files here
                        output_values=["abi", "bin-runtime"],
                        solc_version=args.version,
                        optimize=args.optimize
                    )
                
            with open(precompiled_solidity_path, "w") as f:
                f.write(json.dumps(compiled_json))
        else:
            with open(precompiled_solidity_path, "r") as f:
                compiled_json = json.load(f)
        compiled_jsons.append(compiled_json)
    return compiled_jsons
