from .evm import EVM


def get_runtime_calldata_from_abi(abi, func_name: str, inputs: dict):
    func_interface = next(filter(lambda e: e.get("name", None) == func_name and e.get("type", None) == "function", abi))
    func_names = [x["name"] for x in func_interface["inputs"]]
    func_vals = [inputs[name] for name in func_names]
    return encodeABI(abi, func_name, func_vals)


class Py_EVM(EVM):
    def __init__(self, args) -> None:
        super().__init__(args)

    def run_bytecode(self, args, compilation_result, abi_args) -> None:
        abi_desc, bytecode = compilation_result["abi"], compilation_result["bin-runtime"]
        instrs = Program.from_binary(bytecode)
        func_name, inputs = abi_args
        print(instrs)
        print(abi_desc)
        print(func_name, inputs)

        calldata = get_runtime_calldata_from_abi(abi_desc, func_name, inputs)
        print(calldata)

    def run_instr(self, instr: int, *state):
        print(instr, INSTR_INFO[instr])
        match instr:
            case _:
                pass
