from .evm import EVM
from typing import Tuple
from eth_utils import (
    decode_hex,
)

def get_runtime_calldata_from_abi(abi, func_name: str, inputs: dict):
    func_interface = next(filter(lambda e: e.get("name", None) == func_name and e.get("type", None) == "function", abi))
    func_names = [x["name"] for x in func_interface["inputs"]]
    func_vals = [inputs[name] for name in func_names]
    return abi, func_name, func_vals

ACC_START = 45_000

class Py_EVM(EVM):
    
    def __init__(self, args, bytecode, contract_abi, account_num=2) -> None:
        super().__init__(args)
        self.bytecode_str = bytecode
        self.contract_abi = contract_abi
        self.bytecode = decode_hex(bytecode)

        # Create accounts 
        assert account_num >= 2 # Require at least a deployer and an attacker
        self.accounts = [ACC_START+i for i in range(account_num)]
        self.deployer_acc = self.accounts[1]
        self.attacker_acc = self.accounts[0]
        
        # Send money to these initialized accounts
        self.balances = {addr: 100000 for i,addr in enumerate(self.accounts)}

        self.reset()

    # TODO: implement this 
    def reset(self):
        pass

    # TODO: Implement this
    def run_bytecode(self, f_name, f_args, f_value:int=0, sender_id:int=0) -> Tuple[int, int]:
        # Returns (result, type) type==0: int, type==1: address
        # By Default, it executes under the attacker
        sender_account = self.accounts[sender_id] if sender_id!=0 else self.attacker_acc

        return (0, 0)


