#!/usr/bin/env python3
from .evm import EVM
import sys
from Crypto.Hash import keccak
from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3, EthereumTesterProvider

# Append path of py-evm repo for imports
from eth.db.atomic import AtomicDB
from eth.exceptions import VMError
from eth.vm.computation import BaseComputation
from eth_keys import KeyAPI
from eth_keys.datatypes import PrivateKey
from eth_utils import (
    to_wei,
    to_canonical_address, 
    decode_hex,
    is_checksum_address
)

from eth import constants, Chain
from eth.vm.forks.byzantium import ByzantiumVM
from eth.vm.forks.constantinople import ConstantinopleVM
from eth.vm.forks.arrow_glacier import ArrowGlacierVM
from eth_account import Account

sys.path.append('../../py-evm/')



ADDRESS = bytes.fromhex("123456789A123456789A123456789A123456789A")
GAS_PRICE = 1
VERBOSE = False
Web3.DEBUG = True


def base_db() -> AtomicDB:
    return AtomicDB()


lazy_key_api = KeyAPI(backend=None)


def funded_address_private_key() -> PrivateKey:
    return lazy_key_api.PrivateKey(
        decode_hex('0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8')
    )


def funded_address() -> bytes:
    return funded_address_private_key().public_key.to_canonical_address()


def funded_address_initial_balance():
    return to_wei(1000, 'ether')


class ChainHelper(object):

    def chain_with_block_validation(self, base_db, funded_address, funded_address_initial_balance):
        """
        Return a Chain object containing just the genesis block.
        The Chain's state includes one funded account, which can be found in the
        funded_address in the chain itself.
        This Chain will perform all validations when importing new blocks, so only
        valid and finalized blocks can be used with it. If you want to test
        importing arbitrarily constructe, not finalized blocks, use the
        chain_without_block_validation fixture instead.
        """
        genesis_params = {
            "coinbase": to_canonical_address("8888f1f195afa192cfee860698584c030f4c9db1"),
            "difficulty": 131072,
            "extra_data": b"B",
            "gas_limit": 3141592,
            "nonce": decode_hex("0102030405060708"),
            "receipt_root": decode_hex("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"),  # noqa: E501
            "timestamp": 1422494849,
            "transaction_root": decode_hex("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"),  # noqa: E501
        }
        genesis_state = {
            funded_address: {
                "balance": funded_address_initial_balance,
                "nonce": 0,
                "code": b"",
                "storage": {},
                "constantinopleBlock": 0,
            },
            
        }
        klass = Chain.configure(
            __name__='TestChain',
            vm_configuration=(
                (constants.GENESIS_BLOCK_NUMBER, ArrowGlacierVM),
            ),
        )
        chain = klass.from_genesis(base_db, genesis_params, genesis_state)
        return chain

    def import_block_without_validation(chain, block):
        return super(type(chain), chain).import_block(block, perform_validation=False)

    def chain_without_block_validation(
            self,
            request,
            base_db,
            funded_address,
            funded_address_initial_balance):
        """
        Return a Chain object containing just the genesis block.
        This Chain does not perform any validation when importing new blocks.
        The Chain's state includes one funded account and a private key for it,
        which can be found in the funded_address and private_keys variables in the
        chain itself.
        """
        # Disable block validation so that we don't need to construct finalized blocks.
        overrides = {
            'import_block': self.import_block_without_validation,
            'validate_block': lambda self, block: None,
        }

        byzantiumVMForTesting = ArrowGlacierVM.configure(validate_seal=lambda block: None)
        chain_class = request.param
        klass = chain_class.configure(
            __name__='TestChainWithoutBlockValidation',
            vm_configuration=(
                (constants.GENESIS_BLOCK_NUMBER, byzantiumVMForTesting),
            ),
            **overrides,
        )
        genesis_params = {
            'block_number': constants.GENESIS_BLOCK_NUMBER,
            'difficulty': constants.GENESIS_DIFFICULTY,
            'gas_limit': constants.GENESIS_GAS_LIMIT,
            'parent_hash': constants.GENESIS_PARENT_HASH,
            'coinbase': constants.GENESIS_COINBASE,
            'nonce': constants.GENESIS_NONCE,
            'mix_hash': constants.GENESIS_MIX_HASH,
            'extra_data': constants.GENESIS_EXTRA_DATA,
            'timestamp': 1501851927,
        }
        genesis_state = {
            funded_address: {
                'balance': funded_address_initial_balance,
                'nonce': 0,
                'code': b'',
                'storage': {},
            }
        }
        chain = klass.from_genesis(base_db, genesis_params, genesis_state)
        return chain


class VMExecutionError(Exception):

    def __init__(self, msg: str, vm_error: VMError):
        self.msg = msg
        self.vm_error: vm_error


class Simulator(object):

    def __init__(self):
        self.chain = ChainHelper().chain_with_block_validation(base_db(),
                                                               funded_address(),
                                                               funded_address_initial_balance())
        self.vm = self.chain.get_vm()

    def executeCode(self, gas, data, code, value:int=0, sender=ADDRESS) -> BaseComputation:
        """
        Executes the given bytecode sequence
        :param gas:
        :param data:
        :param code:
        :return:
        """
        origin = None
        baseComputation = self.vm.execute_bytecode(origin, GAS_PRICE, gas, sender, sender, value, data, code)
        return baseComputation


def main(inputCode) -> None:

    # create simulator for VM
    sim = Simulator()
    # convert cli string input to tbytes input
    inputAsBytes = decode_hex(inputCode)

    if VERBOSE:
        print("Code: " + inputCode)

    # execute raw bytecode
    computation = sim.executeCode(1000000000000, b'', inputAsBytes)

    # check, if error during execution occured
    if computation.is_error:
        origin = computation._error
        exc = VMExecutionError(str(origin), origin)
        raise exc

    if VERBOSE:
        print("Gas used: " + str(computation.get_gas_used()))
        print("Remaining gas: " + str(computation.get_gas_remaining()))
        print("Value Returned: " + str(computation.output))
        
        print(computation.get_log_entries())
        print("Stack: " + str(computation._stack.values))



def get_runtime_calldata_from_abi(abi, func_name: str, inputs: dict):
    func_interface = next(filter(lambda e: e.get("name", None) == func_name and e.get("type", None) == "function", abi))
    func_names = [x["name"] for x in func_interface["inputs"]]
    func_vals = [inputs[name] for name in func_names]
    return abi, func_name, func_vals

def pretty_print_bytecode(hex_string):

    pretty_hex_string = '\n'.join([f"[{i/2}] {hex_string[i:i+2]}" for i in range(0, len(hex_string), 2)])
    return pretty_hex_string
    
class Py_EVM(EVM):
    
    def __init__(self, args, bytecode, contract_abi, account_num=2) -> None:
        super().__init__(args)
        self.bytecode_str = bytecode
        self.contract_abi = contract_abi
        self.bytecode = decode_hex(bytecode)

        # W3 stuff
        self.tester_provider = EthereumTesterProvider(EthereumTester(PyEVMBackend()))
        self.eth_tester = self.tester_provider.ethereum_tester
        self.w3 = Web3(self.tester_provider)

        # Create accounts 
        assert account_num >= 2 # Require at least a deployer and an attacker
        self.accounts = [Account.create(str(i)) for i in range(account_num)]
        self.deployer_acc = self.accounts[1]
        self.attacker_acc = self.accounts[0]
        self.accounts = {acc.address : acc for acc in self.accounts}
        
        # Send money to these initialized accounts
        for i,acc in enumerate(self.accounts):
            sender = self.eth_tester.get_accounts()[i]
            self.eth_tester.send_transaction({'from': sender,
                                              'to': acc,
                                              'gas': 30000,
                                              'value': self.eth_tester.get_balance(sender) // 2})

        self.reset(bytecode, contract_abi)


    def reset(self, bytecode, abi):
        self.contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        # Run constructor from deployer
        # self.run_bytecode("constructor", [], 0, self.deployer_acc.address)
        # tx_hash = self.contract.constructor().transact({'from': self.deployer_acc.address, 'gas': 2000000})
        # tx = self.contract.constructor().transact()

        # tx = self.contract.constructor().buildTransaction({'from': self.deployer_acc.address, 
        #                                                    'gas': 2000000, 
        #                                                    'nonce': self.w3.eth.getTransactionCount(self.deployer_acc.address),})
        # signed_tx = self.deployer_acc.sign_transaction(tx)
        # tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        # tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
        

        tx_hash = self.contract.constructor().transact(transaction={'from': self.eth_tester.get_accounts()[4], 'gas': 3100000})
        tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
        
        self.contract_address = tx_receipt.contractAddress
        self.contract_object = self.w3.eth.contract(abi=abi, address=self.contract_address, bytecode=bytecode)
        print("end")

    # TODO: Fix the caller of run_bytecode in env
    def run_bytecode(self, f_name, f_args, f_value:int=0, sender_addr=None):
        # By Default, it executes under the attacker
        sender_account = self.accounts[sender_addr] if sender_addr!=None else self.attacker_acc

        tx = getattr(self.contract_object.functions, f_name)(*f_args) \
                .buildTransaction({
                'from': sender_account.address,
                'value': f_value,
                'gas': 1500000,
                'nonce': self.w3.eth.getTransactionCount(sender_account.address),
            })
        # TODO: Check if the data produced here is the same as the one hard coded
        signed_tx = sender_account.sign_transaction(tx)
        tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
        r = self.contract_object.functions.contributions(*f_args).transact()
        r = self.contract_object.functions.contributions(*f_args).transact({
                'value': f_value,
                'gas': 1500000,
            })
        return r.hex()




# class Py_EVM(EVM):
    
#     def __init__(self, args, bytecode, deployer="0x38644552091a69125d5dfcb7b8c2659029395bdf") -> None:
#         super().__init__(args)
#         self.bytecode_str = bytecode
#         self.bytecode = decode_hex(bytecode)

#         deployer_raw = deployer[2:] if deployer.startswith("0x") else deployer
#         self.deployer = to_canonical_address(deployer)
        
#         self.reset()


#     def reset(self):
#         constructor_hash = decode_hex("90fa17bb") # h("constructor()")
#         self.sim = Simulator()
#         self.sim.executeCode(10000000000, constructor_hash, self.bytecode, sender=self.deployer)


#     def run_bytecode(self, f_args, f_value:int=0) -> None:
#         # convert cli string input to tbytes input
#         args_as_bytes = decode_hex(f_args)

#         # execute raw bytecode
#         computation = self.sim.executeCode(10000000000, args_as_bytes, self.bytecode, f_value)

#         # check, if error during execution occured
#         if computation.is_error:
#             origin = computation._error
#             print(computation._stack.values)
#             exc = VMExecutionError(str(origin), origin)
#             raise exc

#         # TODO: Get the value read from view functions
        

#         if VERBOSE:
#             print("Gas used: " + str(computation.get_gas_used()))
#             print("Remaining gas: " + str(computation.get_gas_remaining()))
#             print("Value Returned: " + str(computation.output))
#             print(computation.get_log_entries())
#             print("Stack: " + str(computation._stack.values))
        
#         return computation.output


if __name__=="__main__":
    bytecode = "60806040526004361061004e5760003560e01c80633ccfd60b1461009357806342e94c90146100aa5780638da5cb5b146100ef578063d7bb99ba14610120578063f10fdf5c146101285761008e565b3661008e5760003411801561007157503360009081526020819052604090205415155b61007a57600080fd5b600180546001600160a01b03191633179055005b600080fd5b34801561009f57600080fd5b506100a861013d565b005b3480156100b657600080fd5b506100dd600480360360208110156100cd57600080fd5b50356001600160a01b03166101d8565b60408051918252519081900360200190f35b3480156100fb57600080fd5b506101046101ea565b604080516001600160a01b039092168252519081900360200190f35b6100a86101f9565b34801561013457600080fd5b506100dd610255565b6001546001600160a01b0316331461019c576040805162461bcd60e51b815260206004820152601760248201527f63616c6c6572206973206e6f7420746865206f776e6572000000000000000000604482015290519081900360640190fd5b6001546040516001600160a01b03909116904780156108fc02916000818181858888f193505050501580156101d5573d6000803e3d6000fd5b50565b60006020819052908152604090205481565b6001546001600160a01b031681565b66038d7ea4c68000341061020c57600080fd5b3360008181526020819052604080822080543401908190556001546001600160a01b031683529082205492909152111561025357600180546001600160a01b031916331790555b565b336000908152602081905260409020549056fea2646970667358221220060eb3bedd321cc4d27e64cf9813e1061a92f752d32925d9c69e42364a3ccb2264736f6c63430007000033"
    args = {}
    evm = Py_EVM(args, bytecode)
    print("huhis")
