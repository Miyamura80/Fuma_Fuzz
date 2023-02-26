import gym
from gym import spaces
import numpy as np
from dotmap import DotMap
from evm import Py_EVM, EVM
from utils import get_bytecode
import os.path as osp
from typing import Tuple, Optional
from Crypto.Hash import keccak
from typing import List


# State - Function description settings (ERROR if contract exceeds this)
MAX_FUNC_VIEWABLE: int = 10 
MAX_FUNC_NONPAYABLE: int = 10
MAX_FUNC_PAYABLE: int = 5
MAX_FUNC_TOTAL: int = MAX_FUNC_VIEWABLE + MAX_FUNC_NONPAYABLE + MAX_FUNC_PAYABLE

MAX_ARGUMENT_COUNT: int = 3

# Action - Function argument settings
ADDRESSES = ["0x69420552091A69125d5DfCb7b8C2659029395Bdf", "0x38644552091A69125d5DfCb7b8C2659029395Bdf"]
ADDRESS_SET_SIZE: int = len(ADDRESSES) # Default |A|=2, A[0] is attacker, A[1] is deployer, A[2],... is victims

params = {
    # Attack environment setup
    "contract_initial_balance": 10,
    "attacker_initial_balance": 8, # 0.1 ETH, unit in gwei
    "max_attack_time": 10, # Number of function invocations done before finishing

    # Action - Function argument settings
    "uint256_arg_min": 0, # 1 gwei
    "uint256_arg_max": 10 # 1 ETH
}
params["arg_int_size"] = params["uint256_arg_max"] - params["uint256_arg_min"] + 1
params["a_set_size"] = params["arg_int_size"] + ADDRESS_SET_SIZE
params = DotMap(params)


class PyEVM_Env(gym.Env):
    """
    JAX Compatible version of EVM environment for RL agent

    ================================================================
    ENVIRONMENT DESCRIPTION - 'PyEVM'
    ================================================================
    STATE:
    - Contract ABI information (order the payable & nonpayable attributes first)
    - Attacker Balance
    - Contract Balance
    - Victim(s) Balances
    - Time
    - Dynamic State
        - Return results of all `view` functions (not pure, since it doesn't read state)
    
    >>> (
            f_p1, ..., f_p|f_p|, f_n1, ..., f_n|f_n|, f_v1, ..., f_n|f_v|,
            attacker_balance (A[0]),
            contract_balance (A[1]),
            A[2], A[3], A[4], ... A[|A|-1],
            time,
            o(f_v1), o(f_v2), ... , o(f_n|f_v|)
        )
    ----------------------------------------------------------------
    ACTION:
    - f_payable <ARGUMENTS> <VALUE>
    - f_nonpayable <ARGUMENTS>

    >>> (function_id, payment_gwei, c_1, c_2, c_3)

    Where c_i == -1 means argument doesn't exist
    Where c_i from [0,10] -> maps to uint256, c_i from [11,12] -> maps to addresses
    ----------------------------------------------------------------
    REWARD:
    - A reward of +v is given if agent balance is increased by v gwei
    - A reward of +epsilon for state change to view_function_output, with decay over time
    - A reward of minus number of actions taken weighted by gas usage
    
    ----------------------------------------------------------------
    TERMINATION:
    - Termination if contract is entirely drained or max_attack_time reached

    """

    def __init__(self, contract_names):
        super(PyEVM_Env, self).__init__()
        contract_name = contract_names[0]

        args = {"source": contract_name, 
                "recompile": False,
                "version": "0.7.0",
                "optimize": True,
        }
        args = DotMap(args)

        root_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "..", ".."))
        compiled_json = get_bytecode(args, root_dir)

        self.contract_abi = compiled_json["abi"]
        # self.bytecode = "604260005260206000F3"
        self.bytecode = compiled_json["bin-runtime"]

        self.action_func_num = len(list(filter(lambda x: x["stateMutability"]=="payable" or x["stateMutability"]=="nonpayable", self.contract_abi)))
        
        # Utils
        self.action_function_lookup, self.dynamic_obs_func_lookup  = create_function_lookup(self.contract_abi)
        
        # Setting up the observation attributes
        self.default_params = params
        temp_params = self.default_params
        arg_int_size = temp_params.uint256_arg_max - temp_params.uint256_arg_min + 1
        
        # EVM 
        self.evm = Py_EVM(args, self.bytecode)

        self.arg_int_size = arg_int_size
        self.obs_shape = (
            5, 
            MAX_FUNC_NONPAYABLE+MAX_FUNC_PAYABLE+MAX_FUNC_VIEWABLE, 
            max(MAX_ARGUMENT_COUNT, ADDRESS_SET_SIZE)
        )

        self.reset()

        """
        Action Space
        """
        argument_index = 10+ADDRESS_SET_SIZE
        low = np.array(
            [0, 0, 0, 0, 0], dtype=np.int64
        )
        high = np.array(
            [self.action_func_num-1, params.attacker_initial_balance, 
                argument_index, argument_index, argument_index], dtype=np.int64
        )
        self.action_space = spaces.Box(low, high, (5,), np.int64)

        """
        Observation space
        """
        self.observation_space = spaces.Box(0, 10**(params.contract_initial_balance+1), self.obs_shape, np.float64)

    def step(self, action):
        """Perform single timestep state transition."""
        compilation_result = {"abi": self.contract_abi, "bin-runtime": self.bytecode}
        self.state, reward = step_evm(self.state, compilation_result, action, self.evm, self.action_function_lookup)

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(self.state)
        info = {}
        return (
            self.get_obs(self.state),
            reward,
            done,
            info,
        )

    def reset(self):
        self.evm.reset()
        abi_function_onehot, abi_argument_onehot = contract_abi_json_to_array(self.contract_abi)
        dynamic_observation = reset_dynamic_observation(self.contract_abi, self.evm)

        state = DotMap()
        state.abi_function_onehot=abi_function_onehot
        state.abi_argument_onehot=abi_argument_onehot
        state.attacker_balance=10**params.attacker_initial_balance
        state.contract_balance=10**params.contract_initial_balance
        state.view_function_output=dynamic_observation
        state.time=0
        self.state = state
        return self.get_obs(state)

    """
    >>> (
            f_p1, ..., f_p|f_p|, f_n1, ..., f_n|f_n|, f_v1, ..., f_n|f_v|,
            attacker_balance,
            contract_balance,
            o(f_v1), o(f_v2), ... , o(f_n|f_v|)
        )

        Dim 0: Info of function presence in ABI
            - e.g.  I(f_p1), ..., I(f_p|f_p|), 
                    I(f_n1), ..., I(f_n|f_n|), 
                    I(f_v1), ..., I(f_n|f_v|)
            - Shape: (|f_p| + |f_n| + |f_v|, 1)
        Dim 1: Info of argument types (address)
            - e.g.  I(c^1_p1==addr), ..., I(c^1_p|f_p|==addr),
                    I(c^1_n1==addr), ..., I(c^1_n|f_n|==addr),
                    I(c^1_v1==addr), ..., I(c^1_v|f_v|==addr),
            - Shape: (|f_p| + |f_n| + |f_v|, m)
        Dim 2: Info of argument types (uint256)
            - e.g.  I(c^1_p1==uint256), ..., I(c^1_p|f_p|==uint256)
                    I(c^1_n1==uint256), ..., I(c^1_n|f_n|==uint256)
                    I(c^1_v1==uint256), ..., I(c^1_v|f_v|==uint256)
            - Shape: (|f_p| + |f_n| + |f_v|, m)
        Dim 3: Balances
            - e.g.  A[0]
                    A[1]
            - Shape: (|A|, 1)
        Dim 4: Time
            - e.g.  t
            - Shape: (1,1)
        Dim 5: Dynamic observation (view function output)
            - e.g.  O(f_v1)
                    O(f_v2)
                    ...
                    O(f_v|f_v|)
            - Shape: (|f_v|, m*(|I|+|A|))
    """
    def get_obs(self, state: DotMap) -> np.ndarray:
        """Return observation from raw state trafo."""
        obs = np.zeros(self.obs_shape, dtype=np.int64)
        # Set the observation tensor according to above spec
        obs[0, :, 0] = state.abi_function_onehot
        obs[1:3, :, 0:MAX_ARGUMENT_COUNT] = state.abi_argument_onehot
        obs[3, 0:ADDRESS_SET_SIZE, 0] = np.array([state.attacker_balance, state.contract_balance])
        obs[4, :MAX_FUNC_VIEWABLE, :ADDRESS_SET_SIZE] = state.view_function_output

        return obs.astype(np.float64)
    
    def is_terminal(self, state: DotMap) -> bool:
        """Check whether state is terminal."""
        # Has contract been drained? 
        # BACKLOG: Improve this termination condition
        done1 = state.contract_balance == 0

        done_steps = state.time >= params.max_attack_time
        return (done_steps or done1)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...


# Takes the contract_abi and returns indexing from func_id -> func_name
def create_function_lookup(contract_abi: dict) -> Tuple[dict, dict]:
    f_v_set, f_action_set = [], []
    for func in contract_abi:
        if func["stateMutability"]=="view":
            f_v_set.append(func)
        elif func["stateMutability"]=="payable" or func["stateMutability"]=="nonpayable":
            f_action_set.append(func)
    
    f_v_set += [{}] * (MAX_FUNC_VIEWABLE - len(f_v_set))
    f_action_set += [{}] * (MAX_FUNC_PAYABLE + MAX_FUNC_NONPAYABLE - len(f_action_set))

    action_function_lookup = {}
    dynamic_obs_func_lookup = {}
    for i in range(MAX_FUNC_PAYABLE+MAX_FUNC_NONPAYABLE):
        if f_action_set[i] != {} and "name" in f_action_set[i]:
            action_function_lookup[i] = f_action_set[i]["name"]

    for i in range(MAX_FUNC_VIEWABLE):
        if f_v_set[i] != {} and "name" in f_v_set[i]:
            dynamic_obs_func_lookup[i] = f_v_set[i]["name"]

    return action_function_lookup, dynamic_obs_func_lookup


def step_evm(state: DotMap, compilation_result: dict, action: spaces.Box, evm: EVM, fname_lookup: dict) -> Tuple[DotMap, float]:
    
    # Unpack the values from the state
    func_id, payment_gwei, c_1, c_2, c_3 = action
    func_id = round(func_id)

    if func_id >= len(fname_lookup):
        return state, 0.0

    func_name = fname_lookup[func_id+1]
    # arg_val_lookup = {i: }

    inputs = {}
    args = {}
    contract_abi = compilation_result['abi']

    # Execute the action on the EVM
    # evm.run_bytecode(args, compilation_result, abi_args)

    new_attacker_balance = state.attacker_balance
    new_contract_balance = state.contract_balance
    evm_state_obs = []

    dynamic_observation = reset_dynamic_observation(contract_abi, evm)

    new_state = DotMap()
    new_state.abi_function_onehot=state.abi_function_onehot
    new_state.abi_argument_onehot=state.abi_argument_onehot
    new_state.attacker_balance=new_attacker_balance
    new_state.contract_balance=new_contract_balance
    new_state.view_function_output=dynamic_observation
    new_state.time=state.time + 1


    reward = 0.0

    return new_state, reward

# BACKLOG: With int inputs as well
def reset_dynamic_observation(contract_abi: dict, evm: EVM):
    viewable_abi = [func for i,func in enumerate(contract_abi) if func['stateMutability']=='view']
    f_hash_typemap = [(keccak_of_func_signature(func['name'],[inp["type"] for inp in func['inputs']]), [inp["type"] for inp in func['inputs']]) for func in viewable_abi]

    result = np.zeros((MAX_FUNC_VIEWABLE, ADDRESS_SET_SIZE), dtype=np.int64)
    for i, (f_hash, inps) in enumerate(f_hash_typemap):
        for j,addr in enumerate(ADDRESSES):
            f_arg = f_hash if len(inps)==0 else f_hash + pad_argument(addr)
            f_arg = f_arg.decode()
            return_output = evm.run_bytecode(f_arg, contract_abi)
            result[i, j] = int.from_bytes(return_output, "big")

    return result


def keccak_of_func_signature(func_name: str, param_types: List[str]):
    # Create a new Keccak-256 hash object
    h = keccak.new(digest_bits=256)
    input_str = func_name + "(" + ",".join(param_types) + ")"
    h.update(str.encode(input_str))
    hex_hash = h.hexdigest()
    return hex_hash[0:8].encode('utf-8')

def pad_argument(arg: str):
    if arg.startswith('0x'):
        arg = arg[2:]
    encoded = arg.encode('utf-8')
    length = len(encoded)
    padded_encoded = b'0' * (64 - length) + encoded \
        if length < 64 else encoded
    return padded_encoded


def contract_abi_json_to_array(contract_abi: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Assertion about proper contract ABI requirements met
    f_v_set, f_p_set, f_n_set = [], [], []
    for func in contract_abi:
        if func["stateMutability"]=="view":
            f_v_set.append(func)
        elif func["stateMutability"]=="payable":
            f_p_set.append(func)
        elif func["stateMutability"]=="nonpayable":
            f_n_set.append(func)

    assert len(f_v_set) <= MAX_FUNC_VIEWABLE
    assert len(f_p_set) <= MAX_FUNC_PAYABLE
    assert len(f_n_set) <= MAX_FUNC_NONPAYABLE

    abi_function_onehot = np.zeros((MAX_FUNC_TOTAL),np.int64) 
    abi_argument_onehot = np.zeros((2, MAX_FUNC_TOTAL, MAX_ARGUMENT_COUNT),np.int64)

    f_v_set += [{}] * (MAX_FUNC_VIEWABLE - len(f_v_set))
    f_p_set += [{}] * (MAX_FUNC_PAYABLE - len(f_p_set))
    f_n_set += [{}] * (MAX_FUNC_NONPAYABLE - len(f_n_set))
    whole_list = f_p_set + f_n_set + f_v_set

    argument_to_dim = {"address": 0, "uint256": 1}

    for i in range(0, MAX_FUNC_TOTAL):
        if whole_list[i] != {} and "inputs" in whole_list[i]:
            abi_function_onehot[i] = 1
            for j, argument in enumerate(whole_list[i]["inputs"]):
                abi_argument_onehot[argument_to_dim[argument["type"]], i, j] = 1

    return abi_function_onehot, abi_argument_onehot


