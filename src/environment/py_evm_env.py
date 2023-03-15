import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, List
import chex
from flax import struct
from evm import Py_EVM, EVM
import os.path as osp
from dotmap import DotMap
from utils import get_bytecode
from functools import partial
from jax import jit
from Crypto.Hash import keccak
from typing import List
from eth_utils import encode_hex



"""
ASSUMPTIONS:
-1 Target Contract Only
-No Exploits through contract creation
-The contract content data is not needed for finding the bugs
-Gas is not a relevant decision for the action space
-Only argument types are uint256 and address types (limited)
    -Addresses are irrelevant to function - only that they are distinct
    -uint256 just simply goes up in powers of 10 from 1 gwei to 1ETH
"""


@struct.dataclass
class EnvState:
    abi_function_onehot: chex.Array
    abi_argument_onehot: chex.Array
    attacker_balance: int
    contract_balance: int
    view_function_output: chex.Array
    time: int
    # contract: chex.Array # BACKLOG: we ignore this for now!
    # victim_balances: int # BACKLOG: for later! 

@struct.dataclass
class EnvParams:
    # Attack environment setup
    contract_initial_balance: int = 10
    attacker_initial_balance: int = 8 # 0.1 ETH, unit in gwei
    max_attack_time: int = 10 # Number of function invocations done before finishing

    # Action - Function argument settings
    uint256_arg_min: int = 0 # 1 gwei
    uint256_arg_max: int = 10 # 1 ETH


# State - Function description settings (ERROR if contract exceeds this)
MAX_FUNC_VIEWABLE: int = 10 
MAX_FUNC_NONPAYABLE: int = 10
MAX_FUNC_PAYABLE: int = 5
MAX_FUNC_TOTAL: int = MAX_FUNC_VIEWABLE + MAX_FUNC_NONPAYABLE + MAX_FUNC_PAYABLE

MAX_ARGUMENT_COUNT: int = 3

# Action - Function argument settings
ADDRESS_SET_SIZE: int = 2 # Default |A|=2, A[0] is attacker, A[1] is contract, A[2],... is victims

ADDRESSES = {
                "0x69420552091a69125d5dfcb7b8c2659029395bdf": 0, 
                "0x38644552091a69125d5dfcb7b8c2659029395bdf": 1, 
            }

class PyEVM_Env(environment.Environment):
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
    # TODO: Collapse this into args
    def __init__(self, contract_names):
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
        self.bytecode = compiled_json["bin-runtime"]

        self.action_func_num = len(list(filter(lambda x: x["stateMutability"]=="payable" or x["stateMutability"]=="nonpayable", self.contract_abi)))
        
        # Utils
        self.action_function_lookup, self.dynamic_obs_func_lookup  = create_function_lookup(self.contract_abi)
        
        # Setting up the observation attributes
        temp_params = self.default_params
        arg_int_size = temp_params.uint256_arg_max - temp_params.uint256_arg_min + 1
        
        # EVM 
        self.evm = Py_EVM(args, self.bytecode, self.contract_abi)

        self.arg_int_size = arg_int_size
        self.obs_shape = (
            5, 
            MAX_FUNC_NONPAYABLE+MAX_FUNC_PAYABLE+MAX_FUNC_VIEWABLE, 
            max(MAX_ARGUMENT_COUNT, ADDRESS_SET_SIZE)
        )
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        state, reward = step_evm(params, state, self.contract_abi, action, self.evm, self.action_function_lookup)

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.float32(reward),
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        if params is None:
            params = self.default_params
        self.evm.reset()
        abi_function_onehot, abi_argument_onehot = contract_abi_json_to_array(self.contract_abi)
        dynamic_observation = reset_dynamic_observation(self.contract_abi, params, self.evm)
        state = EnvState(
            abi_function_onehot=abi_function_onehot,
            abi_argument_onehot=abi_argument_onehot,
            attacker_balance=10**params.attacker_initial_balance,
            contract_balance=10**params.contract_initial_balance,
            view_function_output=dynamic_observation, 
            time=0,
        )
        return self.get_obs(state), state

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
    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(self.obs_shape, dtype=jnp.int64)
        # Set the observation tensor according to above spec
        obs = obs.at[0, :, 0].set(state.abi_function_onehot)
        obs = obs.at[1:3, :, 0:MAX_ARGUMENT_COUNT].set(state.abi_argument_onehot)
        obs = obs.at[3, 0:ADDRESS_SET_SIZE, 0].set(jnp.array([state.attacker_balance, state.contract_balance]))
        obs = obs.at[4, :MAX_FUNC_VIEWABLE, :ADDRESS_SET_SIZE].set(state.view_function_output)

        return obs.astype(jnp.float64)


    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Has contract been drained? 
        # BACKLOG: Improve this termination condition
        done1 = state.contract_balance == 0

        done_steps = state.time >= params.max_attack_time
        return jnp.logical_or(done_steps, done1)

    @property
    def name(self) -> str:
        """Environment name."""
        return "PyEVM"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params

        # Where c_i from [0,10] -> maps to uint256, c_i from [11,12] -> maps to addresses
        # >>> (function_id, payment_gwei, c_1, c_2, c_3)
        argument_index = 10+ADDRESS_SET_SIZE
        low = jnp.array(
            [0, 0, 0, 0, 0], dtype=jnp.int64
        )
        high = jnp.array(
            [self.action_func_num-1, params.attacker_initial_balance, 
                argument_index, argument_index, argument_index], dtype=jnp.int64
        )
        return spaces.Box(low, high, (5,), jnp.int64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 10**params.attacker_initial_balance, self.obs_shape, jnp.int64)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        (num_dim, height, width) = self.obs_shape
        view_function_output_width = MAX_ARGUMENT_COUNT*(self.arg_int_size + ADDRESS_SET_SIZE)
        return spaces.Dict(
            {
                "abi_function_onehot": spaces.Box(0, 1, (height, 1)),
                "abi_argument_onehot": spaces.Box(0, 1, (2, height, MAX_ARGUMENT_COUNT)),
                "attacker_balance": spaces.Box(0, jnp.finfo(jnp.int64).max, (1,), dtype=jnp.int64),
                "contract_balance": spaces.Box(0, jnp.finfo(jnp.int64).max, (1,), dtype=jnp.int64),
                "view_function_output": spaces.Box(0,jnp.finfo(jnp.int64).max (MAX_FUNC_VIEWABLE, view_function_output_width)),
                "time": spaces.Discrete(0, params.max_attack_time, (1,), dtype=jnp.int8),
            }
        )

# Takes the contract_abi and returns indexing from func_id -> func_name
def create_function_lookup(contract_abi: dict) -> Tuple[chex.Array, chex.Array]:
    # These are the sets that are viewable, or actionable
    f_v_set = [func for func in contract_abi if func["stateMutability"]=="view"]
    f_action_set = [func for func in contract_abi if func["stateMutability"]=="payable" or func["stateMutability"]=="nonpayable"]
    
    f_v_set += [{}] * (MAX_FUNC_VIEWABLE - len(f_v_set))
    f_action_set += [{}] * (MAX_FUNC_PAYABLE + MAX_FUNC_NONPAYABLE - len(f_action_set))

    action_function_lookup = jnp.array([get_f_hash_int(contract_abi, f_action_set[i]["name"]) if f_action_set[i] != {} and "name" in f_action_set[i] else 0 for i in range(MAX_FUNC_PAYABLE+MAX_FUNC_NONPAYABLE)])
    dynamic_obs_func_lookup = jnp.array([get_f_hash_int(contract_abi, f_v_set[i]["name"]) if f_v_set[i] != {} and "name" in f_v_set[i] else 0 for i in range(MAX_FUNC_VIEWABLE)])

    return action_function_lookup, dynamic_obs_func_lookup

def contract_abi_json_to_array(contract_abi: dict) -> Tuple[chex.Array, chex.Array]:
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

    abi_function_onehot = jnp.zeros((MAX_FUNC_TOTAL),jnp.int64) 
    abi_argument_onehot = jnp.zeros((2, MAX_FUNC_TOTAL, MAX_ARGUMENT_COUNT),jnp.int64)

    f_v_set += [{}] * (MAX_FUNC_VIEWABLE - len(f_v_set))
    f_p_set += [{}] * (MAX_FUNC_PAYABLE - len(f_p_set))
    f_n_set += [{}] * (MAX_FUNC_NONPAYABLE - len(f_n_set))
    whole_list = f_p_set + f_n_set + f_v_set

    argument_to_dim = {"address": 0, "uint256": 1}

    for i in range(0, MAX_FUNC_TOTAL):
        if whole_list[i] != {} and "inputs" in whole_list[i]:
            abi_function_onehot = abi_function_onehot.at[i].set(1)
            for j, argument in enumerate(whole_list[i]["inputs"]):
                abi_argument_onehot = abi_argument_onehot.at[argument_to_dim[argument["type"]], i, j].set(1)

    return abi_function_onehot, abi_argument_onehot

# TODO
def step_evm(params: EnvParams, state: EnvState, contract_abi: dict, action: spaces.Box, evm: EVM, fname_lookup: chex.Array) -> Tuple[EnvState, float]:
    # Unpack the values from the state
    # func_id, payment_gwei, c_1, c_2, c_3 = action
    # func_hash: int = jnp.array(fname_lookup)[func_id]

    inputs = {}
    args = {}

    # Execute the action on the EVM
    # evm.run_bytecode(args, compilation_result, abi_args)

    new_attacker_balance = state.attacker_balance
    new_contract_balance = state.contract_balance
    dynamic_observation = reset_dynamic_observation(contract_abi, params, evm)


    state = EnvState(
        state.abi_function_onehot,
        state.abi_argument_onehot,
        new_attacker_balance,
        new_contract_balance,
        dynamic_observation,
        state.time + 1,
    )
    reward = 0.0

    return state, reward

def get_f_hash_int(contract_abi: dict, f_name: str) -> int:
    # Find the function input signature
    func = [func for func in contract_abi if "name" in func and func["name"]==f_name][0]
    input_types = [inp["type"] for inp in func["inputs"]]
    f_hash = keccak_of_func_signature(f_name, input_types)
    return int(f_hash.decode(), 16)

powers_of_10 = {i: 10**i for i in range(0, 10)}
def get_hash_function_call(contract_abi: dict, f_name: str, arguments: List[int], params: EnvParams) -> str:
    # Find the function input signature
    func = [func for func in contract_abi if "name" in func and func["name"]==f_name][0]
    input_types = [inp["type"] for inp in func["inputs"]]
    f_hash = keccak_of_func_signature(f_name, input_types)
    
    arg_in_values = []
    for i,arg in enumerate(arguments):
        if i < len(input_types)-1:
            if arg <= params.arg_int_size and input_types[i]=="uint256":
                c_i = powers_of_10[arg]
            elif arg < params.arg_int_size + ADDRESS_SET_SIZE and input_types[i]=="address":
                c_i = ADDRESSES.keys()[arg - params.arg_int_size]
            else:
                c_i = 0

            c_i = pad_argument(c_i)
            arg_in_values.append(c_i)
    
    f_arg = f_hash + b''.join(arg_in_values)
    return f_arg.decode()


def keccak_of_func_signature(func_name: str, param_types: List[str]):
    # Create a new Keccak-256 hash object
    h = keccak.new(digest_bits=256)
    input_str = func_name + "(" + ",".join(param_types) + ")"
    h.update(str.encode(input_str))
    hex_hash = h.hexdigest()
    return hex_hash[0:8].encode('utf-8')

def pad_argument(arg: any):
    if arg.startswith('0x'):
        arg = arg[2:]
    encoded = arg.encode('utf-8')
    length = len(encoded)
    padded_encoded = b'0' * (64 - length) + encoded \
        if length < 64 else encoded
    return padded_encoded


# BACKLOG: With int inputs as well
def reset_dynamic_observation(contract_abi: dict, params: EnvParams, evm: EVM) -> chex.Array:
    viewable_abi = [func for i,func in enumerate(contract_abi) if func['stateMutability']=='view']
    f_hash_typemap = [(keccak_of_func_signature(func['name'],[inp["type"] for inp in func['inputs']]), [inp["type"] for inp in func['inputs']], func['name'], func['outputs'][0]['type']) for func in viewable_abi]

    result = jnp.zeros((MAX_FUNC_VIEWABLE, ADDRESS_SET_SIZE), dtype=jnp.int64)
    for i, (f_hash, inps, f_name, return_type) in enumerate(f_hash_typemap):
        for j,addr in enumerate(evm.accounts):
            f_args = [] if len(inps)==0 else [addr]
            (return_output, r_type) = evm.run_bytecode(f_name, f_args, 0)
            # TODO: Gotta handle the case where it returns address
            result.at[i, j].set(return_output)

    return result



    #     jnp.maximum(0, state.pos - 1) * (action == 1)
    #     + jnp.minimum(9, state.pos + 1) * (action == 3)
    #     + state.pos * jnp.logical_and(action != 1, action != 3)
    # new_x = (
    #     (state.ball_x - 1) * (state.ball_dir == 0)
    #     + (state.ball_x + 1) * (state.ball_dir == 1))

    # border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    # new_x = jax.lax.select(
    #     border_cond_x, (0 * (new_x < 0) + 9 * (new_x > 9)), new_x
    # )
    # ball_dir = jax.lax.select(
    #     border_cond_x, jnp.array([1, 0, 3, 2])[state.ball_dir], state.ball_dir
    # )

    # # Spawn new bricks if there are no more around - everything is collected
    # spawn_bricks = jnp.logical_and(
    #     brick_cond, jnp.count_nonzero(brick_map) == 0
    # )
