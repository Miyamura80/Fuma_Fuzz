import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from evm import Py_EVM, EVM
from utils import get_bytecode
import os.path as osp
from dotmap import DotMap

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
    # contract: chex.Array # TODO: we ignore this for now!
    # victim_balances: int # TODO: for later! 

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
        self.evm = Py_EVM(args)

        self.arg_int_size = arg_int_size
        self.obs_shape = (
            6, 
            MAX_FUNC_NONPAYABLE+MAX_FUNC_PAYABLE+MAX_FUNC_VIEWABLE, 
            MAX_ARGUMENT_COUNT*(ADDRESS_SET_SIZE+arg_int_size)
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
        compilation_result = {"abi": self.contract_abi, "bin-runtime": self.bytecode}
        state, reward = step_evm(state, compilation_result, action, self.evm, self.action_function_lookup)

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        if params is None:
            params = self.default_params
        abi_function_onehot, abi_argument_onehot = contract_abi_json_to_array(self.contract_abi)
        dynamic_observation = reset_dynamic_observation(self.contract_abi, params)
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
        # BACKLOG: Seperate this into seperate dimensions
        obs = obs.at[3, 0:ADDRESS_SET_SIZE, 0].set(jnp.array([state.attacker_balance, state.contract_balance]))
        obs = obs.at[4, 0, 0].set(state.time)
        obs = obs.at[5, :MAX_FUNC_VIEWABLE, MAX_ARGUMENT_COUNT*(self.arg_int_size + ADDRESS_SET_SIZE)].set(state.view_function_output)

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
def step_evm(state: EnvState, compilation_result: dict, action: spaces.Box, evm: EVM, fname_lookup: dict) -> Tuple[EnvState, float]:
    
    # Unpack the values from the state
    func_id, payment_gwei, c_1, c_2, c_3 = action

    func_name = fname_lookup[func_id]
    # arg_val_lookup = {i: }

    inputs = {}
    args = {}
    abi_args = (func_name, inputs)

    # Execute the action on the EVM
    # evm.run_bytecode(args, compilation_result, abi_args)

    new_attacker_balance = state.attacker_balance
    new_contract_balance = state.contract_balance
    evm_state_obs = []


    state = EnvState(
        state.abi_info,
        new_attacker_balance,
        new_contract_balance,
        evm_state_obs,
        state.time + 1,
    )
    reward = 0.0

    return state, reward

# TODO
def reset_dynamic_observation(contract_abi: dict, params: EnvParams):
    pass




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
