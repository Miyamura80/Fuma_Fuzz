import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

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
    abi_info: chex.Array
    attacker_balance: int
    contract_balance: int
    view_function_output: chex.Array
    time: int
    # contract: chex.Array # TODO: we ignore this for now!


@struct.dataclass
class EnvParams:
    # Attack environment setup
    contract_initial_balance: int = 10
    attacker_initial_balance: int = 8 # 0.1 ETH, unit in gwei
    max_attack_time: int = 10 # Number of function invocations done before finishing

    # Action - Function argument settings
    address_set_size: int = 2 # Default |A|=2, A[0] is creator, A[1] is attacker
    uint256_arg_min: int = 1 # 1 gwei
    uint256_arg_max: int = 8 # 1 ETH


# State - Function description settings (ERROR if contract exceeds this)
MAX_FUNC_VIEWABLE: int = 10 
MAX_FUNC_NONPAYABLE: int = 10
MAX_FUNC_PAYABLE: int = 5
MAX_ARGUMENT_COUNT: int = 3


class PyEVM(environment.Environment):
    """
    JAX Compatible version of EVM environment for RL agent

    ================================================================
    ENVIRONMENT DESCRIPTION - 'PyEVM'
    ================================================================
    STATE:
    - Contract ABI information (order the payable & nonpayable attributes first)
    - Attacker Balance
    - Contract Balance
    - Time
    - Dynamic State
        - Return results of all `view` functions (not pure, since it doesn't read state)
    
    >>> (
            f_p1, ..., f_p|f_p|, f_n1, ..., f_n|f_n|, f_v1, ..., f_n|f_v|,
            attacker_balance,
            contract_balance,
            o(f_v1), o(f_v2), ... , o(f_n|f_v|)
        )
    ----------------------------------------------------------------
    ACTION:
    - f_payable <ARGUMENTS> <VALUE>
    - f_nonpayable <ARGUMENTS>

    >>> (function_id, payment_gwei, c_1, c_2, c_3)

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

    def __init__(self, contract_abi, bytecode):
        self.contract_abi = contract_abi
        self.func_num = len(filter(lambda x: x["stateMutability"]=="payable" or x["stateMutability"]=="nonpayable", contract_abi))
        self.bytecode = bytecode
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
        state, reward = step_evm(state, self.contract_abi, action)

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
        state = EnvState(
            abi_info=contract_abi_json_to_array(self.contract_abi),
            attacker_balance=10**params.attacker_initial_balance,
            contract_balance=10**params.contract_initial_balance,
            view_function_output=jnp.zeros([]), # TODO: Requires quite a bit of thinking
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
    """
    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        payable_func_dim = 0
        nonpayable_func_dim = 0
        view_func_dim = 0

        # TODO: Implement this

        # obs = jnp.zeros(self.obs_shape, dtype=bool)
        # # Set the position of the player paddle, paddle, trail & brick map
        # obs = obs.at[9, state.pos, 0].set(1)
        # obs = obs.at[state.ball_y, state.ball_x, 1].set(1)
        # obs = obs.at[state.last_y, state.last_x, 2].set(1)
        # obs = obs.at[:, :, 3].set(state.brick_map)
        # return obs.astype(jnp.float32)
        

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
        argument_index = 10+params.address_set_size
        low = jnp.array(
            [0, 0, 0, 0, 0], dtype=jnp.int64
        )
        high = jnp.array(
            [self.func_num-1, params.attacker_initial_balance, 
                argument_index, argument_index, argument_index], dtype=jnp.int64
        )
        return spaces.Box(low, high, (5,), jnp.int64)

    # TODO
    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "abi_info": spaces.Box(0,1 (10,10)),
                "attacker_balance": spaces.Box(0,10**params.attacker_initial_balance, (1,), dtype=jnp.int64),
                "contract_balance": spaces.Box(0,10**params.attacker_initial_balance, (1,), dtype=jnp.int64),
                "view_function_output": spaces.Box(0,1 (10,10)),
                "time": spaces.Discrete(params.max_attack_time),
            }
        )


# TODO
def contract_abi_json_to_array(contract_abi: dict) -> chex.Array:
    pass

# TODO
def step_evm(state: EnvState, contract_abi: dict, action: spaces.Box) -> Tuple[EnvState, float]:
    
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
