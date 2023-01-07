
# RL Smart Contract Bug Discovery


# TODO

- [ ] Look through professional RL repo setup (maybe google)
- [ ] Implement evm simulation environment compatible with JAX PPO for simple example
- [ ] PyEVM environment for very specific "level" with a specific contract


# Formalisation


### Modelling

The "relevant addresses" $|A|=2$: 
- `creator`
  - `owner` (if not same as `creator`)
- `attacker`

Initial Parameters:
- `attacker_balance` : `0.1 ETH = 100M gwei` (approx £100)

$I=\text{Set of values to send ETH}$


### State Space: $State$

- `contributions[creator]` $:: |I|$
- `contributions[attacker]` $:: |I|$  
- `owner` $:: |A|$

Thus state space is: $|State|= |A||I|^2$


### Action Space: $Action$

#### Public functions (With potential EVM state modification) 
- `Fallback`
- `contribute()` 
- `withdraw()`

Each public function associated with $|I|$ size of`msg.value` to send with it. 

Thus action space is: $|Action|=3|I|$


# Backlog


