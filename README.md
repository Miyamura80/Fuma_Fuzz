
# Fuma Fuzz - RL Smart Contract Bug Discovery


# TODO

- [ ] Think through how to model action spaces for different ABI
- [ ] Implement evm simulation environment compatible with JAX PPO for simple example
- [ ] Implement reward for small state change, with a decay over time
- [ ] Implement bias towards reentrancy via reward engineering
- [ ] (HARD) Reward behaviour of some variable number going up 
- [ ] (HARD) Reward new coverage -> but explicitly the new coverage only
- [ ] (HARD) Reward calls to dependency of contract -> try go for more coverage 
- [ ] (HARD) Did a register change?

# Backlog

- [ ] PyEVM environment for very specific "level" with a specific contract


# Formalisation


### Environment Parameters

- `attacker_initial_balance` : `0.1 ETH = 100M gwei` (approx £100)
- `address_set_size` : `2` (The "relevant addresses" $|A|=2$)
  - $A[0]$ = `creator`
  - $A[1]$ = `attacker`
  - $A[2]$ = `victim_1`
  



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


