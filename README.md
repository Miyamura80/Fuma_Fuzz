
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
- (BREAK DOWN) Handle more than 1 target contract
- Exploiting randomness (e.g. creating contracts to exploit randomness)
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


# Data Analysis 
## ABI
### Function & Contract Count
![image](https://user-images.githubusercontent.com/38335479/211324732-f329d28b-445c-4dc3-9dff-d0ccf6c8e858.png)
![image](https://user-images.githubusercontent.com/38335479/211325654-3fc52d1e-d938-4b86-9105-ed9203d180a3.png)


### Function count by state mutability
![image](https://user-images.githubusercontent.com/38335479/211325051-6007aed3-6a0e-41d7-9090-d126c20a4dda.png)
![image](https://user-images.githubusercontent.com/38335479/211325071-dc271644-941d-44f5-a1df-a9def1a22845.png)


### Argument Count for each function type 
![image](https://user-images.githubusercontent.com/38335479/211928388-236a720a-4ef0-4fc2-b29d-cd21c899afb8.png)

### Argument data types

Pie chart `n=80611` from a sample of 1223 `.sol` source files.

![image](https://user-images.githubusercontent.com/38335479/211932493-af2f366c-f423-47fd-8700-d22d3731acd7.png)


### Return data types

n = 43181

![image](https://user-images.githubusercontent.com/38335479/212086053-1388e593-ba6d-47e0-8758-e669dd0604d8.png)

