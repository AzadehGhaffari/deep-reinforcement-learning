#The n-step learning method in reinforcement learning updates estimates based on the sum of the next 
#n rewards, as well as the estimated value (bootstrapped) from the state 
#n steps ahead, blending standard TD and MC approaches.

#Intuition
#Instead of updating after every action, n-step learning accumulates 
#n rewards, then uses this sequence and the bootstrapped value to update the original state-action pair.

#Key Formula
#For state st , the n-step return is:

#G t(n) =R t+1+γR t+2+γ 2R t+3+…+γ n−1R t+n+γ nV(s t+n)
#where 

#γ is the discount factor, and 
#V(s t+n) is the estimate at the (future) nth state.

#Simple Python Example
#Here's simple code implementing n-step TD learning for a generic episodic RL environment:
# n-step TD update for state-value function
alpha = 0.1         # learning rate
gamma = 0.99        # discount factor
n = 3               # number of steps
V = {}              # state-value dictionary

def n_step_update(states, rewards):
    # states: list of states [s_t, ..., s_{t+n}]
    # rewards: list of rewards [r_{t+1}, ..., r_{t+n}]
    G = 0
    for i in range(len(rewards)):
        G += (gamma ** i) * rewards[i]
    if len(states) > n:
        G += (gamma ** n) * V.get(states[-1], 0)   # bootstrapped value from s_{t+n}
    s_t = states
    V[s_t] = V.get(s_t, 0) + alpha * (G - V.get(s_t, 0))
