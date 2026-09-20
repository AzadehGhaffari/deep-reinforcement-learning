#The Bellman equation is a fundamental recursive relationship in
#reinforcement learning and dynamic programming, expressing the
#value of a state as its immediate reward plus the expected discounted
#value of the next state.

# Value function for Bellman equation
#This program updates the value of each state by applying the
#Bellman equation—reward plus discounted value of the next state.
#Bellman Equation Explained
#Recursive structure: V(s)=max_a[R(s,a)+γV(s′)], where 
#s is the current state, 
#a is the action, 
#R(s,a) is the immediate reward, 
#s′ is the next state, and 
#γ is the discount factor
#The discount factor in Bellman's equation determines how much future rewards are valued compared to immediate rewards

#Purpose: It helps agents evaluate not just immediate rewards but also potential future rewards, encouraging better long-term decisions.

#Implementation: Used in value iteration, policy iteration, and Q-learning, forming the backbone for many RL algorithms.

#Learning Value Iteration
#The simple function above shows value updates for two states. In real applications, these updates are
#repeated until values converge, modeling how the Bellman equation is used in learning optimal policies.

#This example demonstrates the core Bellman equation mechanism used in reinforcement learning and dynamic programming.
def bellman_value(V, R, gamma, transitions, state):
    """
    V: dictionary of value estimates for states.
    R: dictionary of rewards for (state, action).
    gamma: discount factor (0 < gamma < 1).
    transitions: dictionary (state, action) -> next_state.
    state: current state.

    Returns updated value for the current state using Bellman equation.
    """
    value = float('-inf')
    for action in transitions[state]:
        next_state = transitions[state][action]
        reward = R.get((state, action), 0)
        val = reward + gamma * V[next_state]
        if val > value:
            value = val
    return value

# Example usage:
states = ['A', 'B']
V = {'A': 0, 'B': 0}
R = {('A', 'go_B'): 5, ('B', 'go_A'): 2}
transitions = {'A': {'go_B': 'B'}, 'B': {'go_A': 'A'}}
gamma = 0.9
for state in states:
    V[state] = bellman_value(V, R, gamma, transitions, state)
print(V)
