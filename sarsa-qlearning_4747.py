#The key difference between SARSA and Q-learning is that SARSA is
#an on-policy algorithm, updating its Q-values based on the
#action actually taken next,
#while Q-learning is an off-policy algorithm,
#updating its Q-values based on the best possible next action,
#regardless of the policy's choice.


import numpy as np
import random

#This code shows the basic mechanics:

#SARSA updates 
#Q(s,a) values using the actual next action 
#a′chosen by the policy (on-policy),

#Q-learning updates 
#Q(s,a) using the maximal 
#Q(s′,a′) over all actions 
#a ′(off-policy).

#Both use temporal difference updates but differ in how they choose the next action value to update towards. 
#This difference affects how they learn optimal policies in environments where exploration strategies influence performance.

# Environment setup (toy example)
states = [0, 1, 2]
actions = [0, 1]  # Two possible actions in each state

# Parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Exploration rate

# Initialize Q-tables
Q_sarsa = np.zeros((len(states), len(actions)))
Q_qlearning = np.zeros((len(states), len(actions)))

def choose_action(Q, state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return np.argmax(Q[state])      # Exploit

# Dummy reward and next state function for testing
def step(state, action):
    next_state = (state + 1) % len(states)  # Cycle through states
    reward = 1 if next_state == 0 else 0    # Reward when returning to start
    return next_state, reward

# One step update for SARSA
def sarsa_update(state, action, reward, next_state, next_action):
    predict = Q_sarsa[state, action]
    target = reward + gamma * Q_sarsa[next_state, next_action]
    Q_sarsa[state, action] += alpha * (target - predict)

# One step update for Q-learning
def qlearning_update(state, action, reward, next_state):
    predict = Q_qlearning[state, action]
    target = reward + gamma * np.max(Q_qlearning[next_state])
    Q_qlearning[state, action] += alpha * (target - predict)

# Run a few episodes to demonstrate
for episode in range(5):
    state = 0
    action = choose_action(Q_sarsa, state)
    for _ in range(10):
        next_state, reward = step(state, action)

        # SARSA update needs to select next action according to current policy
        next_action = choose_action(Q_sarsa, next_state)
        sarsa_update(state, action, reward, next_state, next_action)

        # Q-learning update uses max next Q value, independent of policy
        qlearning_update(state, action, reward, next_state)

        state = next_state
        action = next_action

print("Q-values from SARSA:")
print(Q_sarsa)
print("\nQ-values from Q-learning:")
print(Q_qlearning)
