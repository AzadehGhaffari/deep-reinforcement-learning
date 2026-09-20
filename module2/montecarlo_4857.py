#A simple Monte Carlo method for reinforcement learning involves
#simulating episodes under a policy and averaging the returns
#(rewards) for each state to estimate state values

import numpy as np

#Code Explanation
#Environment: The code defines a sequential environment with 5 states; moving to the next state
#always gives a reward of 1, and the last state is terminal.

#Policy: A random policy is used for demonstration, choosing actions randomly.

#Monte Carlo Evaluation: For each episode, the code keeps track of all (state, reward) pairs and computes the total
#return (sum of discounted rewards) for every state visited, then averages those returns to build the value table estimate.

#Output: The final value table estimates, after 1000 episodes, shows the expected total reward from starting in each state under the random policy.

#This code demonstrates how Monte Carlo methods use sampled experiences to estimate value functions in
#reinforcement learning, without needing a model of the environment.

# Environment with 5 states; each step gives reward 1 until terminal
class SimpleEnvironment:
    def __init__(self, num_states=5):
        self.num_states = num_states

    def step(self, state):
        reward = 0
        terminal = False
        if state < self.num_states - 1:
            next_state = state + 1
            reward = 1
        else:
            next_state = state
            terminal = True
        return next_state, reward, terminal

    def reset(self):
        return 0  # Starting state

# Random policy
def random_policy(state, num_actions=5):
    return np.random.choice(num_actions)

# Monte Carlo method to evaluate the policy
def monte_carlo_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    value_table = np.zeros(env.num_states)
    returns = {state: [] for state in range(env.num_states)}
    for _ in range(num_episodes):
        state = env.reset()
        episode = []
        # Generate one episode
        while True:
            action = policy(state)
            next_state, reward, terminal = env.step(action)
            episode.append((state, reward))
            if terminal:
                break
            state = next_state
        # Calculate the return and update the value table
        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            returns[state].append(G)
            value_table[state] = np.mean(returns[state])
    return value_table

num_episodes = 1000
env = SimpleEnvironment(num_states=5)
v = monte_carlo_policy_evaluation(random_policy, env, num_episodes)
print("The value table is:")
print(v)
