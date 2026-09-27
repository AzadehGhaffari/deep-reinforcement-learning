#Here is a simple example of tabular Dyna-Q implementation in Python that combines real experience updates and model-planning updates:
#This code defines a Dyna-Q agent with a Q-table and a model storing observed transitions. The agent updates Q-values
#through real experience and also performs planning updates from the stored model transitions, which speeds up learning compared to plain Q-learning.

#This code is simple and intended for tabular problem settings with a small number of states and discrete actions.

import numpy as np

class DynaQAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.planning_steps = planning_steps
        
        self.Q = np.zeros((n_states, n_actions))  # Q-value table
        self.model = {}  # model to store transitions (state, action): (reward, next_state)
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)  # explore
        else:
            return np.argmax(self.Q[state])  # exploit
    
    def learn(self, state, action, reward, next_state):
        # Q-learning update
        q_predict = self.Q[state, action]
        q_target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (q_target - q_predict)
        
        # Update model with the observed transition
        self.model[(state, action)] = (reward, next_state)
        
        # Planning with simulated experience
        for _ in range(self.planning_steps):
            s, a = list(self.model.keys())[np.random.choice(len(self.model))]
            r, s_ = self.model[(s, a)]
            q_predict = self.Q[s, a]
            q_target = r + self.gamma * np.max(self.Q[s_])
            self.Q[s, a] += self.alpha * (q_target - q_predict)


# Example usage:
if __name__ == "__main__":
    n_states = 10
    n_actions = 2
    agent = DynaQAgent(n_states, n_actions)
    
    # Simulating an environment interaction loop
    for episode in range(20):
        state = np.random.randint(n_states)
        for step in range(10):
            action = agent.choose_action(state)
            # Simulate environment response
            next_state = (state + 1) % n_states
            reward = 1 if next_state == n_states - 1 else 0
            agent.learn(state, action, reward, next_state)
            state = next_state
    print("Learned Q-values:\n", agent.Q)
