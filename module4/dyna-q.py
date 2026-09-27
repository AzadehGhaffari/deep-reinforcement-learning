#Here is a simple Dyna-Q reinforcement learning example in Python. The code below implements the core of Dyna-Q
#including Q-learning updates, model updates, and planning steps based on experience simulation. It is designed for a deterministic environment:

#This minimal example shows the Dyna-Q loop integrating real experience updates and model-based simulated planning
#steps to speed up learning. The agent maintains a model of transitions, uses it to simulate past experiences, and updates Q-values accordingly

import numpy as np
import random

class DynaQAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha       # learning rate
        self.gamma = gamma       # discount factor
        self.epsilon = epsilon   # exploration rate
        self.planning_steps = planning_steps
        
        self.Q = np.zeros((n_states, n_actions))  # Q-value table
        # Model stores reward and next state for each state-action pair
        self.model = np.full((n_states, n_actions), None)
        self.visited_state_actions = set()
        
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        # Q-learning update from real experience
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        # Update model with observed transition
        self.model[state, action] = (reward, next_state)
        self.visited_state_actions.add((state, action))
        
        # Planning: simulate experience from model and update Q values
        for _ in range(self.planning_steps):
            sim_state, sim_action = random.choice(list(self.visited_state_actions))
            sim_reward, sim_next_state = self.model[sim_state, sim_action]
            best_sim_next_action = np.argmax(self.Q[sim_next_state])
            sim_td_target = sim_reward + self.gamma * self.Q[sim_next_state, best_sim_next_action]
            sim_td_error = sim_td_target - self.Q[sim_state, sim_action]
            self.Q[sim_state, sim_action] += self.alpha * sim_td_error

# Example usage with a simple deterministic environment
if __name__ == "__main__":
    n_states = 6
    n_actions = 2
    
    # Simple deterministic environment (reward, next_state) for each state-action pair
    env = {
        (0,0): (0, 1), (0,1): (0, 2),
        (1,0): (1, 3), (1,1): (0, 4),
        (2,0): (0, 1), (2,1): (1, 5),
        (3,0): (0, 3), (3,1): (0, 0),
        (4,0): (0, 2), (4,1): (1, 5),
        (5,0): (0, 5), (5,1): (0, 5)
    }
    
    agent = DynaQAgent(n_states, n_actions)
    episodes = 20
    max_steps = 10
    
    for episode in range(episodes):
        state = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            reward, next_state = env[(state, action)]
            agent.update(state, action, reward, next_state)
            state = next_state
    
    print("Learned Q-values:")
    print(agent.Q)
