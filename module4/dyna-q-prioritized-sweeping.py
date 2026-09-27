#To write simple code combining Dyna-Q with prioritized sweeping, it helps to understand that
#Dyna-Q is a model-based RL algorithm where the agent learns from both real and simulated experiences,
#updating its Q-values using a learned model of the environment. Prioritized Sweeping improves planning efficiency
#by focusing on states/actions that are likely to have the largest impact on the value function update, using a priority queue.

#A Python pseudocode sketch that includes both Dyna-Q machinery and a prioritized sweeping mechanism:

import numpy as np
import heapq

class DynaQPrioritizedSweepingAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5, theta=0.0001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.theta = theta  # priority threshold
        
        self.Q = np.zeros((n_states, n_actions))
        self.model = dict()  # model: (state, action) -> (reward, next_state)
        self.predecessors = dict()  # reverse model: next_state -> set of (state, action)
        
        self.priority_queue = []
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update_model(self, state, action, reward, next_state):
        self.model[(state, action)] = (reward, next_state)
        # Track predecessors for prioritized sweeping
        if next_state not in self.predecessors:
            self.predecessors[next_state] = set()
        self.predecessors[next_state].add((state, action))
    
    def q_update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.Q[next_state])
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * delta
        return delta
    
    def planning(self):
        steps = 0
        while self.priority_queue and steps < self.planning_steps:
            # Pop the highest priority element (max heap simulated with negative priority)
            priority, (state, action) = heapq.heappop(self.priority_queue)
            priority = -priority  # invert back to positive
            
            reward, next_state = self.model[(state, action)]
            delta = self.q_update(state, action, reward, next_state)
            
            # For each predecessor of the current state
            if state in self.predecessors:
                for pred_state, pred_action in self.predecessors[state]:
                    r_p, s_p_next = self.model[(pred_state, pred_action)]
                    target_p = r_p + self.gamma * np.max(self.Q[state])
                    delta_p = target_p - self.Q[pred_state, pred_action]
                    if abs(delta_p) > self.theta:
                        # Push with negative delta for max-priority queue
                        heapq.heappush(self.priority_queue, (-abs(delta_p), (pred_state, pred_action)))
            
            steps += 1
    
    def learn(self, state, action, reward, next_state):
        # Update Q with real experience
        delta = self.q_update(state, action, reward, next_state)
        
        # Update model
        self.update_model(state, action, reward, next_state)
        
        # If priority is large enough, add to priority queue
        if abs(delta) > self.theta:
            heapq.heappush(self.priority_queue, (-abs(delta), (state, action)))
        
        # Planning step
        self.planning()
