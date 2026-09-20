#Here is a simple example of code that approximates the state value function 
#V(s) using a linear function approximator in reinforcement learning. It uses a parameter vector (weights) and feature extraction from the state. The value is estimated as a weighted sum of state features, and the weights are updated using a temporal-difference (TD) erro

#This code demonstrates how to approximate and update a state value
#function using linear function approximation with features.
#The update uses the difference between the target value and the
#current value estimate to adjust the weights in the direction that
#reduces error. This is a core concept in reinforcement learning
#value function approximation
import numpy as np

class LinearValueFunction:
    def __init__(self, feature_dim, alpha=0.1):
        self.alpha = alpha              # learning rate
        self.weights = np.zeros(feature_dim)  # initialize weights

    def features(self, state):
        # Example feature extractor:
        # Convert the state to a feature vector (can be customized)
        # Here, assuming state is already a numpy array of features
        return state

    def value(self, state):
        # Estimate the value V(s) as weighted sum of features
        x = self.features(state)
        return np.dot(self.weights, x)

    def update(self, state, target):
        # TD update: adjust weights towards target value
        x = self.features(state)
        prediction = self.value(state)
        td_error = target - prediction
        self.weights += self.alpha * td_error * x

# Example usage:
vf = LinearValueFunction(feature_dim=3, alpha=0.1)
state = np.array([1.0, 0.5, -0.2])  # some example feature vector for a state
target_value = 10.0                  # example target (e.g., reward + discounted next value)

print("Before update:", vf.value(state))
vf.update(state, target_value)
print("After update:", vf.value(state))
