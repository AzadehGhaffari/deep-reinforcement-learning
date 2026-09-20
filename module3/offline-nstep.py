#To compare offline return and n-step return methods in reinforcement learning with simple code, here is a concise
#Python example illustrating both concepts in a basic setting:

#Offline return here means computing the return (sum of discounted rewards) from a fixed, pre-collected trajectory (typical offline RL approach).

#n-step return means computing the return over n steps plus the value estimate after those n steps (used in n-step temporal difference methods).
#This simple code compares the usual offline return (full discounted sum of rewards) with the n-step return (partial discounted
#sum plus a bootstrapped next state value). This type of comparison is common in reinforcement learning algorithms where n-step
#returns help propagate value estimates faster while offline returns are computed on full trajectories from fixed datasets.
import numpy as np

def offline_return(rewards, gamma):
    """
    Calculate the full offline return: sum of discounted rewards over the entire trajectory
    :param rewards: list of rewards collected offline
    :param gamma: discount factor
    :return: total discounted return
    """
    return sum((gamma ** i) * r for i, r in enumerate(rewards))

def n_step_return(rewards, gamma, n, next_value=0):
    """
    Calculate n-step return: sum of discounted rewards for n steps plus discounted next state's value estimate
    :param rewards: list of rewards starting from current timestep
    :param gamma: discount factor
    :param n: number of steps to look ahead
    :param next_value: value estimate after n steps
    :return: n-step return
    """
    n = min(n, len(rewards))
    discounted_sum = sum((gamma ** i) * rewards[i] for i in range(n))
    return discounted_sum + (gamma ** n) * next_value

# Example usage
rewards = [1, 0, 2, 3, 1]
gamma = 0.9
n = 3
next_value = 5  # example value estimate at step n

offline_ret = offline_return(rewards, gamma)
n_step_ret = n_step_return(rewards, gamma, n, next_value)

print(f"Offline return (full trajectory): {offline_ret:.4f}")
print(f"{n}-step return: {n_step_ret:.4f}")
