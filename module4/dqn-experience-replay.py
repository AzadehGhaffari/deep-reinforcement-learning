#To demonstrate how experience replay is used in Deep Q-Network (DQN)
#reinforcement learning, here is a simple Python code example outlining the
#core concepts:

#The ReplayMemory class stores experience tuples (state, action, reward,next_state, done).

#Experiences are stored during environmental interaction.

#During training, a batch of random samples is drawn from the replay buffer.

#The DQN neural network updates Q-values ​​based on these samples.

#This process breaks temporal correlations between sequential data samples
#and improves training stability and efficiency.

#This code captures the key use of experience replay in DQN training—storing
#and sampling past experiences for more stable and efficient learning.
#The target network is used to calculate stable target Q-values.
#The buffer reduces the correlation of samples and allows multiple
#reuses of past experience for better sample efficiency

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

# Define a simple neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# A namedtuple to represent a single transition
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Replay memory to store experience tuples
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Update function using experience replay batch
def update_network(main_net, target_net, memory, optimizer, batch_size, gamma=0.99):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.stack(batch.state)
    actions = torch.tensor(batch.action).unsqueeze(1)
    rewards = torch.tensor(batch.reward).unsqueeze(1)
    next_states = torch.stack(batch.next_state)
    dones = torch.tensor(batch.done).unsqueeze(1)

    # Current Q values
    current_q_values = main_net(states).gather(1, actions)

    # Target Q values from next states using target network
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(current_q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Example usage:
input_dim = 4  # e.g., state space size
output_dim = 2  # e.g., number of actions
main_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(main_net.state_dict())
optimizer = optim.Adam(main_net.parameters())

memory = ReplayMemory(10000)
batch_size = 32

# During interaction with environment (pseudo code):
# state, action, reward, next_state, done = env.step(...)
# memory.push(state_tensor, action, reward, next_state_tensor, done)

# Periodically call update_network to train on experience replay batches
# update_network(main_net, target_net, memory, optimizer, batch_size)
