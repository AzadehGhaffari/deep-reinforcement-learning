#CartPole is a classic benchmark problem in reinforcement learning (RL)
#where an agent learns to balance a vertical pole attached to a cart
#moving along a track.

#TorchRL is PyTorch's official modular library
#built specifically for scaling and simplifying deep reinforcement
#learning applications.

#Using CartPole with TorchRL serves as the
#canonical "Hello World" example to demonstrate modular RL training
#pipelines—including environments, neural network policies, collectors,
#and replay buffers—using PyTorch.

#Key Concepts
#The Problem: The state consists of 4 continuous variables (cart position, cart velocity, pole
#angle, pole angular velocity).

#The action space is discrete: push left(0) or right (1).

#The objective is to keep the pole balanced for as many time steps as possible.  The Tools in TorchRL:GymWrapper / EnvBase:

#Wraps environments (like Gymnasium's CartPole-v1) into TensorDict-native
#objects.

#TensorDict: The core TorchRL data structure that bundles states,
#actions, rewards, and next states into a structured batch passed
#seamlessly between GPU and CPU.

#Actor / QValueActor: Modular PyTorch networks wrapped to output policy distributions or Q-values.

#SyncDataCollector: Handles environment interactions and rollout collections automatically.

#TensorDictReplayBuffer: Handles off-policy sample storage and experience sampling.

#Minimal Example (DQN with
#TorchRL)The snippet below demonstrates how CartPole is built and trained
#using Deep Q-Learning (DQN) in TorchRL:


import torch
import torch.nn as nn
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import QValueActor
from torchrl.objectives import DQNLoss
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

# 1. Environment
env = GymEnv("CartPole-v1")

# 2. Policy Network (Q-Value Actor)
net = nn.Sequential(
    nn.Linear(env.observation_spec["observation"].shape[-1], 64),
    nn.ReLU(),
    nn.Linear(64, env.action_spec.space.n)
)
actor = QValueActor(net, in_keys=["observation"], action_space=env.action_spec)

# 3. Collector (Gathers trajectories)
collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=32,
    total_frames=5000,
)

# 4. Replay Buffer & Loss Module
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=10000),
    sampler=SamplerWithoutReplacement()
)
loss_module = DQNLoss(actor, action_space=env.action_spec)
optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

# 5. Training Loop
for i, tensordict_data in enumerate(collector):
    # Add collected experience to memory
    replay_buffer.extend(tensordict_data.reshape(-1))
    
    if len(replay_buffer) >= 64:
        subdata = replay_buffer.sample(64)
        loss_vals = loss_module(subdata)
        loss = loss_vals["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_module.update_target_network()

env.close()
