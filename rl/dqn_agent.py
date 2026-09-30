from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.dqn_network import DQNetwork
from rl.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config: dict) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(config["project"]["device"])
        dqn_cfg = config["dqn"]

        self.gamma = float(dqn_cfg["gamma"])
        self.batch_size = int(dqn_cfg["batch_size"])
        self.epsilon = float(dqn_cfg["epsilon_start"])
        self.epsilon_end = float(dqn_cfg["epsilon_end"])
        self.epsilon_decay = float(dqn_cfg["epsilon_decay"])
        self.warmup_steps = int(dqn_cfg["warmup_steps"])
        self.learn_steps = 0

        hidden_dims = list(dqn_cfg["hidden_dims"])
        self.policy_net = DQNetwork(state_size, action_size, hidden_dims).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(dqn_cfg["learning_rate"]))
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(int(dqn_cfg["buffer_size"]))

    def select_action(self, state: np.ndarray, exploit: bool = False) -> int:
        if not exploit and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_t).gather(1, actions_t)
        next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.learn_steps % int(self.config["dqn"]["target_update_frequency"]) == 0:
            self.update_target_network()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_end))