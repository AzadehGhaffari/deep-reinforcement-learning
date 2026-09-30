from __future__ import annotations

import torch
from torch import nn


class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_size
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)