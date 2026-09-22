"""Non-recurrent reference controller for controlled experiments."""

import torch
from torch import nn


class BaselineMLP(nn.Module):
    """Maps the same observation vector as FlyBrain directly to two motors."""

    def __init__(self, observation_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
