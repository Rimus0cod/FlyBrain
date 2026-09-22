"""Small learned visual population for the first milestone."""

from torch import Tensor, nn


class VisualEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Tanh())

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)
