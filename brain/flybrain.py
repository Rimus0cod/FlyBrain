"""Minimal Vision -> Central Complex -> Motor controller."""

import torch
from torch import nn

from .central_complex import CentralComplex
from .visual import VisualEncoder


class FlyBrainController(nn.Module):
    """Stateful controller used in the first 2D navigation milestone.

    The public interface deliberately matches :class:`BaselineMLP`: an observation
    batch becomes two normalized motor commands.  Unlike the baseline, the heading
    state persists between calls and must be reset at episode boundaries.
    """

    def __init__(
        self, visual_dim: int = 4, inertial_dim: int = 1, ring_size: int = 16
    ) -> None:
        super().__init__()
        self.visual = VisualEncoder(visual_dim, ring_size)
        self.central_complex = CentralComplex(
            num_neurons=ring_size, sensory_input_dim=ring_size + inertial_dim
        )
        self.motor = nn.Sequential(nn.Linear(ring_size, 2), nn.Sigmoid())
        self.inertial_dim = inertial_dim

    def reset_state(self, batch_size: int = 1, device=None) -> None:
        self.central_complex.reset_state(batch_size=batch_size, device=device)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        visual, inertial = torch.split(
            observation, [observation.shape[-1] - self.inertial_dim, self.inertial_dim], dim=-1
        )
        visual_activity = self.visual(visual)
        heading = self.central_complex(torch.cat((visual_activity, inertial), dim=-1))
        return self.motor(heading)
