"""Minimal Vision -> Central Complex -> Motor controller."""

from dataclasses import dataclass

from torch import Tensor, nn

from .central_complex import CentralComplex
from .visual import VisualEncoder


@dataclass(frozen=True)
class FlyBrainState:
    """Episode-specific runtime activity; it contains no learned parameters."""

    heading: Tensor


class FlyBrainController(nn.Module):
    """Pure recurrent controller used in the first 2D navigation milestone."""

    def __init__(
        self,
        visual_dim: int = 4,
        inertial_dim: int = 1,
        ring_size: int = 16,
        topology_constrained: bool = False,
    ) -> None:
        super().__init__()
        self.visual = VisualEncoder(visual_dim, ring_size)
        self.central_complex = CentralComplex(
            num_neurons=ring_size,
            sensory_input_dim=ring_size + inertial_dim,
            topology_constrained=topology_constrained,
        )
        self.motor = nn.Sequential(nn.Linear(ring_size, 2), nn.Sigmoid())
        self.inertial_dim = inertial_dim

    def initial_state(self, batch_size: int, device=None) -> FlyBrainState:
        return FlyBrainState(self.central_complex.initial_state(batch_size, device))

    def forward(self, observation: Tensor, state: FlyBrainState | None = None) -> tuple[Tensor, FlyBrainState]:
        if observation.shape[-1] <= self.inertial_dim:
            raise ValueError("observation must contain visual and inertial features")
        if state is None:
            state = self.initial_state(observation.shape[0], observation.device)
        visual, inertial = observation[..., :-self.inertial_dim], observation[..., -self.inertial_dim:]
        visual_activity = self.visual(visual)
        heading = self.central_complex(
            self._concat_features(visual_activity, inertial), state.heading
        )
        return self.motor(heading), FlyBrainState(heading)

    @staticmethod
    def _concat_features(visual_activity: Tensor, inertial: Tensor) -> Tensor:
        import torch

        return torch.cat((visual_activity, inertial), dim=-1)
