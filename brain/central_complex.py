"""A compact recurrent heading circuit with an optional hard topology mask."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CentralComplex(nn.Module):
    """Ring-attractor engineering abstraction with external runtime state."""

    def __init__(
        self, num_neurons: int = 16, sensory_input_dim: int = 2, topology_constrained: bool = False
    ) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.topology_constrained = topology_constrained
        self.W_in = nn.Linear(sensory_input_dim, num_neurons, bias=False)
        self.W_rec_raw = nn.Parameter(self._init_ring_weights())
        self.register_buffer("connectivity_mask", self._make_connectivity_mask())

    def _make_connectivity_mask(self) -> Tensor:
        indices = torch.arange(self.num_neurons)
        distance = (indices[:, None] - indices[None, :]).abs()
        ring_distance = torch.minimum(distance, self.num_neurons - distance)
        return (ring_distance <= 1).to(torch.float32)

    def _init_ring_weights(self) -> Tensor:
        weights = torch.full((self.num_neurons, self.num_neurons), -0.2)
        for index in range(self.num_neurons):
            weights[index, index] = 1.0
            weights[index, (index - 1) % self.num_neurons] = 0.5
            weights[index, (index + 1) % self.num_neurons] = 0.5
        return weights

    @property
    def recurrent_weights(self) -> Tensor:
        """Effective synapses; forbidden entries are exactly zero when masked."""
        if self.topology_constrained:
            return self.W_rec_raw * self.connectivity_mask
        return self.W_rec_raw

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        device = device or self.W_rec_raw.device
        state = torch.zeros(batch_size, self.num_neurons, device=device)
        state[:, 0] = 1.0
        return state

    def forward(self, sensory_input: Tensor, state: Tensor | None = None) -> Tensor:
        if state is None:
            state = self.initial_state(sensory_input.shape[0], sensory_input.device)
        if state.shape != (sensory_input.shape[0], self.num_neurons):
            raise ValueError("state must have shape [batch_size, num_neurons]")
        stimulus = self.W_in(sensory_input)
        ring_effect = torch.matmul(state, self.recurrent_weights)
        new_state = F.relu(ring_effect + stimulus)
        return new_state / (new_state.sum(dim=-1, keepdim=True) + 1e-8)
