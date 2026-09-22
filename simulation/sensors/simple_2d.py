"""Onboard-like observations for ``Simple2DBody``.

The target is supplied only to construct a coarse visual signal.  Its coordinates
are never returned in the policy observation.
"""

import math

import torch


class Simple2DSensors:
    """Produces `[sin(bearing), cos(bearing), distance, closing, angular_velocity]`."""

    observation_dim = 5

    def __init__(self, max_range: float = 10.0) -> None:
        self.max_range = max_range
        self.previous_distance = None

    def reset(self) -> None:
        self.previous_distance = None

    def observe(self, body_state: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        delta = target - body_state["pos"]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        world_bearing = torch.atan2(delta[:, 1:2], delta[:, 0:1])
        relative_bearing = world_bearing - body_state["angle"]
        closing = torch.zeros_like(distance)
        if self.previous_distance is not None:
            closing = (self.previous_distance - distance).clamp(-self.max_range, self.max_range)
        self.previous_distance = distance.detach()
        return torch.cat(
            (
                torch.sin(relative_bearing),
                torch.cos(relative_bearing),
                (distance / self.max_range).clamp(0.0, 1.0),
                closing / self.max_range,
                body_state["ang_vel"] / (2 * math.pi),
            ),
            dim=-1,
        )
