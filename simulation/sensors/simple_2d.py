"""Onboard-like observations for :class:`~simulation.body.Simple2DBody`.

The navigation target belongs to the simulator. It is rendered only as a
low-resolution visual beacon; its coordinates, bearing and distance never enter
the policy observation. Temporal differences are computed from consecutive
beacon observations.
"""

import math

import torch


class Simple2DSensors:
    """    Four panoramic beacon receptors, their temporal difference, and a
    normalized angular-rate sensor.

    Each receptor has a broad, fixed preferred direction in the body frame.
    Receptor intensity is an engineering visual abstraction, not a claim about a
    biological photoreceptor model.
    """

    vision_dim = 4
    temporal_dim = vision_dim
    inertial_dim = 1
    observation_dim = vision_dim + temporal_dim + inertial_dim

    def __init__(self, visual_noise_std: float = 0.0, imu_noise_std: float = 0.0) -> None:
        self.visual_noise_std = visual_noise_std
        self.imu_noise_std = imu_noise_std
        self._receptor_angles = torch.linspace(0, 2 * math.pi, steps=self.vision_dim + 1)[:-1]
        self._previous_visual: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset temporal sensor state at the start of an episode."""
        self._previous_visual = None

    def observe(self, body_state: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        delta = target - body_state["pos"]
        world_bearing = torch.atan2(delta[:, 1:2], delta[:, 0:1])
        relative_bearing = world_bearing - body_state["angle"]
        receptor_angles = self._receptor_angles.to(body_state["pos"].device)
        phase = relative_bearing - receptor_angles.unsqueeze(0)
        visual = torch.relu(torch.cos(phase))
        if self._previous_visual is None:
            temporal_difference = torch.zeros_like(visual)
        else:
            temporal_difference = visual - self._previous_visual
        self._previous_visual = visual.detach()
        angular_velocity = body_state["ang_vel"] / (2 * math.pi)
        if self.visual_noise_std:
            visual = visual + torch.randn_like(visual) * self.visual_noise_std
        if self.imu_noise_std:
            angular_velocity = angular_velocity + torch.randn_like(angular_velocity) * self.imu_noise_std
        return torch.cat(
            (
                visual.clamp(0.0, 1.0),
                temporal_difference.clamp(-1.0, 1.0),
                angular_velocity.clamp(-1.0, 1.0),
            ),
            dim=-1,
        )
