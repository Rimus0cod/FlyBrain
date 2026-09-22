"""A deliberately simple 2D point-navigation task for controller comparisons."""

import torch

from simulation.body import Simple2DBody
from simulation.sensors import Simple2DSensors


class Navigation2DEnvironment:
    """Keeps training rewards and ground truth outside the controller interface."""

    def __init__(self, batch_size: int = 1, target_radius: float = 0.35, device="cpu") -> None:
        self.body = Simple2DBody(batch_size=batch_size, device=device)
        self.sensors = Simple2DSensors()
        self.batch_size = batch_size
        self.target_radius = target_radius
        self.device = device
        self.target = None

    def reset(self) -> torch.Tensor:
        self.body.reset()
        self.sensors.reset()
        self.target = (torch.rand(self.batch_size, 2, device=self.device) * 2 - 1) * 4
        return self.sensors.observe(self.body.get_state(), self.target)

    def step(self, motors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        state = self.body.step(motors[:, 0:1], motors[:, 1:2])
        distance = torch.linalg.vector_norm(self.target - state["pos"], dim=-1)
        terminated = distance < self.target_radius
        reward = -distance + terminated.float() * 10.0
        observation = self.sensors.observe(state, self.target)
        return observation, reward, terminated, {"distance": distance}
