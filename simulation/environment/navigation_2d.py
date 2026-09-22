"""A deliberately simple 2D point-navigation task for controller comparisons."""

import torch

from simulation.body import Simple2DBody
from simulation.sensors import Simple2DSensors


class Navigation2DEnvironment:
    """Keeps training rewards and ground truth outside the controller interface."""

    def __init__(
        self,
        batch_size: int = 1,
        target_radius: float = 0.35,
        arena_radius: float = 6.0,
        max_steps: int = 200,
        device="cpu",
    ) -> None:
        self.body = Simple2DBody(batch_size=batch_size, device=device)
        self.sensors = Simple2DSensors()
        self.batch_size = batch_size
        self.target_radius = target_radius
        self.arena_radius = arena_radius
        self.max_steps = max_steps
        self.device = device
        self.target = None
        self.steps = 0
        self.path_length = torch.zeros(batch_size, device=device)
        self.initial_distance = torch.zeros(batch_size, device=device)
        self.previous_distance = torch.zeros(batch_size, device=device)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        self.body.reset()
        self.sensors.reset()
        self.steps = 0
        self.path_length.zero_()
        self.target = (torch.rand(self.batch_size, 2, device=self.device) * 2 - 1) * 4
        self.initial_distance = torch.linalg.vector_norm(self.target - self.body.pos, dim=-1)
        self.previous_distance = self.initial_distance.clone()
        return self.sensors.observe(self.body.get_state(), self.target)

    def step(
        self, motors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.target is None:
            raise RuntimeError("Call reset() before step().")
        state = self.body.step(motors[:, 0:1], motors[:, 1:2])
        self.steps += 1
        distance = torch.linalg.vector_norm(self.target - state["pos"], dim=-1)
        self.path_length += torch.linalg.vector_norm(state["vel"], dim=-1) * self.body.dt
        success = distance < self.target_radius
        collision = torch.linalg.vector_norm(state["pos"], dim=-1) >= self.arena_radius
        terminated = success | collision
        truncated = torch.full_like(terminated, self.steps >= self.max_steps)
        progress_reward = self.previous_distance - distance
        stability_reward = -0.01 * state["ang_vel"].abs().squeeze(-1)
        energy_penalty = -0.001 * motors.square().mean(dim=-1)
        completion_reward = success.float() * 10.0
        collision_penalty = -collision.float() * 5.0
        reward = progress_reward + stability_reward + energy_penalty + completion_reward + collision_penalty
        self.previous_distance = distance
        observation = self.sensors.observe(state, self.target)
        return observation, reward, terminated, truncated, {
            "distance": distance,
            "success": success,
            "collision": collision,
            "path_length": self.path_length.clone(),
            "initial_distance": self.initial_distance.clone(),
            "progress_reward": progress_reward,
            "stability_reward": stability_reward,
            "energy_penalty": energy_penalty,
            "completion_reward": completion_reward,
            "collision_penalty": collision_penalty,
        }
