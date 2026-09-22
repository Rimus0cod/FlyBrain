"""Compact 2D differential-drive body used only for the Milestone 001 proof of concept."""

from __future__ import annotations

import math

import torch


class Simple2DBody:
    """A bounded 2D body with two normalized, bidirectional actuator commands.

    This is intentionally an engineering abstraction rather than a quadrotor model.
    A motor value of 0.5 is neutral; the sum controls forward/reverse acceleration
    and the difference controls yaw torque. Velocity and angular rate are bounded
    so the navigation task is numerically stable and physically reachable.
    """

    def __init__(
        self,
        batch_size: int = 1,
        dt: float = 0.05,
        device: str | torch.device = "cpu",
        mass: float = 1.0,
        inertia: float = 0.5,
        linear_damping: float = 1.5,
        angular_damping: float = 2.5,
        max_thrust: float = 3.0,
        max_torque: float = 2.5,
        max_speed: float = 1.5,
        max_angular_speed: float = 2.5,
    ) -> None:
        self.batch_size = batch_size
        self.dt = dt
        self.device = torch.device(device)
        self.mass = mass
        self.inertia = inertia
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.reset()

    def reset(self) -> None:
        """Reset all agents to the origin with random heading."""
        self.pos = torch.zeros((self.batch_size, 2), device=self.device)
        self.vel = torch.zeros((self.batch_size, 2), device=self.device)
        self.angle = torch.rand((self.batch_size, 1), device=self.device) * 2 * math.pi
        self.ang_vel = torch.zeros((self.batch_size, 1), device=self.device)

    def step(self, motor_left: torch.Tensor, motor_right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance one step using normalized motor commands in [0, 1]."""
        m_left = torch.clamp(motor_left, 0.0, 1.0)
        m_right = torch.clamp(motor_right, 0.0, 1.0)

        # 0.5 is neutral. The sum controls linear acceleration and the
        # difference controls rotation.
        forward_command = m_left + m_right - 1.0
        turn_command = m_right - m_left

        linear_force = forward_command * self.max_thrust
        torque = turn_command * self.max_torque

        ang_acc = torque / self.inertia
        self.ang_vel = self.ang_vel + ang_acc * self.dt
        self.ang_vel = self.ang_vel * math.exp(-self.angular_damping * self.dt)
        self.ang_vel = self.ang_vel.clamp(
            -self.max_angular_speed, self.max_angular_speed
        )
        self.angle = (self.angle + self.ang_vel * self.dt) % (2 * math.pi)

        heading = torch.cat((torch.cos(self.angle), torch.sin(self.angle)), dim=-1)
        linear_acc = linear_force * heading / self.mass
        self.vel = self.vel + linear_acc * self.dt
        self.vel = self.vel * math.exp(-self.linear_damping * self.dt)

        speed = torch.linalg.vector_norm(self.vel, dim=-1, keepdim=True)
        scale = (self.max_speed / speed.clamp_min(1e-8)).clamp(max=1.0)
        self.vel = self.vel * scale
        self.pos = self.pos + self.vel * self.dt

        return self.get_state()

    def get_state(self) -> dict[str, torch.Tensor]:
        return {
            "pos": self.pos.clone(),
            "vel": self.vel.clone(),
            "angle": self.angle.clone(),
            "ang_vel": self.ang_vel.clone(),
        }
