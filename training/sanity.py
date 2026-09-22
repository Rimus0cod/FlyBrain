"""Non-learning controllers used only to validate the Milestone 001 task."""

import math

import torch


def beacon_action(observation: torch.Tensor) -> torch.Tensor:
    """Turn toward the four-receptor beacon estimate and apply forward thrust.

    This is a task-physics diagnostic, not a policy baseline and not a learned
    controller. It uses only the public observation vector.
    """
    visual = observation[:, :4]
    direction_x = visual[:, 0] - visual[:, 2]
    direction_y = visual[:, 1] - visual[:, 3]
    heading_error = torch.atan2(direction_y, direction_x) / math.pi
    turn = heading_error.clamp(-1.0, 1.0) * 0.20
    # The 2D body's thrust scale is intentionally large; this low thrust avoids
    # crossing the marker between discrete simulation steps.
    thrust = torch.full_like(turn, 0.05)
    return torch.stack(((thrust - turn).clamp(0.0, 1.0), (thrust + turn).clamp(0.0, 1.0)), dim=-1)
