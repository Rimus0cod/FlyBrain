"""Phase 2 solvability checks for the Milestone 001 navigation task."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from simulation.environment import Navigation2DEnvironment
from training.run_experiment import load_checkpoint


@dataclass(frozen=True)
class SolvabilityResult:
    controller: str
    episodes: int
    success_rate: float
    collision_rate: float
    truncation_rate: float
    mean_final_distance: float
    mean_steps: float


class RandomController:
    """Uniform random actuator commands; observation-independent."""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(observation[..., :2]) * 0.9 + 0.05


class ScriptedBeaconController:
    """Uses only the four visual beacon receptors plus no privileged state."""

    _angles = torch.tensor((0.0, math.pi / 2, math.pi, 3 * math.pi / 2))

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        visual = observation[..., :4]
        angles = self._angles.to(observation.device, observation.dtype)
        direction = torch.stack((angles.cos(), angles.sin()), dim=-1)
        vector = visual @ direction
        desired_heading = torch.atan2(vector[..., 1], vector[..., 0])

        # Positive bearing is a left turn in the body frame. Keep steering
        # bounded and reduce forward drive while significantly misaligned.
        turn = torch.clamp(desired_heading / 1.2, -0.4, 0.4)
        alignment = torch.cos(desired_heading).clamp(min=0.0)
        base = 0.55 + 0.25 * alignment
        left = (base - turn).clamp(0.05, 0.95)
        right = (base + turn).clamp(0.05, 0.95)
        return torch.stack((left, right), dim=-1)


def run_controller(
    controller: Callable[[torch.Tensor], torch.Tensor],
    episodes: int = 20,
    start_seed: int = 10_000,
) -> SolvabilityResult:
    environment = Navigation2DEnvironment(batch_size=1)
    successes = collisions = truncations = 0
    final_distances: list[float] = []
    step_counts: list[int] = []

    for episode in range(episodes):
        observation = environment.reset(seed=start_seed + episode)
        for step_index in range(1, environment.max_steps + 1):
            action = controller(observation)
            observation, _, terminated, truncated, info = environment.step(action)
            if bool(terminated.item()) or bool(truncated.item()):
                successes += int(info["success"].item())
                collisions += int(info["collision"].item())
                truncations += int(truncated.item() and not terminated.item())
                final_distances.append(float(info["distance"].item()))
                step_counts.append(step_index)
                break

    return SolvabilityResult(
        controller=controller.__class__.__name__,
        episodes=episodes,
        success_rate=successes / episodes,
        collision_rate=collisions / episodes,
        truncation_rate=truncations / episodes,
        mean_final_distance=sum(final_distances) / len(final_distances),
        mean_steps=sum(step_counts) / len(step_counts),
    )


@torch.no_grad()
def run_checkpoint(
    checkpoint: Path, episodes: int, start_seed: int = 20_000
) -> dict[str, float]:
    policy = load_checkpoint(checkpoint)
    environment = Navigation2DEnvironment(batch_size=1)
    successes = collisions = truncations = 0
    rewards = []
    final_distances = []

    for episode in range(episodes):
        observation = environment.reset(seed=start_seed + episode)
        state = policy.initial_state(torch.device("cpu"))

        for _ in range(environment.max_steps):
            action, _, _, state = policy.act(observation, state, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            rewards.append(float(reward.item()))
            if bool(terminated.item()) or bool(truncated.item()):
                successes += int(info["success"].item())
                collisions += int(info["collision"].item())
                truncations += int(truncated.item() and not terminated.item())
                final_distances.append(float(info["distance"].item()))
                break

    return {
        "controller": policy.controller_name,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "truncation_rate": truncations / episodes,
        "mean_reward_per_step": sum(rewards) / len(rewards),
        "mean_final_distance": sum(final_distances) / len(final_distances),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {
        "random": run_controller(
            RandomController(), episodes=args.episodes, start_seed=args.start_seed
        ).__dict__,
        "scripted": run_controller(
            ScriptedBeaconController(),
            episodes=args.episodes,
            start_seed=args.start_seed,
        ).__dict__,
    }

    if args.checkpoint:
        results["trained_checkpoint"] = run_checkpoint(
            args.checkpoint, episodes=args.episodes, start_seed=args.start_seed + 10_000
        )

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
