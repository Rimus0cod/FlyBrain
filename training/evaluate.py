"""Deterministic Milestone 001 policy evaluation and comparison metrics."""

from __future__ import annotations

import time
from typing import Any

import torch

from .ppo import PPOPolicy


def random_controller(observation: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return independent uniform motor commands for solvability diagnostics."""
    return torch.rand(
        observation.shape[0],
        2,
        device=observation.device,
        generator=generator,
    )


def scripted_controller(observation: torch.Tensor) -> torch.Tensor:
    """Steer toward the beacon using only the onboard observation contract."""
    visual = observation[..., :4]
    bearing_x = visual[..., 0] - visual[..., 2]
    bearing_y = visual[..., 1] - visual[..., 3]
    bearing = torch.atan2(bearing_y, bearing_x)
    angular_velocity = observation[..., -1]
    turn = (torch.sin(bearing) - 0.5 * angular_velocity).unsqueeze(-1)
    thrust = (0.02 + 0.18 * torch.cos(bearing).clamp_min(0.0)).unsqueeze(-1)
    return torch.cat((thrust - turn, thrust + turn), dim=-1).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(policy: PPOPolicy, environment: Any, episodes: int, seed: int) -> dict[str, float]:
    return evaluate_controller(
        policy,
        environment,
        episodes,
        seed,
        recurrent_policy=policy,
    )


@torch.no_grad()
def evaluate_controller(
    controller: Any,
    environment: Any,
    episodes: int,
    seed: int,
    recurrent_policy: PPOPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate a policy or diagnostic controller under identical metrics."""
    successes = collisions = 0
    rewards: list[float] = []
    efficiencies: list[float] = []
    successful_efficiencies: list[float] = []
    latencies: list[float] = []
    control_changes: list[float] = []
    progress_rewards: list[float] = []
    positive_progress: list[float] = []
    start_distances: list[float] = []
    end_distances: list[float] = []
    action_means: list[float] = []
    action_stds: list[float] = []
    beta_stds: list[float] = []
    progress_by_step: dict[int, list[float]] = {}
    distance_by_step: dict[int, list[float]] = {}
    for episode in range(episodes):
        observation = environment.reset(seed=seed + episode)
        state = (
            recurrent_policy.initial_state(recurrent_policy.log_concentration.device)
            if recurrent_policy is not None
            else None
        )
        if recurrent_policy is None and hasattr(controller, "generator"):
            controller.generator.manual_seed(seed + episode)
        total_reward = 0.0
        previous_action = None
        while True:
            started = time.perf_counter_ns()
            if recurrent_policy is not None:
                distribution, next_state = recurrent_policy.distribution(observation, state)
                action = distribution.mean
                beta_stds.append(float(distribution.variance.sqrt().mean()))
                state = next_state
            else:
                action = controller(observation)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            action_means.append(float(action.mean()))
            action_stds.append(float(action.std(unbiased=False)))
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += float(reward.item())
            progress_value = float(info["progress_reward"].item())
            distance_value = float(info["distance"].item())
            step_index = len(progress_rewards)
            progress_rewards.append(progress_value)
            positive_progress.append(float(progress_value > 0.0))
            progress_by_step.setdefault(step_index, []).append(progress_value)
            distance_by_step.setdefault(step_index, []).append(distance_value)
            if previous_action is None:
                start_distances.append(float(info["initial_distance"].item()))
            if previous_action is not None:
                control_changes.append(float((action - previous_action).abs().mean().item()))
            previous_action = action
            if bool((terminated | truncated).item()):
                successes += int(info["success"].item())
                collisions += int(info["collision"].item())
                if bool(info["success"].item()):
                    progress = (info["initial_distance"] - info["distance"]).clamp_min(0.0)
                    efficiency = float((progress / info["path_length"].clamp_min(1e-6)).clamp(max=1.0).item())
                    successful_efficiencies.append(efficiency)
                    efficiencies.append(efficiency)
                else:
                    # A failed episode contributes zero to cohort-level path
                    # efficiency; this avoids reporting a near-perfect number
                    # from one success when most episodes collide.
                    efficiencies.append(0.0)
                rewards.append(total_reward)
                end_distances.append(float(info["distance"].item()))
                break
    return {
        "evaluation_episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "path_efficiency": sum(efficiencies) / episodes,
        "successful_path_efficiency": sum(successful_efficiencies) / len(successful_efficiencies) if successful_efficiencies else 0.0,
        "mean_reward": sum(rewards) / len(rewards),
        "control_stability": sum(control_changes) / len(control_changes) if control_changes else 0.0,
        "parameter_count": (
            sum(parameter.numel() for parameter in recurrent_policy.parameters())
            if recurrent_policy is not None
            else 0
        ),
        "inference_latency_ms": sum(latencies) / len(latencies),
        "progress_reward_mean": sum(progress_rewards) / len(progress_rewards),
        "positive_progress_fraction": sum(positive_progress) / len(positive_progress),
        "distance_start_mean": sum(start_distances) / len(start_distances),
        "distance_end_mean": sum(end_distances) / len(end_distances),
        "action_mean": sum(action_means) / len(action_means),
        "action_std": sum(action_stds) / len(action_stds),
        "beta_std_mean": sum(beta_stds) / len(beta_stds) if beta_stds else 0.0,
        "progress_reward_by_step": [
            sum(progress_by_step[index]) / len(progress_by_step[index])
            for index in sorted(progress_by_step)
        ],
        "distance_by_step": [
            sum(distance_by_step[index]) / len(distance_by_step[index])
            for index in sorted(distance_by_step)
        ],
    }


class RandomController:
    """State-free random baseline with an isolated random-number stream."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return random_controller(observation, self.generator)
