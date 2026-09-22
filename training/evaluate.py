"""Deterministic Milestone 001 policy evaluation and comparison metrics."""

from __future__ import annotations

import time
from typing import Any

import torch

from .ppo import PPOPolicy


@torch.no_grad()
def evaluate(policy: PPOPolicy, environment: Any, episodes: int, seed: int) -> dict[str, float]:
    successes = collisions = 0
    rewards: list[float] = []
    efficiencies: list[float] = []
    successful_efficiencies: list[float] = []
    latencies: list[float] = []
    control_changes: list[float] = []
    for episode in range(episodes):
        observation = environment.reset(seed=seed + episode)
        state = policy.initial_state(policy.log_concentration.device)
        total_reward = 0.0
        previous_action = None
        while True:
            started = time.perf_counter_ns()
            action, _, _, state = policy.act(observation, state, deterministic=True)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += float(reward.item())
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
                break
    return {
        "evaluation_episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "path_efficiency": sum(efficiencies) / episodes,
        "successful_path_efficiency": sum(successful_efficiencies) / len(successful_efficiencies) if successful_efficiencies else 0.0,
        "mean_reward": sum(rewards) / len(rewards),
        "control_stability": sum(control_changes) / len(control_changes) if control_changes else 0.0,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "inference_latency_ms": sum(latencies) / len(latencies),
    }
