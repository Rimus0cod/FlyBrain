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
    latencies: list[float] = []
    for episode in range(episodes):
        observation = environment.reset(seed=seed + episode)
        state = policy.initial_state(policy.log_std.device)
        total_reward = 0.0
        previous_action = None
        stability: list[float] = []
        while True:
            started = time.perf_counter_ns()
            action, _, _, state = policy.act(observation, state, deterministic=True)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += float(reward.item())
            if previous_action is not None:
                stability.append(float((action - previous_action).abs().mean().item()))
            previous_action = action
            if bool((terminated | truncated).item()):
                successes += int(info["success"].item())
                collisions += int(info["collision"].item())
                if bool(info["success"].item()):
                    efficiencies.append(float((info["initial_distance"] / info["path_length"].clamp_min(1e-6)).item()))
                rewards.append(total_reward)
                break
    return {
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "path_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else 0.0,
        "mean_reward": sum(rewards) / len(rewards),
        "control_stability": sum(stability) / len(stability) if stability else 0.0,
        "parameter_count": float(sum(parameter.numel() for parameter in policy.parameters())),
        "inference_latency_ms": sum(latencies) / len(latencies),
    }
