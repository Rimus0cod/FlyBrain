"""Small, dependency-free PPO implementation for Milestone 001."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from brain import BaselineMLP, FlyBrainController, FlyBrainState


@dataclass(frozen=True)
class PPOConfig:
    training_steps: int = 2_048
    rollout_steps: int = 128
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


class PPOPolicy(nn.Module):
    """Actor-critic adapter that gives both controllers an identical PPO API."""

    def __init__(self, controller_name: str, observation_dim: int = 5) -> None:
        super().__init__()
        if controller_name == "baseline_mlp":
            self.controller: nn.Module = BaselineMLP(observation_dim)
        elif controller_name == "flybrain":
            self.controller = FlyBrainController(visual_dim=observation_dim - 1)
        else:
            raise ValueError(f"Unknown controller: {controller_name}")
        self.controller_name = controller_name
        self.log_std = nn.Parameter(torch.full((2,), -0.7))
        self.value_head = nn.Sequential(
            nn.Linear(observation_dim, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def initial_state(self, device: torch.device) -> FlyBrainState | None:
        if self.controller_name == "flybrain":
            return self.controller.initial_state(1, device)  # type: ignore[union-attr]
        return None

    def distribution(self, observation: Tensor, state: FlyBrainState | None) -> tuple[Normal, FlyBrainState | None]:
        if self.controller_name == "flybrain":
            mean, next_state = self.controller(observation, state)  # type: ignore[operator]
        else:
            mean, next_state = self.controller(observation), None
        return Normal(mean, self.log_std.exp().expand_as(mean)), next_state

    def value(self, observation: Tensor) -> Tensor:
        return self.value_head(observation).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: Tensor, state: FlyBrainState | None, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor, FlyBrainState | None]:
        distribution, next_state = self.distribution(observation, state)
        action = distribution.mean if deterministic else distribution.sample()
        action = action.clamp(0.0, 1.0)
        return action, distribution.log_prob(action).sum(-1), self.value(observation), next_state


@dataclass
class Rollout:
    observations: list[Tensor]
    actions: list[Tensor]
    log_probs: list[Tensor]
    rewards: list[Tensor]
    dones: list[Tensor]
    values: list[Tensor]
    states: list[FlyBrainState | None]
    last_value: Tensor


class PPOTrainer:
    """On-policy PPO trainer for a single deterministic 2D environment stream."""

    def __init__(self, policy: PPOPolicy, config: PPOConfig, device: str = "cpu") -> None:
        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    def collect_rollout(self, environment: Any, observation: Tensor, state: FlyBrainState | None) -> tuple[Rollout, Tensor, FlyBrainState | None, int]:
        data = {name: [] for name in ("observations", "actions", "log_probs", "rewards", "dones", "values", "states")}
        completed_episodes = 0
        for _ in range(self.config.rollout_steps):
            data["observations"].append(observation.detach())
            data["states"].append(state)
            action, log_prob, value, next_state = self.policy.act(observation, state)
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated | truncated
            data["actions"].append(action.detach())
            data["log_probs"].append(log_prob.detach())
            data["rewards"].append(reward.detach())
            data["dones"].append(done.detach())
            data["values"].append(value.detach())
            observation, state = next_observation, next_state
            if bool(done.item()):
                completed_episodes += 1
                observation = environment.reset()
                state = self.policy.initial_state(self.device)
        return Rollout(**data, last_value=self.policy.value(observation).detach()), observation, state, completed_episodes

    def update(self, rollout: Rollout) -> dict[str, float]:
        rewards = torch.stack(rollout.rewards).squeeze(-1)
        dones = torch.stack(rollout.dones).squeeze(-1).float()
        values = torch.stack(rollout.values).squeeze(-1)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=self.device)
        next_value = rollout.last_value.squeeze()
        for index in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * next_value * nonterminal - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_log_probs = torch.stack(rollout.log_probs).squeeze(-1)
        actions = rollout.actions
        metrics: dict[str, float] = {}
        for _ in range(self.config.update_epochs):
            new_log_probs, entropies, predicted_values = [], [], []
            for observation, action, state in zip(rollout.observations, actions, rollout.states):
                distribution, _ = self.policy.distribution(observation, state)
                new_log_probs.append(distribution.log_prob(action).sum(-1).squeeze())
                entropies.append(distribution.entropy().sum(-1).squeeze())
                predicted_values.append(self.policy.value(observation).squeeze())
            new_log_probs = torch.stack(new_log_probs)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = nn.functional.mse_loss(torch.stack(predicted_values), returns)
            entropy = torch.stack(entropies).mean()
            loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            metrics = {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach())}
        return metrics
