"""Small, dependency-free PPO implementation for Milestone 001."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Beta

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

    def __init__(
        self,
        controller_name: str,
        observation_dim: int = 9,
    ) -> None:
        super().__init__()

        if controller_name == "baseline_mlp":
            self.controller: nn.Module = BaselineMLP(observation_dim)

        elif controller_name == "flybrain":
            self.controller = FlyBrainController(
                visual_dim=observation_dim - 1
            )

        else:
            raise ValueError(
                f"Unknown controller: {controller_name}"
            )

        self.controller_name = controller_name

        # Beta has support exactly on (0, 1), so sampled actuator
        # commands need no clamp and their log-probability remains valid.
        self.log_concentration = nn.Parameter(
            torch.full((2,), 2.5)
        )

        if controller_name == "baseline_mlp":
            nn.init.constant_(
                self.controller.network[4].bias,  # type: ignore[union-attr]
                -2.2,
            )
        else:
            nn.init.constant_(
                self.controller.motor[0].bias,  # type: ignore[union-attr]
                -2.2,
            )

        value_input_dim = observation_dim + (
            16 if controller_name == "flybrain" else 0
        )

        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def initial_state(
        self,
        device: torch.device,
    ) -> FlyBrainState | None:
        if self.controller_name == "flybrain":
            return self.controller.initial_state(  # type: ignore[union-attr]
                1,
                device,
            )

        return None

    def distribution(
        self,
        observation: Tensor,
        state: FlyBrainState | None,
    ) -> tuple[Beta, FlyBrainState | None]:

        if self.controller_name == "flybrain":
            mean, next_state = self.controller(  # type: ignore[operator]
                observation,
                state,
            )
        else:
            mean, next_state = self.controller(
                observation
            ), None

        concentration = (
            self.log_concentration
            .exp()
            .expand_as(mean)
        )

        return (
            Beta(
                mean * concentration + 1.0,
                (1.0 - mean) * concentration + 1.0,
            ),
            next_state,
        )

    def value(
        self,
        observation: Tensor,
        state: FlyBrainState | None = None,
    ) -> Tensor:

        if self.controller_name == "flybrain":
            if state is None:
                raise ValueError(
                    "FlyBrain critic requires the recurrent state"
                )

            observation = torch.cat(
                (
                    observation,
                    state.heading,
                ),
                dim=-1,
            )

        return self.value_head(observation).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        state: FlyBrainState | None,
        deterministic: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        FlyBrainState | None,
    ]:

        distribution, next_state = self.distribution(
            observation,
            state,
        )

        action = (
            distribution.mean
            if deterministic
            else distribution.sample()
        )

        return (
            action,
            distribution.log_prob(action).sum(-1),
            self.value(observation, state),
            next_state,
        )


@dataclass
class Rollout:
    observations: list[Tensor]
    actions: list[Tensor]
    log_probs: list[Tensor]
    rewards: list[Tensor]
    terminated: list[Tensor]
    episode_ended: list[Tensor]
    bootstrap_values: list[Tensor]
    values: list[Tensor]
    states: list[FlyBrainState | None]
    episode_rewards: list[float]
    episode_successes: int
    episode_collisions: int
    reward_components: dict[str, list[Tensor]]


def compute_gae(
    rewards: Tensor,
    terminated: Tensor,
    episode_ended: Tensor,
    bootstrap_values: Tensor,
    values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(
        (),
        device=rewards.device,
    )

    for index in reversed(range(len(rewards))):
        delta = (
            rewards[index]
            + gamma * bootstrap_values[index]
            - values[index]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * (1.0 - episode_ended[index])
            * gae
        )

        advantages[index] = gae

    return advantages


def _correlation(
    left: Tensor,
    right: Tensor,
) -> float:

    left_centered = left - left.mean()
    right_centered = right - right.mean()

    denominator = (
        left_centered.square().sum().sqrt()
        * right_centered.square().sum().sqrt()
    )

    if float(denominator) == 0.0:
        return 0.0

    return float(
        (left_centered * right_centered).sum()
        / denominator
    )


def _lagged_pairs(
    signal: Tensor,
    action_feature: Tensor,
    episode_ended: Tensor,
    lag: int,
) -> tuple[Tensor, Tensor]:

    if lag == 0:
        return signal, action_feature

    if lag >= signal.numel():
        return signal[:0], action_feature[:0]

    valid = torch.ones(
        signal.numel() - lag,
        dtype=torch.bool,
        device=signal.device,
    )

    for offset in range(lag):
        valid &= ~episode_ended[
            offset:
            signal.numel() - lag + offset
        ].bool()

    return (
        signal[lag:][valid],
        action_feature[:-lag][valid],
    )


def _lagged_credit_metrics(
    advantages: Tensor,
    progress_rewards: Tensor,
    forward_command: Tensor,
    abs_turn_command: Tensor,
    episode_ended: Tensor,
    lags: tuple[int, ...] = (0, 1, 2, 4, 8, 16),
) -> dict[str, float]:

    metrics: dict[str, float] = {}

    for lag in lags:
        lagged_advantage, lagged_forward = _lagged_pairs(
            advantages,
            forward_command,
            episode_ended,
            lag,
        )

        _, lagged_turn = _lagged_pairs(
            advantages,
            abs_turn_command,
            episode_ended,
            lag,
        )

        lagged_progress, lagged_forward_for_progress = (
            _lagged_pairs(
                progress_rewards,
                forward_command,
                episode_ended,
                lag,
            )
        )

        _, lagged_turn_for_progress = _lagged_pairs(
            progress_rewards,
            abs_turn_command,
            episode_ended,
            lag,
        )

        suffix = f"_lag_{lag}"

        metrics[
            f"advantage_forward_command_correlation{suffix}"
        ] = _correlation(
            lagged_advantage,
            lagged_forward,
        )

        metrics[
            f"advantage_abs_turn_correlation{suffix}"
        ] = _correlation(
            lagged_advantage,
            lagged_turn,
        )

        metrics[
            f"progress_forward_command_correlation{suffix}"
        ] = _correlation(
            lagged_progress,
            lagged_forward_for_progress,
        )

        metrics[
            f"progress_abs_turn_correlation{suffix}"
        ] = _correlation(
            lagged_progress,
            lagged_turn_for_progress,
        )

        positive = lagged_advantage > 0

        metrics[
            f"positive_advantage_forward_mean{suffix}"
        ] = (
            float(
                lagged_forward[positive].mean()
            )
            if bool(positive.any())
            else 0.0
        )

        metrics[
            f"negative_advantage_forward_mean{suffix}"
        ] = (
            float(
                lagged_forward[~positive].mean()
            )
            if bool((~positive).any())
            else 0.0
        )

        metrics[
            f"positive_advantage_abs_turn_mean{suffix}"
        ] = (
            float(
                lagged_turn[positive].mean()
            )
            if bool(positive.any())
            else 0.0
        )

        metrics[
            f"negative_advantage_abs_turn_mean{suffix}"
        ] = (
            float(
                lagged_turn[~positive].mean()
            )
            if bool((~positive).any())
            else 0.0
        )

    return metrics


class PPOTrainer:
    """On-policy PPO trainer for a single deterministic 2D environment stream."""

    def __init__(
        self,
        policy: PPOPolicy,
        config: PPOConfig,
        device: str = "cpu",
        episode_seed: int = 0,
    ) -> None:

        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
        )

        self.episode_seed = episode_seed
        self.episode_index = 0

    def collect_rollout(
        self,
        environment: Any,
        observation: Tensor,
        state: FlyBrainState | None,
    ) -> tuple[
        Rollout,
        Tensor,
        FlyBrainState | None,
        int,
    ]:

        data = {
            name: []
            for name in (
                "observations",
                "actions",
                "log_probs",
                "rewards",
                "terminated",
                "episode_ended",
                "bootstrap_values",
                "values",
                "states",
            )
        }

        component_names = (
            "progress_reward",
            "stability_reward",
            "energy_penalty",
            "completion_reward",
            "collision_penalty",
        )

        reward_components = {
            name: []
            for name in component_names
        }

        completed_episodes = 0
        episode_rewards: list[float] = []

        episode_successes = 0
        episode_collisions = 0

        current_episode_reward = 0.0

        for _ in range(self.config.rollout_steps):
            data["observations"].append(
                observation.detach()
            )

            data["states"].append(state)

            action, log_prob, value, next_state = (
                self.policy.act(
                    observation,
                    state,
                )
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            done = terminated | truncated

            data["actions"].append(
                action.detach()
            )

            data["log_probs"].append(
                log_prob.detach()
            )

            data["rewards"].append(
                reward.detach()
            )

            for name in component_names:
                reward_components[name].append(
                    info[name].detach()
                )

            current_episode_reward += float(
                reward.item()
            )

            data["terminated"].append(
                terminated.detach()
            )

            data["episode_ended"].append(
                done.detach()
            )

            data["values"].append(
                value.detach()
            )

            # A time-limit truncation bootstraps from V(s[t+1]),
            # but a true terminal state does not.
            next_value = self.policy.value(
                next_observation,
                next_state,
            ).detach()

            data["bootstrap_values"].append(
                torch.where(
                    terminated,
                    torch.zeros_like(next_value),
                    next_value,
                )
            )

            observation = next_observation
            state = next_state

            if bool(done.item()):
                completed_episodes += 1

                episode_rewards.append(
                    current_episode_reward
                )

                current_episode_reward = 0.0

                episode_successes += int(
                    info["success"].item()
                )

                episode_collisions += int(
                    info["collision"].item()
                )

                self.episode_index += 1

                observation = environment.reset(
                    seed=self.episode_seed
                    + self.episode_index
                )

                state = self.policy.initial_state(
                    self.device
                )

        return (
            Rollout(
                **data,
                episode_rewards=episode_rewards,
                episode_successes=episode_successes,
                episode_collisions=episode_collisions,
                reward_components=reward_components,
            ),
            observation,
            state,
            completed_episodes,
        )

    def update(
        self,
        rollout: Rollout,
    ) -> dict[str, float]:

        rewards = torch.stack(
            rollout.rewards
        ).squeeze(-1)

        terminated = torch.stack(
            rollout.terminated
        ).squeeze(-1).float()

        episode_ended = torch.stack(
            rollout.episode_ended
        ).squeeze(-1).float()

        bootstrap_values = torch.stack(
            rollout.bootstrap_values
        ).squeeze(-1)

        values = torch.stack(
            rollout.values
        ).squeeze(-1)

        # Truncations bootstrap their own V(s[t+1]),
        # but neither terminal nor truncated episodes may
        # leak advantages into the next reset.
        advantages = compute_gae(
            rewards,
            terminated,
            episode_ended,
            bootstrap_values,
            values,
            self.config.gamma,
            self.config.gae_lambda,
        )

        gae_decay_metrics = {}

        for lag in (1, 2, 4, 8, 16):
            if lag < len(rewards):
                gae_decay_metrics[
                    f"gae_discount_lag_{lag}"
                ] = (
                    self.config.gamma
                    * self.config.gae_lambda
                ) ** lag

        returns = advantages + values

        raw_advantage_mean = advantages.mean()

        raw_advantage_std = advantages.std(
            unbiased=False
        )

        advantages = (
            advantages - raw_advantage_mean
        ) / (
            advantages.std(unbiased=False)
            + 1e-8
        )

        old_log_probs = torch.stack(
            rollout.log_probs
        ).squeeze(-1)

        actions = rollout.actions

        sampled_actions = torch.stack(
            actions
        ).squeeze(1)

        forward_command = sampled_actions.mean(
            dim=-1
        )

        abs_turn_command = (
            sampled_actions[:, 1]
            - sampled_actions[:, 0]
        ).abs()

        progress_rewards = torch.stack(
            rollout.reward_components[
                "progress_reward"
            ]
        ).squeeze(-1)

        metrics: dict[str, float] = {}

        for epoch in range(
            self.config.update_epochs
        ):
            new_log_probs = []
            entropies = []
            predicted_values = []

            for observation, action, state in zip(
                rollout.observations,
                actions,
                rollout.states,
            ):
                distribution, _ = (
                    self.policy.distribution(
                        observation,
                        state,
                    )
                )

                new_log_probs.append(
                    distribution.log_prob(
                        action
                    ).sum(-1).squeeze()
                )

                entropies.append(
                    distribution.entropy()
                    .sum(-1)
                    .squeeze()
                )

                predicted_values.append(
                    self.policy.value(
                        observation,
                        state,
                    ).squeeze()
                )

            new_log_probs = torch.stack(
                new_log_probs
            )

            ratio = (
                new_log_probs
                - old_log_probs
            ).exp()

            clipped = ratio.clamp(
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio,
            )

            clip_fraction = (
                (
                    (ratio < 1.0 - self.config.clip_ratio)
                    |
                    (ratio > 1.0 + self.config.clip_ratio)
                )
                .float()
                .mean()
            )

            approx_kl = (
                old_log_probs
                - new_log_probs
            ).mean()

            policy_loss = -torch.minimum(
                ratio * advantages,
                clipped * advantages,
            ).mean()

            value_loss = nn.functional.mse_loss(
                torch.stack(predicted_values),
                returns,
            )

            entropy = torch.stack(
                entropies
            ).mean()

            loss = (
                policy_loss
                + self.config.value_coefficient
                * value_loss
                - self.config.entropy_coefficient
                * entropy
            )

            self.optimizer.zero_grad()

            loss.backward()

            gradient_norm = (
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
            )

            self.optimizer.step()

            if (
                not torch.isfinite(loss)
                or not torch.isfinite(
                    new_log_probs
                ).all()
            ):
                raise FloatingPointError(
                    "PPO update produced NaN or Inf"
                )

        metrics = {
            **gae_decay_metrics,

            "mean_action": float(
                sampled_actions.mean()
            ),

            "std_action": float(
                sampled_actions.std(
                    unbiased=False
                )
            ),

            "min_action": float(
                sampled_actions.min()
            ),

            "max_action": float(
                sampled_actions.max()
            ),

            "mean_value": float(
                values.mean()
            ),

            "mean_advantage": float(
                raw_advantage_mean
            ),

            "mean_ratio": float(
                ratio.mean().detach()
            ),

            "ratio_std": float(
                ratio.std(
                    unbiased=False
                ).detach()
            ),

            "ratio_min": float(
                ratio.min().detach()
            ),

            "ratio_max": float(
                ratio.max().detach()
            ),

            "clip_fraction": float(
                clip_fraction.detach()
            ),

            "approx_kl": float(
                approx_kl.detach()
            ),

            "policy_loss": float(
                policy_loss.detach()
            ),

            "value_loss": float(
                value_loss.detach()
            ),

            "entropy": float(
                entropy.detach()
            ),

            "gradient_norm": float(
                gradient_norm.detach()
            ),

            "episode_reward": (
                sum(rollout.episode_rewards)
                / len(rollout.episode_rewards)
                if rollout.episode_rewards
                else 0.0
            ),

            "episode_success_rate": (
                rollout.episode_successes
                / len(rollout.episode_rewards)
                if rollout.episode_rewards
                else 0.0
            ),

            "episode_collision_rate": (
                rollout.episode_collisions
                / len(rollout.episode_rewards)
                if rollout.episode_rewards
                else 0.0
            ),

            **{
                name: float(
                    torch.stack(
                        component_values
                    ).mean()
                )
                for name, component_values
                in rollout.reward_components.items()
            },

            "raw_advantage_std": float(
                raw_advantage_std
            ),

            "positive_advantage_fraction": float(
                (advantages > 0)
                .float()
                .mean()
            ),

            "advantage_progress_correlation": (
                _correlation(
                    advantages,
                    progress_rewards,
                )
            ),

            "advantage_forward_command_correlation": (
                _correlation(
                    advantages,
                    forward_command,
                )
            ),

            "advantage_abs_turn_correlation": (
                _correlation(
                    advantages,
                    abs_turn_command,
                )
            ),

            "progress_forward_command_correlation": (
                _correlation(
                    progress_rewards,
                    forward_command,
                )
            ),

            "progress_abs_turn_correlation": (
                _correlation(
                    progress_rewards,
                    abs_turn_command,
                )
            ),

            "positive_advantage_action_mean": (
                float(
                    sampled_actions[
                        advantages > 0
                    ].mean()
                )
                if bool(
                    (advantages > 0).any()
                )
                else 0.0
            ),

            "negative_advantage_action_mean": (
                float(
                    sampled_actions[
                        advantages <= 0
                    ].mean()
                )
                if bool(
                    (advantages <= 0).any()
                )
                else 0.0
            ),

            "positive_advantage_forward_mean": (
                float(
                    forward_command[
                        advantages > 0
                    ].mean()
                )
                if bool(
                    (advantages > 0).any()
                )
                else 0.0
            ),

            "negative_advantage_forward_mean": (
                float(
                    forward_command[
                        advantages <= 0
                    ].mean()
                )
                if bool(
                    (advantages <= 0).any()
                )
                else 0.0
            ),

            "positive_advantage_abs_turn_mean": (
                float(
                    abs_turn_command[
                        advantages > 0
                    ].mean()
                )
                if bool(
                    (advantages > 0).any()
                )
                else 0.0
            ),

            "negative_advantage_abs_turn_mean": (
                float(
                    abs_turn_command[
                        advantages <= 0
                    ].mean()
                )
                if bool(
                    (advantages <= 0).any()
                )
                else 0.0
            ),

            **_lagged_credit_metrics(
                advantages,
                progress_rewards,
                forward_command,
                abs_turn_command,
                episode_ended,
            ),
        }

        return metrics