"""PPO probability-support and bootstrap regression tests."""

import unittest

import torch

from training.ppo import PPOConfig, PPOPolicy, PPOTrainer, Rollout, compute_gae


class PPOPolicyTests(unittest.TestCase):
    def test_gae_assigns_positive_advantage_to_positive_progress(self) -> None:
        advantages = compute_gae(
            rewards=torch.tensor([1.0]),
            terminated=torch.tensor([0.0]),
            episode_ended=torch.tensor([1.0]),
            bootstrap_values=torch.tensor([0.0]),
            values=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        self.assertGreater(advantages.item(), 0.0)

    def test_beta_actions_are_in_bounds_and_have_finite_log_prob(self) -> None:
        policy = PPOPolicy("baseline_mlp")
        observation = torch.zeros(1, 9)
        action, log_prob, _, _ = policy.act(observation, None)

        self.assertTrue(torch.all((action > 0.0) & (action < 1.0)))
        self.assertTrue(torch.isfinite(log_prob).all())
        distribution, _ = policy.distribution(observation, None)
        self.assertTrue(torch.all(distribution.concentration1 > 0))
        self.assertTrue(torch.all(distribution.concentration0 > 0))

    def test_recurrent_policy_state_is_external_to_parameters(self) -> None:
        policy = PPOPolicy("flybrain")
        state = policy.initial_state(torch.device("cpu"))
        _, _, _, next_state = policy.act(torch.zeros(1, 9), state)

        self.assertIsNotNone(next_state)
        self.assertNotIn("heading", {name for name, _ in policy.named_parameters()})


    def test_update_reports_action_credit_diagnostics(self) -> None:
        policy = PPOPolicy("baseline_mlp")
        observations = [torch.zeros(1, 9) for _ in range(3)]
        actions = [
            torch.tensor([[0.18, 0.20]]),
            torch.tensor([[0.16, 0.28]]),
            torch.tensor([[0.22, 0.18]]),
        ]
        reward_components = {
            "progress_reward": [
                torch.tensor([0.10]),
                torch.tensor([-0.02]),
                torch.tensor([0.03]),
            ],
            "stability_reward": [torch.zeros(1) for _ in range(3)],
            "energy_penalty": [torch.zeros(1) for _ in range(3)],
            "completion_reward": [torch.zeros(1) for _ in range(3)],
            "collision_penalty": [torch.zeros(1) for _ in range(3)],
        }
        rewards = [sum(parts) for parts in zip(*reward_components.values())]
        rollout = Rollout(
            observations=observations,
            actions=actions,
            log_probs=[torch.zeros(1) for _ in range(3)],
            rewards=rewards,
            terminated=[torch.tensor([False]), torch.tensor([False]), torch.tensor([True])],
            episode_ended=[torch.tensor([False]), torch.tensor([False]), torch.tensor([True])],
            bootstrap_values=[torch.zeros(1) for _ in range(3)],
            values=[policy.value(obs).detach() for obs in observations],
            states=[None, None, None],
            episode_rewards=[sum(float(r.item()) for r in rewards)],
            episode_successes=0,
            episode_collisions=1,
            reward_components=reward_components,
        )
        metrics = PPOTrainer(
            policy, PPOConfig(update_epochs=1)
        ).update(rollout)

        for key in (
            "advantage_forward_command_correlation",
            "advantage_abs_turn_correlation",
            "progress_forward_command_correlation",
            "progress_abs_turn_correlation",
            "positive_advantage_forward_mean",
            "negative_advantage_forward_mean",
            "positive_advantage_abs_turn_mean",
            "negative_advantage_abs_turn_mean",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))

    def test_flybrain_critic_requires_and_uses_recurrent_state(self) -> None:
        policy = PPOPolicy("flybrain")
        observation = torch.zeros(1, 9)
        state = policy.initial_state(torch.device("cpu"))

        with self.assertRaises(ValueError):
            policy.value(observation)
        first = policy.value(observation, state)
        second = policy.value(
            observation,
            type(state)(state.heading.roll(1, dims=-1)),
        )

        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
