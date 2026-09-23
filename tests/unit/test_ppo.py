"""PPO probability-support and bootstrap regression tests."""

import unittest

import torch

from training.ppo import PPOPolicy, compute_gae


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
