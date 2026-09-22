"""PPO probability-support and bootstrap regression tests."""

import unittest

import torch

from training.ppo import PPOPolicy


class PPOPolicyTests(unittest.TestCase):
    def test_beta_actions_are_in_bounds_and_have_finite_log_prob(self) -> None:
        policy = PPOPolicy("baseline_mlp")
        observation = torch.zeros(1, 5)
        action, log_prob, _, _ = policy.act(observation, None)

        self.assertTrue(torch.all((action > 0.0) & (action < 1.0)))
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_recurrent_policy_state_is_external_to_parameters(self) -> None:
        policy = PPOPolicy("flybrain")
        state = policy.initial_state(torch.device("cpu"))
        _, _, _, next_state = policy.act(torch.zeros(1, 5), state)

        self.assertIsNotNone(next_state)
        self.assertNotIn("heading", {name for name, _ in policy.named_parameters()})


if __name__ == "__main__":
    unittest.main()
