"""Task physics sanity tests independent of PPO."""

import unittest

import torch

from simulation.environment import Navigation2DEnvironment
from training.sanity import beacon_action


class EnvironmentSanityTests(unittest.TestCase):
    def test_random_actions_are_finite_and_within_physical_contract(self) -> None:
        environment = Navigation2DEnvironment(max_steps=20)
        observation = environment.reset(seed=1)
        for _ in range(20):
            observation, reward, terminated, truncated, _ = environment.step(torch.rand(1, 2))
            self.assertTrue(torch.isfinite(observation).all())
            self.assertTrue(torch.isfinite(reward).all())
            if bool((terminated | truncated).item()):
                break

    def test_scripted_beacon_controller_reaches_some_seeded_targets(self) -> None:
        successes = 0
        for seed in range(20):
            environment = Navigation2DEnvironment(max_steps=200)
            observation = environment.reset(seed=seed)
            while True:
                observation, _, terminated, truncated, info = environment.step(beacon_action(observation))
                if bool((terminated | truncated).item()):
                    successes += int(info["success"].item())
                    break
        self.assertGreater(successes, 0)


if __name__ == "__main__":
    unittest.main()
