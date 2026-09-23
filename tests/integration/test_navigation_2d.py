"""Integration test for the 2D environment-policy wiring."""

import unittest

import torch

from brain import BaselineMLP, FlyBrainController
from simulation.environment import Navigation2DEnvironment


class Navigation2DIntegrationTests(unittest.TestCase):
    def test_each_controller_matches_environment_action_contract(self) -> None:
        environment = Navigation2DEnvironment(batch_size=2)
        observation = environment.reset()
        controllers = (
            BaselineMLP(observation_dim=observation.shape[-1]),
            FlyBrainController(visual_dim=observation.shape[-1] - 1),
        )

        for controller in controllers:
            if isinstance(controller, FlyBrainController):
                motors, _ = controller(observation, controller.initial_state(batch_size=2))
            else:
                motors = controller(observation)
            next_observation, reward, terminated, truncated, info = environment.step(motors)

            self.assertEqual(tuple(next_observation.shape), (2, 9))
            self.assertEqual(tuple(reward.shape), (2,))
            self.assertEqual(tuple(terminated.shape), (2,))
            self.assertEqual(tuple(truncated.shape), (2,))
            self.assertEqual(tuple(info["distance"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
