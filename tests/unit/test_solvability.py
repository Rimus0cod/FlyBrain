"""Regression tests proving that the Milestone 001 task is physically solvable."""

import unittest

import torch

from training.solvability import RandomController, ScriptedBeaconController, run_controller


class SolvabilityTests(unittest.TestCase):
    def test_random_controller_outputs_valid_actions(self) -> None:
        action = RandomController()(torch.zeros(1, 5))
        self.assertEqual(tuple(action.shape), (1, 2))
        self.assertTrue(torch.all((action >= 0.05) & (action <= 0.95)))

    def test_scripted_beacon_solves_fixed_regression_seeds(self) -> None:
        result = run_controller(ScriptedBeaconController(), episodes=10, start_seed=0)
        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.collision_rate, 0.0)
        self.assertEqual(result.truncation_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
