"""Unit tests for observations exposed by the first 2D sensor model."""

import unittest

import torch

from simulation.sensors import Simple2DSensors


class Simple2DSensorsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sensors = Simple2DSensors(max_range=10.0)
        self.state = {
            "pos": torch.tensor([[0.0, 0.0]]),
            "vel": torch.zeros(1, 2),
            "angle": torch.zeros(1, 1),
            "ang_vel": torch.zeros(1, 1),
        }

    def test_observation_encodes_relative_bearing_and_normalized_distance(self) -> None:
        observation = self.sensors.observe(self.state, torch.tensor([[3.0, 4.0]]))

        expected = torch.tensor([[0.8, 0.6, 0.5, 0.0, 0.0]])
        torch.testing.assert_close(observation, expected)

    def test_closing_feature_is_positive_when_target_distance_decreases(self) -> None:
        self.sensors.observe(self.state, torch.tensor([[5.0, 0.0]]))
        self.state["pos"] = torch.tensor([[1.0, 0.0]])

        observation = self.sensors.observe(self.state, torch.tensor([[5.0, 0.0]]))

        self.assertGreater(observation[0, 3].item(), 0.0)

    def test_reset_clears_temporal_closing_state(self) -> None:
        self.sensors.observe(self.state, torch.tensor([[5.0, 0.0]]))
        self.sensors.reset()

        observation = self.sensors.observe(self.state, torch.tensor([[4.0, 0.0]]))

        self.assertEqual(observation[0, 3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
