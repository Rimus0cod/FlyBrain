"""Unit tests for observations exposed by the first 2D sensor model."""

import unittest

import torch

from simulation.sensors import Simple2DSensors


class Simple2DSensorsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sensors = Simple2DSensors()
        self.state = {
            "pos": torch.tensor([[0.0, 0.0]]),
            "vel": torch.zeros(1, 2),
            "angle": torch.zeros(1, 1),
            "ang_vel": torch.zeros(1, 1),
        }

    def test_observation_is_a_coarse_visual_beacon_and_imu_signal(self) -> None:
        observation = self.sensors.observe(self.state, torch.tensor([[3.0, 4.0]]))

        self.assertEqual(tuple(observation.shape), (1, 9))
        self.assertTrue(torch.all((observation[:, :4] >= 0.0) & (observation[:, :4] <= 1.0)))
        self.assertTrue(torch.all(observation[:, 4:8] == 0.0))
        self.assertEqual(observation[0, 8].item(), 0.0)

    def test_temporal_signal_changes_when_beacon_bearing_changes(self) -> None:
        self.sensors.observe(self.state, torch.tensor([[1.0, 0.0]]))
        changed = dict(self.state)
        changed["angle"] = torch.tensor([[torch.pi / 2]])
        observation = self.sensors.observe(changed, torch.tensor([[1.0, 0.0]]))

        self.assertGreater(observation[0, 4:8].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
