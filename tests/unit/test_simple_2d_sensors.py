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

        self.assertEqual(tuple(observation.shape), (1, 5))
        self.assertTrue(torch.all((observation[:, :4] >= 0.0) & (observation[:, :4] <= 1.0)))
        self.assertEqual(observation[0, 4].item(), 0.0)

    def test_beacon_response_does_not_encode_distance(self) -> None:
        near = self.sensors.observe(self.state, torch.tensor([[1.0, 0.0]]))
        self.sensors.reset()
        far = self.sensors.observe(self.state, torch.tensor([[5.0, 0.0]]))

        torch.testing.assert_close(near, far)


if __name__ == "__main__":
    unittest.main()
