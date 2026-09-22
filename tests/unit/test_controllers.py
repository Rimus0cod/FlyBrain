"""Unit tests for policy output contracts and recurrent controller state."""

import unittest

import torch

from brain import BaselineMLP, FlyBrainController


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.observation = torch.randn(3, 5)

    def test_baseline_returns_two_normalized_motor_commands(self) -> None:
        motors = BaselineMLP(observation_dim=5)(self.observation)

        self.assertEqual(tuple(motors.shape), (3, 2))
        self.assertTrue(torch.all((motors >= 0.0) & (motors <= 1.0)))

    def test_flybrain_returns_two_normalized_motor_commands_and_keeps_ring_state(self) -> None:
        controller = FlyBrainController(visual_dim=4)
        controller.reset_state(batch_size=3)

        motors = controller(self.observation)

        self.assertEqual(tuple(motors.shape), (3, 2))
        self.assertTrue(torch.all((motors >= 0.0) & (motors <= 1.0)))
        self.assertEqual(tuple(controller.central_complex.state.shape), (3, 16))
        torch.testing.assert_close(
            controller.central_complex.state.sum(dim=-1), torch.ones(3)
        )

    def test_flybrain_resets_state_when_batch_size_changes(self) -> None:
        controller = FlyBrainController(visual_dim=4)
        controller.reset_state(batch_size=1)

        controller(torch.randn(2, 5))

        self.assertEqual(tuple(controller.central_complex.state.shape), (2, 16))


if __name__ == "__main__":
    unittest.main()
