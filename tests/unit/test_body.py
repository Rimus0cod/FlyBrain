"""Unit tests for the deterministic parts of the 2D body dynamics."""

import unittest

import torch

from simulation.body import Simple2DBody


class Simple2DBodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = Simple2DBody(batch_size=2, dt=0.05)
        self.body.angle.zero_()
        self.body.ang_vel.zero_()

    def test_zero_motor_commands_keep_body_at_rest(self) -> None:
        state = self.body.step(torch.zeros(2, 1), torch.zeros(2, 1))

        torch.testing.assert_close(state["pos"], torch.zeros(2, 2))
        torch.testing.assert_close(state["vel"], torch.zeros(2, 2))
        torch.testing.assert_close(state["ang_vel"], torch.zeros(2, 1))

    def test_equal_motor_commands_produce_forward_motion_without_turning(self) -> None:
        state = self.body.step(torch.ones(2, 1), torch.ones(2, 1))

        self.assertTrue(torch.all(state["pos"][:, 0] > 0))
        torch.testing.assert_close(state["pos"][:, 1], torch.zeros(2))
        torch.testing.assert_close(state["ang_vel"], torch.zeros(2, 1))
        torch.testing.assert_close(state["angle"], torch.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
