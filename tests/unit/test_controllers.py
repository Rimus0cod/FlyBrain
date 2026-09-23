"""Unit tests for policy output contracts and recurrent controller state."""

import unittest

import torch

from brain import BaselineMLP, CentralComplex, FlyBrainController


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.observation = torch.randn(3, 9)

    def test_baseline_returns_two_normalized_motor_commands(self) -> None:
        motors = BaselineMLP(observation_dim=9)(self.observation)

        self.assertEqual(tuple(motors.shape), (3, 2))
        self.assertTrue(torch.all((motors >= 0.0) & (motors <= 1.0)))

    def test_flybrain_returns_two_normalized_motor_commands_and_keeps_ring_state(self) -> None:
        controller = FlyBrainController(visual_dim=8)
        state = controller.initial_state(batch_size=3)
        motors, next_state = controller(self.observation, state)

        self.assertEqual(tuple(motors.shape), (3, 2))
        self.assertTrue(torch.all((motors >= 0.0) & (motors <= 1.0)))
        self.assertEqual(tuple(next_state.heading.shape), (3, 16))
        torch.testing.assert_close(
            next_state.heading.sum(dim=-1), torch.ones(3)
        )

    def test_flybrain_resets_state_when_batch_size_changes(self) -> None:
        controller = FlyBrainController(visual_dim=8)
        state = controller.initial_state(batch_size=1)
        _, next_state = controller(torch.randn(2, 9), None)

        self.assertEqual(tuple(state.heading.shape), (1, 16))
        self.assertEqual(tuple(next_state.heading.shape), (2, 16))

    def test_topology_mask_keeps_forbidden_synapses_at_zero_after_optimization(self) -> None:
        circuit = CentralComplex(sensory_input_dim=5, topology_constrained=True)
        optimizer = torch.optim.Adam(circuit.parameters(), lr=0.01)
        state = circuit.initial_state(batch_size=2)
        loss = circuit(torch.randn(2, 5), state).sum()
        loss.backward()
        optimizer.step()

        forbidden = circuit.connectivity_mask == 0
        torch.testing.assert_close(circuit.recurrent_weights[forbidden], torch.zeros_like(circuit.recurrent_weights[forbidden]))

    def test_runtime_state_is_not_registered_as_a_parameter_or_buffer(self) -> None:
        controller = FlyBrainController()
        parameter_names = {name for name, _ in controller.named_parameters()}
        buffer_names = {name for name, _ in controller.named_buffers()}

        self.assertNotIn("state", parameter_names | buffer_names)
        self.assertNotIn("heading", parameter_names | buffer_names)


if __name__ == "__main__":
    unittest.main()
