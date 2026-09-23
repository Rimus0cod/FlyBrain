"""Regression test for the end-to-end PPO train/checkpoint/evaluate path."""

import tempfile
import unittest
from pathlib import Path

from training.run_experiment import load_checkpoint, run_model
from simulation.environment import Navigation2DEnvironment
from training.ppo import PPOConfig, PPOPolicy, PPOTrainer
import torch


class TrainingPipelineTests(unittest.TestCase):
    def test_training_writes_checkpoint_and_metrics_for_each_supported_controller(self) -> None:
        config = {
            "seed": 3,
            "environment": {"batch_size": 1, "max_steps": 12},
            "ppo": {"training_steps": 16, "rollout_steps": 8, "update_epochs": 1},
            "evaluation_episodes": 2,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for controller in ("baseline_mlp", "flybrain"):
                metrics = run_model(controller, config, output_dir)
                self.assertTrue((output_dir / f"{controller}.pt").exists())
                self.assertTrue((output_dir / f"{controller}.metrics.json").exists())
                self.assertEqual(metrics["training_steps"], 16)
                self.assertIn("success_rate", metrics)
                self.assertIn("inference_latency_ms", metrics)
                self.assertEqual(
                    set(metrics["diagnostics"]),
                    {"random", "scripted", "trained"},
                )
                self.assertGreaterEqual(metrics["diagnostics"]["scripted"]["success_rate"], 0.0)

    def test_checkpoint_round_trip_restores_policy(self) -> None:
        config = {"seed": 5, "environment": {"batch_size": 1, "max_steps": 8}, "ppo": {"training_steps": 8, "rollout_steps": 8, "update_epochs": 1}, "evaluation_episodes": 1}
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            run_model("flybrain", config, output_dir)
            restored = load_checkpoint(output_dir / "flybrain.pt")
            self.assertEqual(restored.controller_name, "flybrain")
            self.assertTrue(all(parameter.isfinite().all() for parameter in restored.parameters()))

    def test_ppo_update_changes_parameters_for_each_controller(self) -> None:
        for controller in ("baseline_mlp", "flybrain"):
            torch.manual_seed(11)
            environment = Navigation2DEnvironment(max_steps=10)
            observation = environment.reset(seed=11)
            policy = PPOPolicy(controller, observation_dim=observation.shape[-1])
            before = [parameter.detach().clone() for parameter in policy.parameters()]
            trainer = PPOTrainer(policy, PPOConfig(rollout_steps=8, update_epochs=1), episode_seed=11)
            rollout, _, _, _ = trainer.collect_rollout(environment, observation, policy.initial_state(torch.device("cpu")))
            diagnostics = trainer.update(rollout)
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, policy.parameters())))
            self.assertGreaterEqual(diagnostics["min_action"], 0.0)
            self.assertLessEqual(diagnostics["max_action"], 1.0)


if __name__ == "__main__":
    unittest.main()
