"""Regression test for the end-to-end PPO train/checkpoint/evaluate path."""

import tempfile
import unittest
from pathlib import Path

from training.run_experiment import load_checkpoint, run_model


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

    def test_checkpoint_round_trip_restores_policy(self) -> None:
        config = {"seed": 5, "environment": {"batch_size": 1, "max_steps": 8}, "ppo": {"training_steps": 8, "rollout_steps": 8, "update_epochs": 1}, "evaluation_episodes": 1}
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            run_model("flybrain", config, output_dir)
            restored = load_checkpoint(output_dir / "flybrain.pt")
            self.assertEqual(restored.controller_name, "flybrain")
            self.assertTrue(all(parameter.isfinite().all() for parameter in restored.parameters()))


if __name__ == "__main__":
    unittest.main()
