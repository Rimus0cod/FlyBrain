"""Tests for multi-seed experiment aggregation."""

import unittest

from training.run_experiment import aggregate_results


class ExperimentStatisticsTests(unittest.TestCase):
    def test_aggregate_reports_sample_standard_deviation(self) -> None:
        metric_template = {
            "collision_rate": 0.0,
            "path_efficiency": 0.0,
            "successful_path_efficiency": 0.0,
            "mean_reward": 0.0,
            "control_stability": 0.0,
            "parameter_count": 10,
            "controller_parameter_count": 5,
            "inference_latency_ms": 1.0,
        }
        per_seed = {
            "1": {"baseline_mlp": {**metric_template, "success_rate": 0.2}},
            "2": {"baseline_mlp": {**metric_template, "success_rate": 0.6}},
        }
        aggregate = aggregate_results(per_seed)

        summary = aggregate["baseline_mlp"]["success_rate"]
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean"], 0.4)
        self.assertAlmostEqual(summary["std"], 2**-0.5 * 0.4)


if __name__ == "__main__":
    unittest.main()
