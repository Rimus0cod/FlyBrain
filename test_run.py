import json
import tempfile
from pathlib import Path

from training.run_experiment import run_model

config = {
    "seed": 7,
    "environment": {
        "batch_size": 1,
        "target_radius": 0.35,
        "arena_radius": 6.0,
        "max_steps": 120,
        "device": "cpu",
    },
    "ppo": {
        "training_steps": 4096,
        "rollout_steps": 128,
        "update_epochs": 4,
        "learning_rate": 0.0003,
    },
    "evaluation_episodes": 20,
    "diagnostic_episodes": 20,
}

with tempfile.TemporaryDirectory() as directory:
    for name in ("baseline_mlp", "flybrain"):
        m = run_model(name, config, Path(directory))
        d = m["diagnostics"]["trained"]
        u = m["final_update"]

        print("\n" + "=" * 60)
        print(name)

        print("\nEVAL")
        for k in (
            "success_rate",
            "progress_reward_mean",
            "positive_progress_fraction",
            "distance_start_mean",
            "distance_end_mean",
            "action_mean",
            "action_std",
        ):
            print(f"{k:45} {d.get(k)}")

        print("\nCREDIT ASSIGNMENT")
        for k in (
            "advantage_progress_correlation",
            "advantage_forward_command_correlation",
            "advantage_abs_turn_correlation",
            "progress_forward_command_correlation",
            "progress_abs_turn_correlation",
            "positive_advantage_forward_mean",
            "negative_advantage_forward_mean",
            "positive_advantage_abs_turn_mean",
            "negative_advantage_abs_turn_mean",
        ):
            print(f"{k:45} {u.get(k)}")

        print("\nLAGGED CREDIT")
        for lag in (0, 1, 2, 4, 8, 16):
            print(f"  lag={lag}")
            for k in (
                "advantage_forward_command_correlation",
                "advantage_abs_turn_correlation",
                "progress_forward_command_correlation",
                "progress_abs_turn_correlation",
            ):
                key = f"{k}_lag_{lag}"
                print(f"    {k:43} {u.get(key)}")
