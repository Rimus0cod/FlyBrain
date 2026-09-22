"""Run the reproducible Milestone 001 PPO comparison."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from simulation.environment import Navigation2DEnvironment
from training.evaluate import evaluate
from training.ppo import PPOConfig, PPOPolicy, PPOTrainer


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_model(name: str, config: dict, output_dir: Path) -> dict:
    seed = config["seed"]
    set_seed(seed)
    environment_config = config["environment"]
    environment = Navigation2DEnvironment(**environment_config)
    observation = environment.reset(seed=seed)
    policy = PPOPolicy(name, observation_dim=observation.shape[-1])
    ppo_config = PPOConfig(**config["ppo"])
    trainer = PPOTrainer(policy, ppo_config)
    state = policy.initial_state(torch.device("cpu"))
    completed_episodes = 0
    updates = []
    steps = 0
    while steps < ppo_config.training_steps:
        rollout, observation, state, completed = trainer.collect_rollout(environment, observation, state)
        updates.append(trainer.update(rollout))
        completed_episodes += completed
        steps += ppo_config.rollout_steps
    checkpoint_path = output_dir / f"{name}.pt"
    torch.save({"controller": name, "state_dict": policy.state_dict(), "seed": seed, "config": config}, checkpoint_path)
    evaluation_environment = Navigation2DEnvironment(**environment_config)
    metrics = evaluate(policy, evaluation_environment, config["evaluation_episodes"], seed + 10_000)
    metrics.update({"training_steps": steps, "completed_training_episodes": completed_episodes, "final_update": updates[-1]})
    (output_dir / f"{name}.metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/milestone_001_ppo.json"))
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = arguments.output_dir or Path(config["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    results = {name: run_model(name, config, output_dir) for name in config["controllers"]}
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "results": results}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
