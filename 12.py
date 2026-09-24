import torch

from simulation.environment import Navigation2DEnvironment
from training.ppo import PPOConfig, PPOPolicy, PPOTrainer

for controller in ("baseline_mlp", "flybrain"):
    print(f"\n{'=' * 60}")
    print(controller)
    print("=" * 60)

    torch.manual_seed(11)

    environment = Navigation2DEnvironment(
        batch_size=1,
        max_steps=120,
        device="cpu",
    )

    observation = environment.reset(seed=11)

    policy = PPOPolicy(
        controller,
        observation_dim=observation.shape[-1],
    )

    trainer = PPOTrainer(
        policy,
        PPOConfig(
            rollout_steps=128,
            update_epochs=4,
        ),
        episode_seed=11,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }

    rollout, _, _, _ = trainer.collect_rollout(
        environment,
        observation,
        policy.initial_state(torch.device("cpu")),
    )

    diagnostics = trainer.update(rollout)

    print("\nPPO diagnostics:")
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "gradient_norm",
        "mean_ratio",
        "raw_advantage_std",
        "positive_advantage_fraction",
    ):
        if key in diagnostics:
            print(f"  {key:30s}: {diagnostics[key]}")

    print("\nParameter changes:")
    total_delta = 0.0
    changed = 0

    for name, parameter in policy.named_parameters():
        delta = (parameter.detach() - before[name]).abs()
        mean_delta = delta.mean().item()
        max_delta = delta.max().item()
        total_delta += delta.sum().item()

        if max_delta > 0:
            changed += 1

        print(
            f"  {name:40s}"
            f" mean={mean_delta:.8e}"
            f" max={max_delta:.8e}"
        )

    print(f"\nChanged tensors: {changed}/{len(before)}")
    print(f"Total absolute parameter delta: {total_delta:.8e}")
