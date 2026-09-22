# Planned module layout

The current 2D proof of concept remains in its existing modules while the project
migrates in small, testable steps. New production code belongs in the following
areas:

```text
brain/
  perception/   # visual, olfactory and inertial feature encoders
  navigation/   # heading, path integration and goal-direction state
  memory/       # mushroom-body-like associative memory
  control/      # motor decoding and action constraints
  policies/     # public Baseline and FlyBrain policy compositions
simulation/
  physics/      # bodies, actuators and disturbances
  sensors/      # onboard sensor implementations and noise models
  environments/ # RL task contracts, rewards and termination
  scenarios/    # procedural worlds and held-out evaluation cases
  rendering/    # optional diagnostics, never policy input
training/
  algorithms/   # PPO adapters and other optimization algorithms
  runners/      # train/evaluate entry points
  callbacks/    # checkpoints, metrics and telemetry
  evaluation/   # fixed, deterministic benchmark suite
configs/
  models/ environments/ training/ experiments/
tests/
  unit/ integration/ regression/
experiments/
  configs/ runs/
```

Dependencies flow in one direction: `simulation` provides observations;
`brain` turns them into actions; `training` orchestrates both. `experiments`
only selects configurations and records outputs. A policy must never receive
world coordinates, target distance, obstacle geometry, or renderer-only data.

Migration order: first introduce a common environment/policy contract, then move
the existing 2D classes into their target modules without changing behaviour.
Only after the regression tests are in place should new sensors, reward shaping,
or a PPO runner be added.
