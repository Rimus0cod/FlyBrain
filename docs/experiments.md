# FlyBrain — Experimental Protocol

## 1. Research question

Can a compact, recurrent, biologically inspired architecture achieve robust autonomous navigation with less computation while maintaining or improving robustness relative to conventional neural baselines?

## 2. Baselines

Run all experiments against at least:

```text
A: MLP
B: recurrent MLP
C: fly-inspired modular network
D: fly-inspired + topology constraints
```

These are comparison conditions, not rankings. Report their measured values without converting them into a winner label.

## 3. Controlled variables

Keep identical across model comparisons:

- simulator version;
- physics parameters;
- task generator;
- observation budget;
- episode limits;
- training budget;
- random-seed protocol;
- evaluation environments.

## 4. Training split

Use separate environment seeds for training and evaluation.

Recommended structure:

```text
TRAIN
many procedurally generated arenas

VALIDATION
held-out seeds

TEST
held-out geometry + sensor noise + physics variation
```

Never tune the final model directly on the test set.

## 5. Perturbation matrix

Evaluate robustness with controlled changes:

| Perturbation | Levels |
|---|---|
| visual noise | low / medium / high |
| IMU noise | low / medium / high |
| sensor latency | 0 / low / high |
| motor mismatch | ±5% / ±10% / ±20% |
| mass mismatch | ±5% / ±10% |
| wind | none / low / medium |
| partial sensor dropout | 0% / 5% / 15% |

The exact ranges must later be tied to measured hardware limits.

## 6. Core metrics

```text
success_rate
collision_rate
mean_completion_time
path_efficiency
heading_error
control_energy_proxy
parameter_count
model_size
inference_latency
peak_RAM
CPU utilization
```

For stochastic training, report mean and spread across multiple independent seeds.

## 7. Ablation plan

Remove one capability at a time:

```text
full model
- central-complex recurrence
- visual motion features
- looming
- inertial input
- odor input
- associative memory
- topology constraints
```

This identifies which components produce measurable effects.

## 8. Sim2Real plan

Before hardware tests, use progressively stronger domain randomization:

```text
clean simulation
    -> sensor noise
    -> latency
    -> actuator mismatch
    -> wind/disturbance
    -> combined perturbations
```

Only after stable simulation benchmarks should inference be exported to an embedded target.

## 8.1 Phase 2 solvability diagnostic

Before comparing learned architectures, evaluate the same environment with three
controllers on held-out seeds:

```text
random   -> expected poor performance
scripted -> must establish that the task is physically solvable
trained  -> measured PPO result
```

The implementation is in `training/evaluate.py`; `run_experiment` stores the
three result groups under `diagnostics` in each controller metrics file. The
scripted controller uses only the public visual and angular-rate observation,
so a high scripted success rate is evidence of solvability rather than a
privileged-coordinate shortcut.

Each diagnostic also stores `progress_reward_by_step` and `distance_by_step`.
These trajectories are the primary check that a trained policy is approaching
the marker before termination, rather than merely changing its episode-average


PPO credit-assignment diagnostics also report the relationship between normalized
advantage and the two control degrees of freedom exposed by the body:

```text
forward_command = mean(motor_left, motor_right) - 0.5
turn_command = motor_right - motor_left
```

Reported values include advantage/progress correlations with forward command and
absolute turn magnitude, plus conditional means for positive- and negative-advantage
transitions. These are diagnostic only; they do not change the PPO objective.

## 9. Reproducibility

Every run must save:

```text
commit_hash
config
random_seed
training_steps
model_checkpoint
metrics.json
system_info
```

Experiment IDs should be immutable, e.g.:

```text
CC16-PPO-v003-seed07
MBSPARSE-v002-seed11
```

## 10. References

- Borst, A. *Fly visual course control: behaviour, algorithms and circuits*. https://www.nature.com/articles/nrn3799
- *Optic flow based spatial vision in insects*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10354154/
- *Neuroarchitecture of the Drosophila central complex*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6283239/
- *Sensory encoding and memory in the mushroom body: signals, noise, and variability*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11199953/
