# FlyBrain — Reward Design

## 1. Principle

Reward should encode the desired behavior without leaking privileged simulator information or producing a shortcut strategy such as hovering forever.

## 2. Training stages

### Stage A — stabilization

Objective:

- maintain altitude;
- maintain attitude;
- minimize excessive oscillation.

### Stage B — controlled motion

Objective:

- follow a benign waypoint or heading;
- maintain stable forward motion.

### Stage C — obstacle avoidance

Objective:

- avoid collisions;
- preserve progress toward a benign navigation waypoint.

### Stage D — sensory navigation

Objective:

- use visual/inertial/odor-like signals;
- reach a non-harmful source marker.

## 3. Example reward

```text
R =
    + progress_toward_waypoint
    + stable_flight_bonus
    + successful_marker_detection
    + successful_episode_completion
    - collision_penalty
    - excessive_angular_rate
    - excessive_motor_effort
```

Exact coefficients are experimental parameters, not fixed truths.

## 3.1 Milestone 001 contract

The first 2D task logs and sums these explicit terms: progress toward the hidden
marker, angular-rate stability, motor-energy penalty, one-time completion bonus,
and arena-boundary collision penalty. Episodes terminate on success or boundary
collision and truncate at the configured step limit. Every compared policy uses
this same contract; coefficients and limits must be recorded in its config.

## 4. Preventing reward hacking

Watch for these shortcuts:

- hovering at the starting point;
- hugging a wall to avoid a collision;
- spinning while exploiting a heading reward;
- exploiting simulator boundaries;
- repeatedly triggering a detection reward.

Use termination conditions and one-time bonuses where appropriate.

## 5. Reward decomposition

Log every component independently:

```text
progress_reward
stability_reward
collision_penalty
energy_penalty
completion_reward
```

Never inspect only the total reward.

## 6. Success criteria

Do not define success as "the reward went up". The benchmark should include measurable task metrics:

- completion rate;
- collision rate;
- path length;
- time to completion;
- heading error;
- energy proxy;
- inference cost;
- robustness under sensor/physics perturbations.
