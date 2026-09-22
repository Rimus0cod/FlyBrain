# FlyBrain — Architecture

## 1. Purpose

FlyBrain is a research simulator and control architecture inspired by *Drosophila melanogaster*. The engineering objective is to test whether a small, recurrent, biologically constrained controller can perform robust GPS-independent navigation and obstacle avoidance with limited compute.

This document separates:

- **biological grounding** — mechanisms supported by experimental literature;
- **engineering abstraction** — simplified mechanisms used to make the simulator tractable;
- **research hypothesis** — mechanisms that still need to be tested experimentally.

The project is intended for civilian/autonomous-navigation research such as indoor flight, inspection, mapping, search for benign visual markers, and landing.

## 2. High-level architecture

```text
WORLD / PHYSICS
      |
      v
+-------------+
|   SENSORS   |
+-------------+
  |    |    |
  v    v    v
VISION ODOR HALTERES/IMU
  |    |    |
  +----+----+
       |
       v
+-------------------+
| SENSOR PROCESSING |
+-------------------+
       |
       v
+-------------------+
| CENTRAL COMPLEX   |
| heading / memory  |
+-------------------+
       |
       v
+-------------------+
| ASSOCIATIVE MEMORY|
| mushroom-body-like|
+-------------------+
       |
       v
+-------------------+
| MOTOR CIRCUIT     |
+-------------------+
       |
       v
  4 motor commands
       |
       v
    RIGIDBODY
```

The controller is deliberately modular so that each biological abstraction can be removed or replaced in an ablation experiment.

## 3. Core modules

### Visual system

Input: temporal visual-motion features from a panoramic, low-dimensional sensor array.

Output:

- local motion direction/magnitude;
- looming / time-to-contact features;
- coarse directional visual evidence.

A raw raycast distance is **not** treated as optical flow. Temporal changes must be computed between observations.

### Olfactory system

Input: local concentration samples and their temporal derivatives, optionally separated into left/right antenna-like sensors.

Output:

- concentration;
- concentration change;
- intermittency/pulse state;
- local wind estimate.

The agent is not given ground-truth distance to the source.

### Haltere / inertial system

Input: angular velocity and orientation-related signals.

Output: a filtered rotational state used for rapid stabilization and heading estimation.

### Central-complex-like navigation system

A compact recurrent circuit maintains an internal heading representation. A ring-attractor abstraction is used as an engineering model, rather than a claim that the simulated units are literal copies of fly neurons.

### Mushroom-body-like associative memory

A sparse, high-dimensional expansion followed by a compact output layer can implement stimulus-specific associations. Plasticity is controlled by a learning signal rather than unconstrained end-to-end changes everywhere.

### Motor circuit

The brain produces four normalized motor commands. A separate physics layer converts these commands into forces/torques.

## 4. Data flow contract

```text
Physics -> Sensors -> Observation -> Brain -> Motor command -> Physics
```

No module above the physics layer should receive:

- exact world coordinates of the target;
- exact distance to the target;
- privileged obstacle geometry;
- simulator-only labels that will not exist on the real platform.

## 5. First implementation target

Version 0.1 should contain only:

1. Rigidbody quadrotor model;
2. four motor force points;
3. 72 panoramic visual rays;
4. temporal motion/looming calculation;
5. inertial signal;
6. a small recurrent controller;
7. ML-Agents PPO training;
8. telemetry and deterministic evaluation episodes.

Odor and associative memory are added only after stable flight is achieved.

## 6. References

- Borst, A. *Fly visual course control: behaviour, algorithms and circuits*. Nature Reviews Neuroscience (2014). https://www.nature.com/articles/nrn3799
- Wolff, T. & Rubin, G. M. et al. Reviews of the Drosophila central complex and its circuitry. https://pmc.ncbi.nlm.nih.gov/articles/PMC6283239/
- Reviews of Drosophila mushroom-body associative memory. https://pmc.ncbi.nlm.nih.gov/articles/PMC11199953/
