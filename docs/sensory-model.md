# FlyBrain — Sensory Model

## 1. Design rule

The sensory model must expose only information that could plausibly be obtained from onboard sensors. Ground-truth simulation variables are reserved for logging and evaluation, never for policy input.

## 2. Vision

### 2.1 Panoramic receptor layout

Initial version: 72 rays distributed over a forward-biased hemisphere.

Each receptor stores a temporal state:

```text
previous_distance
current_distance
previous_direction
current_direction
```

From these, derive low-dimensional motion descriptors.

### 2.2 Optical-flow abstraction

Raycasts alone are range sensing. To obtain a flow-like signal, compare temporal changes in the angular position or range of scene intersections.

For a receptor `i`:

```text
flow_i = angular_change_i / dt
closing_i = (d_prev - d_now) / dt
```

Use a bounded representation for learning stability.

### 2.3 Looming

A useful collision-related visual feature is a time-to-contact approximation:

```text
tau ~= distance / closing_speed
```

This is an engineering approximation. Low `tau` means rapidly approaching visual structure.

Do not expose exact collision time from the simulator as a privileged label; calculate it from the same noisy quantities available to the policy.

### 2.4 Visual observation vector

Example:

```text
72 x [motion, closing, looming]
+ coarse left/right/global motion summaries
```

Start with normalization to `[-1, 1]`.

## 3. Olfaction

Represent odor as a spatial concentration field rather than a source-distance signal.

Minimum observation:

```text
odor_left
odor_right
Delta_odor_left
Delta_odor_right
pulse_left
pulse_right
wind_x
wind_y
wind_z
```

The field should support:

- source emission;
- wind advection;
- diffusion;
- temporal intermittency;
- random perturbations.

The agent must infer source direction from local measurements.

## 4. Haltere / inertial signal

Start from Rigidbody angular velocity and convert it into the agent frame:

```csharp
Vector3 omegaBody =
    transform.InverseTransformDirection(rb.angularVelocity);
```

Then apply:

- low-pass filtering;
- measurement noise;
- optional latency;
- saturation.

This creates a sensor signal rather than handing the controller the raw physics state.

## 5. Sensor noise model

Domain randomization parameters should include:

```text
visual noise          ± configurable
odor noise            ± configurable
IMU bias              random
IMU white noise       random
sensor latency        random
missing observations  occasional
wind                  random
```

The ranges must be experimentally measured or documented assumptions. Do not tune them only until the benchmark works; otherwise the benchmark can become overfit.

## 6. Sensor ablations

Every experiment should support:

```text
vision only
vision + inertial
vision + odor
vision + odor + inertial
```

This identifies which sensory channels actually contribute to performance.

## 7. References

- Borst, A. *Fly visual course control: behaviour, algorithms and circuits*. Nature Reviews Neuroscience (2014). https://www.nature.com/articles/nrn3799
- *Optic flow based spatial vision in insects* (review). https://pmc.ncbi.nlm.nih.gov/articles/PMC10354154/
