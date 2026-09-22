# FlyBrain — Central Complex Model

## 1. Biological motivation

The Drosophila central complex is strongly associated with orientation and navigation. Experimental and review literature describes heading-related neuronal activity and models based on ring-attractor-like dynamics. These observations motivate a compact recurrent navigation module.

This project uses a **functional abstraction**, not a neuron-for-neuron reconstruction.

## 2. Initial model

Use 16 heading units arranged in a ring:

```text
N0 <-> N1 <-> N2 <-> ... <-> N15
^                         |
|_________________________|
```

Each unit has a preferred heading:

```text
heading_i = 360 deg * i / 16
```

Neighboring excitation and broader inhibition create a localized activity bump.

## 3. State update

A simple discrete model:

```text
h(t+1) = f(
    W_local h(t)
    + W_visual v(t)
    + W_imu i(t)
    + W_goal g(t)
    + bias
)
```

The function `f` may initially be `tanh` or another bounded nonlinearity.

## 4. Why recurrence matters

A feed-forward network must reconstruct heading entirely from the current observation. A recurrent ring provides a persistent internal state, allowing the model to maintain a course estimate through short sensory gaps.

## 5. Learning constraints

Two modes:

### Unconstrained baseline

All allowed weights are trainable.

### Topology-constrained model

A binary connectivity matrix defines which synapses exist:

```text
W_ij = 0          forbidden connection
W_ij = trainable  allowed connection
```

This allows a direct experiment on whether structural constraints improve sample efficiency, robustness, or compute cost.

In code, the trainable raw matrix is multiplied by a registered binary mask on
every forward pass. The effective forbidden synapses are therefore exactly zero,
including after optimizer steps. Heading activity is explicit episode runtime
state passed into and returned from the circuit; it is not a model parameter or
module buffer.

## 6. Evaluation

Measure:

- circular heading error;
- recovery time after perturbation;
- stability under visual dropout;
- stability under IMU noise;
- parameter count;
- inference latency.

## 7. References

- *Neuroarchitecture of the Drosophila central complex*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6283239/
- Campbell & Giocomo, *How a fly's neural compass adapts to an ever-changing world*. Nature (2019). https://www.nature.com/articles/d41586-019-03443-1
