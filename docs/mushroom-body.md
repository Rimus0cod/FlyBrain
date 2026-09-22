# FlyBrain — Mushroom-Body-Like Associative Memory

## 1. Purpose

The mushroom-body-inspired component is an associative memory module. It is not assumed to be a literal reproduction of Drosophila cognition.

The biological literature supports a role for mushroom bodies in stimulus-specific associative memory, including olfactory learning. The exact engineering strategy below is a research model.

## 2. Proposed computational circuit

```text
sensory features
      |
      v
sparse expansion
      |
      v
Kenyon-like units
      |
      v
compact learned output
      |
      v
behavioral modulation
```

The important principles to test are:

- sparse coding;
- pattern separation;
- associative plasticity;
- limited online modification.

Recent reviews emphasize pattern separation, sparse coding, noise/variability, coincidence detection and localized neuromodulation as useful principles for modeling the Drosophila mushroom body. See: https://pmc.ncbi.nlm.nih.gov/articles/PMC11199953/

## 3. Engineering memory

Each benign visual marker or odor class is represented by an activation pattern. A small plastic readout stores the association.

Example abstraction:

```text
K = sparse expansion(sensory_input)
A = A + learning_rate * neuromodulator * K
output = A^T K
```

The exact plasticity rule is a research variable.

## 4. Learning protocol

Separate two modes:

### Offline learning

The model is trained across many episodes.

### Online adaptation

Only the designated memory/readout parameters can change during deployment episodes.

The controller itself remains frozen unless an experiment explicitly studies broader adaptation.

## 5. Few-shot claim

Do not label the system "1-shot" or "2-shot" until it is experimentally demonstrated. Instead measure:

```text
episodes_to_threshold
samples_to_threshold
retention_after_delay
cross-context generalization
```

This turns few-shot learning into a measurable hypothesis.

## 6. Ablations

Compare:

```text
no memory
ordinary MLP memory
sparse associative memory
mushroom-body-inspired memory
```

Use identical environments and training budgets.

## 7. References

- *Sensory encoding and memory in the mushroom body: signals, noise, and variability*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11199953/
- *Cellular and circuit mechanisms of olfactory associative learning in Drosophila*. https://pmc.ncbi.nlm.nih.gov/articles/PMC7147969/
