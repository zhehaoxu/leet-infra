# Reference Approach: Noisy Selection, Softmax over Selected Entries

## Implementation

1. Compute gate logits `x @ gate_w.T`.
2. In training, compute noise scales `softplus(x @ noise_w.T)`, multiply by the supplied `noise` samples, and add them to the logits; skip both steps in inference.
3. Take a descending, stable top-k over the noisy logits, keeping the lower expert index on exact ties.
4. Apply softmax only to the k selected noisy logits.
5. Scatter the weights into an all-zero `[T,E]` matrix, with exactly zero at unselected positions.

`k` must be a true `int`; `bool` is a subclass of `int` and cannot be accepted implicitly. Compute `softplus` as `logaddexp(0,z)` so negative inputs still yield a positive scale; do not substitute an absolute value.

## Correctness checks

The 15 `check_*` functions cover a hand example, zero-noise reduction, noise changing selection, positive and negative softplus inputs, softmax only over selected entries, `k=E` reducing to all-expert softmax, lower-index tie breaks, inference ignoring noise, training using noise, negative scales, noncontiguous inputs, zero rows, 30 randomized scalar-oracle cases, invalid k, and the maximum shape.

The oracle computes logits, softplus, and sorting per token and expert in pure Python, then performs an independent selected-entry softmax; it does not reuse `argsort` or `scatter`. The `k=E` and inference cases distinguish "noise affects only top-k selection" from "noise contaminates final weights".

## Scope and sources

The mechanism comes from noisy top-k gating in Shazeer et al., 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer": $H(x)_i=(xW_g)_i+\mathrm{Softplus}((xW_{noise})_i)\cdot\epsilon_i$, followed by top-k and softmax.

This problem supplies noise samples explicitly and uses no global random generator, so it is fully reproducible. The statement fixes tie order and inference behavior and does not claim equivalence to any framework's random stream, initialization, or complete MoE layer. It includes no load-balancing loss, capacity constraint, or expert forward pass.
