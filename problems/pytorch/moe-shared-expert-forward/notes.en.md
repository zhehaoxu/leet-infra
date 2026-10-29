# Reference Approach: Direct Composition of Two Expert Paths

## Implementation

Shared and routed experts use the same SwiGLU map; only weight selection differs:

1. For each token, gather k sets of `[H,D]` gate/up projections and `[D,H]` down projections using `expert_ids`.
2. Sum routed outputs weighted by `routing_weights` without renormalization.
3. Evaluate the shared expert once with fixed weights and add it directly, with no gate.
4. Compute parameter counts structurally instead of summing tensors or inferring unspecified router parameters.

An `einsum` can process `[T,K,H,D]` at once, but only k experts enter the forward pass. The shared hidden width may differ from routed widths; do not assume they match.

## Correctness checks

There are 14 `check_*` functions: a hand-computed example, zero routing coefficients, zero shared down projection, no coefficient renormalization, selected-slot permutation, all experts, one expert, differing shared widths, perturbation of unselected experts, active parameters not being a batch union, noncontiguous inputs, zero input, 30 randomized scalar-oracle cases, and the maximum shape.

The oracle recomputes SwiGLU, weighted sums, and parameter counts element by element with pure Python scalars, without reusing the reference implementation's broadcast path. In particular, `active_params` counts each token's structural path, not the union of experts visited anywhere in the batch.

## Scope and sources

The mechanism corresponds to the public DeepSeek-V2/V3 design of an always-active shared expert combined with routed experts, and to shared-expert ideas in Qwen MoE models. This problem starts from **supplied** routing results and does not implement a router, train a model, compute load-balancing losses, or account for communication.

The shared path is an **ungated** SwiGLU, unlike sigmoid-gated shared experts in implementations such as Qwen2-MoE; that variant is out of scope. The parameter count covers only the expert matrices given in the problem and must not be conflated with full-model parameters or training memory.
