# Activation Memory and Gradient Checkpointing In Depth

## The four big blocks of training memory

The memory used by one training iteration can be broken down into:

1. **Parameters** (weights); 2. **Gradients** (grads, one per parameter of the same size);
3. **Optimizer state** (Adam's first/second moments, about 2× the parameters; mixed precision also keeps
   fp32 master weights);
4. **Activations** — the intermediate tensors computed in forward and needed in backward.

For small models the first three dominate, but Transformer activations grow with
`batch × seq × hidden × layers`; for long sequences the `s²` term (the attention probability matrix)
explodes, so activations are often the largest block and the one most amenable to "trading time for space".

## What gradient checkpointing does

Standard backward: every layer stores all of its intermediate activations in forward and reads them directly
in backward → activation memory $\propto$ `layers`.
Checkpointing: forward **stores only the layer inputs** (checkpoint boundaries) and discards the intermediate
activations; when backward reaches a layer, it **recomputes** that layer's forward from its input and then
does backward. So the resident part is only `layers × layer input`, and the peak adds one **extra layer's**
full activations:

$$ \underbrace{L\cdot b s h}_{\text{resident inputs}} + \underbrace{E_{\text{full}}}_{\text{currently recomputed layer}} $$

The cost is that the layer's forward is computed twice → roughly +30% compute for a several-fold memory
reduction (about 5× in this example).

## Why the attention term is $b a s^2$

$QK^\top$ produces an $s\times s$ attention score matrix per head, which becomes the probability matrix after
softmax; backward needs it, so $b\times a\times s\times s$ must be stored. This is a **quadratic** cost in
sequence length — the memory killer of long-context training, and exactly the problem FlashAttention (which
does not materialize $s\times s$ and uses tiled online softmax) was designed to solve (see the attention
family in this collection).

## Pitfalls and extensions

- This problem uses a **simplified teaching model**: the list is a deliberately prescribed set of typical
  terms, meant to train "order of magnitude + the idea of checkpointing". Different papers use different
  coefficients (e.g. Korthikanti et al. give a more refined per-layer activation formula), so do not use it
  to match profiler measurements.
- Checkpoint granularity is a choice: full (checkpoint every layer, most memory saved, most extra compute)
  vs. segmented (one boundary every N layers, a compromise).
- Real optimizations also include: ZeRO sharding of parameters/gradients/optimizer state, offloading to
  CPU/NVMe, and activation sharding via sequence/tensor parallelism.
