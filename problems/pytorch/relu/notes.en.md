# ReLU Explained

## Definition

$$ \operatorname{relu}(x)=\max(0,x) $$

Piecewise linear and monotonically non-decreasing, with output on the half-axis $[0,+\infty)$. It is the default activation of modern neural networks and almost a synonym for "no activation function".

## Why ReLU

- **Extremely cheap**: one comparison, no exp/division, cheap in both forward and backward.
- **Sparse activations**: the negative region outputs exactly 0, which brings sparsity and reduces interference between units.
- **Mitigates vanishing gradients**: the gradient is constantly 1 in the positive region, unlike sigmoid/tanh which squash gradients to 0 in the saturation region.

## Implementation and numerical notes

- The reference solution uses `torch.maximum(x, torch.zeros_like(x))`: vectorized, no Python loop. `torch.clamp_min(x, 0)` or `F.relu` are equivalent.
- **Do not write Python branching such as `x if x>0 else 0`** — the statement requires a vectorized implementation, and elementwise scalar conditionals are invalid on tensors.
- The subgradient is not unique at $x=0$; in practice it is handled as 0 (or the right derivative 1), which does not affect judging.

## Dead neurons and their remedies

The gradient in the negative region is constantly 0: if a neuron is negative for every sample, it will stop learning permanently (dying ReLU). Family remedies —

- Leaky-ReLU / ELU / SELU: give the negative region a nonzero response or slope (see the corresponding problems).
- A sensible learning rate and initialization avoid falling into the dead zone en masse.

## Related

ReLU6 = $\operatorname{clip}(x,0,6)$ (adds an upper bound, quantization-friendly); hardtanh and PReLU (learnable negative slope) are variants of it.
