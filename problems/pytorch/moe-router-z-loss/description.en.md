# MoE Routing: z-loss Stability Regularizer

## Problem

As proposed by PaLM / ST-MoE: once the router logits explode (especially in bf16/fp16), the softmax saturates and gradients vanish. **z-loss** penalizes the square of $\log(\sum_e e^{x_e})$, pushing the absolute magnitude of the logits back toward the origin.

$$ \mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \big[\log(\sum_e e^{x_{t,e}})\big]^2 = \frac{1}{T}\sum_t \big[\mathrm{logsumexp}_e(x_{t,e})\big]^2 $$

## Rules

Implement `solve(logits)`, where `logits` has shape $(T, E)$: return the scalar $\mathrm{mean}(\mathrm{logsumexp}(\text{logits}, -1)^2)$.

## Examples

```
logits = [[1, 2, 0, -1],
          [0.5, 0.5, 0.5, 0.5],
          [3, 0, 0, 0]]

logsumexp per row: 2.440190, 1.886294, 3.139206
squares: 5.954525, 3.558106, 9.854615
z_loss = 6.455749
```

## Constraints

- CPU float32; do not multiply by any coefficient (return the raw mean).
