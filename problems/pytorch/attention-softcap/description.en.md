# Attention Soft-Capping (Gemma-style logit softening)

## Problem

When attention logits are extremely large, $\text{softmax}$ collapses to nearly one-hot, gradients vanish, and training becomes unstable.
Gemma 2 introduced **soft-capping**: before the softmax, the logits are squeezed by a smooth `tanh` into the bounded
interval $(-cap, cap)$, which both limits extreme values and retains a differentiable soft threshold.

$$ \text{logits}' = cap \cdot \tanh\!\left(\frac{\text{logits}}{cap}\right),\qquad
p = \text{softmax}(\text{logits}') $$

## Rules

- Implement `solve(scores, cap)`, where `scores` has arbitrary shape and the softmax is taken over the **last dimension**; `cap` is a positive float.
- First compute $cap \cdot \tanh(scores/cap)$, then take the softmax over the last dimension and return the normalized probabilities (each row sums to 1).
- When `cap` is very large, $\tanh(x/cap)\approx x/cap$, so the result approaches a plain softmax over `scores`.

## Examples

```
Input: scores=[[1.0, 2.0, 3.0]], cap=1.0
     (capped = tanh([1,2,3]) = [0.761594, 0.964028, 0.995055])
Output: [[0.286751, 0.351092, 0.362156]]
```

## Constraints

- float32 input; last dimension length $\ge 1$; `cap > 0`
- `scores` may contain extremely large/small values (e.g. $\pm 10^6$); the implementation must ensure no NaN/Inf appears after `tanh`
