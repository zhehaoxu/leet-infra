# Swish / SiLU

## Problem

Implement the Swish activation (called SiLU, Sigmoid Linear Unit, in PyTorch), proposed by Google in 2017. It is a smooth replacement for ReLU, confirmed by NAS search as a better general-purpose activation than ReLU:

$$
\text{swish}(x) = x \cdot \sigma(x)
$$

where $\sigma$ is Sigmoid. Compared with ReLU it has a small dip in the negative region (minimum about $-0.278$ at $x \approx -1.28$) and is smooth and differentiable everywhere.

## Rules

- Implement `solve(x: torch.Tensor) -> torch.Tensor`, returning the elementwise $x \cdot \sigma(x)$.
- You may call `torch.sigmoid` as a primitive; the whole implementation must be vectorized.
- Output shape and dtype match the input.
- For large negative values (e.g. $x = -1000$) the result should approach 0 without any `NaN` (with a stable `sigmoid`, the product $x\cdot\sigma$ is naturally stable).

## Example

```
Input: [-3.0, -1.0, 0.0, 1.0, 3.0]
Output: [-0.14228, -0.26894, 0.0, 0.73106, 2.85772]
```

## Constraints

- Input is a float32 or float64 tensor of any shape
