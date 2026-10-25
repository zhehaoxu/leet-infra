# Global Pooling (Global Avg/Max Pool)

## Problem

Global pooling compresses the entire feature map of each channel into a single number: GAP takes the mean and GMP takes the maximum, and the output is always $(N, C, 1, 1)$. It is the key to ResNet's final layer "arbitrary resolution in, fixed-length vector out", and is also commonly used for CAM interpretability.

$$
\text{GAP}[n,c] = \frac{1}{HW}\sum_{i,j} x[n,c,i,j],\qquad \text{GMP}[n,c] = \max_{i,j} x[n,c,i,j]
$$

## Rules

- Implement `solve(x: torch.Tensor, mode: str = "avg") -> torch.Tensor`, where x has shape `(N, C, *)` (`*` is any number of spatial/sequence dimensions).
- The reduced dimensions are all dimensions with `dim >= 2`, and **`keepdim=True`**—keeping `(N, C, 1, ...)` so that subsequent `flatten`/broadcasting semantics stay clear.
- `mode` is `"avg"` or `"max"`; any other value raises `ValueError`.
- Reduction does not cross channels: each $(n,c)$ is reduced independently.

## Examples

```
Input: x shape (1,2,2,2) = [[[1,2],[3,4]], [[5,6],[7,8]]]
mode="avg" → (1,2,1,1) = [[2.5], [6.5]]
mode="max" → (1,2,1,1) = [[4.0], [8.0]]
```

## Constraints

- float32/float64
- avg matches `F.adaptive_avg_pool2d(x, 1)`; max matches `F.adaptive_max_pool2d(x, 1)`
