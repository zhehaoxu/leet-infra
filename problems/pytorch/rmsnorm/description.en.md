# RMSNorm

## Problem

RMSNorm is the standard normalization in LLMs (LLaMA, Mistral and the like): it drops LayerNorm's mean centering and bias and scales only by the **root mean square**:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon},\qquad
y_i = \frac{x_i}{\text{RMS}(x)}\,\gamma_i
$$

The paper argues that mean centering is not necessary: removing it saves one reduction per layer with virtually no loss in quality.

## Rules

- Implement `solve(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor`.
- Compute $\text{mean}(x^2)$ along the **last dimension** (`keepdim=True`), then `x * rsqrt(ms + eps) * weight`.
- **Do not** subtract the mean and **do not** add a bias parameter — that is its essential difference from LayerNorm, and getting it wrong shows up in the tests (an all-positive input must still be all positive after normalization).
- $\epsilon$ is added **inside** the square root, protecting all-zero rows from dividing by zero.

## Example

```
Input: x = [[1.0, 2.0, 3.0]], γ = [1,1,1], eps = 1e-6
ms = (1+4+9)/3 ≈ 4.666667, rms ≈ 2.1602469
Output: [[0.46291000, 0.92581999, 1.38873005]]
```

## Constraints

- float32/float64, shape `(..., d)`, `weight` of shape `(d,)`
- Invariant to an overall rescaling of the input: `solve(c*x, w) ≈ solve(x, w)` (c>0)
