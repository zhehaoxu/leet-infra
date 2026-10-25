# Mean Squared Error Loss (MSE Loss)

## Problem Description

Mean Squared Error is the most commonly used loss for regression tasks: the mean of the squares of the differences between predictions and targets. It penalizes large errors (double the error, four times the loss) and is differentiable everywhere.

## Rules

- Implement `solve(pred, target, reduction="mean")`, first computing $(\text{pred}-\text{target})^2$ elementwise, then reducing according to `reduction`:
  - `"mean"`: the mean over all elements (default);
  - `"sum"`: the sum over all elements;
  - `"none"`: no reduction; return the elementwise loss with the same shape as `pred`.
- `pred` and `target` have the same shape (or are broadcastable). **Note**: MSE is $\frac1n\sum(p_i-t_i)^2$; there is **no** $\frac12$ coefficient in front.
- Raise `ValueError` for an unknown `reduction`.
- The result must match `F.mse_loss(pred, target, reduction=...)`.

## Example

```
Input: pred = [2.5, 0.0, 2.0, 8.0]
      target = [3.0, -1.0, 2.0, 7.0]
reduction="none" → [0.25, 1.0, 0.0, 1.0]
reduction="mean" → 0.5625
reduction="sum"  → 2.25
```

## Constraints

- Supports float32 / float64; when `reduction` is `"none"` the output dtype matches `pred`
- Scalar reductions (mean/sum) return a 0-dimensional tensor, consistent with the behaviour of `F.mse_loss`
