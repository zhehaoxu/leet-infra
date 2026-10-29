# softmax Backward: Gradient w.r.t. logits from Output and Grad-Output

## Problem Description

softmax is the normalization operator most often passed through by backpropagation: after attention scores, after any "normalize then weight" step. Its Jacobian is the \(n\times n\) matrix \(\partial y_i/\partial x_j = y_i(\delta_{ij}-y_j)\), yet no real implementation builds it — one dot product along the normalized axis is enough:

$$dx_i = y_i\Big(grad_i-\sum_k y_k\,grad_k\Big)$$

Here you are only given the forward output \(y\) (softmax or log_softmax) and the upstream gradient \(grad\), and must recover \(dx\). This is also the only quantity FlashAttention's backward has to recompute.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Explicitly materializing the \(n\times n\) Jacobian and multiplying grad is forbidden; only one reduction along `dim` is allowed.
- `log_input=True` means the forward was log_softmax, and the formula becomes \(dx_i = grad_i - e^{y_i}\sum_k grad_k\); the two branches are **not interchangeable**.
- If grad is constant along `dim` (e.g. all ones), the softmax branch must return all zeros — that is exactly softmax's invariance to shifting the logits.
- Return a tensor with the same shape and dtype as \(y\); mismatched \(y\)/grad shapes, an out-of-range `dim`, and a non-bool `log_input` all raise ValueError.

## Example

Normalized along the last axis; numbers come from the EXAMPLE_* constants filled in from test_cases.py:

```
y    = [[0.5, 0.25, 0.25],   grad = [[1.0, 2.0, 3.0],
        [0.25, 0.5, 0.25]]           [0.0, 1.0, 0.0]]
sum(y*grad) per row = [1.75, 0.5]
dx   = [[-0.375, 0.0625, 0.3125],
        [-0.125, 0.25,   -0.125]]
```

Row 0 term by term: \(0.5(1-1.75)=-0.375\), \(0.25(2-1.75)=0.0625\), \(0.25(3-1.75)=0.3125\).

## Constraints

- Inputs may be float32 or float64; cross-checked against `torch.autograd.grad` with atol=1e-9 (float64) and 1e-6 (float32)
- Any `dim` (including middle axes and negative indices) plus 1-D inputs are supported; accumulate internally in float64 then cast back
