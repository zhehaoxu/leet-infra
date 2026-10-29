# Cross-Entropy Backward: logits Gradient with Label Smoothing and ignore_index

## Problem Description

When training a language model, `F.cross_entropy` is the loss crossed at every step; its backward is a single line \(p - y\), but an industrial implementation must get three annoyances right at once: label smoothing spreads the one-hot into \((1-\varepsilon)\delta + \varepsilon/C\), `ignore_index` zeroes whole rows that must also be excluded from the `mean` denominator, and `reduction='sum'` demands no division at all. Given `logits (N, C)` and `target (N,)`, return a gradient with the same shape as the logits while satisfying all three conventions.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- For each valid row, \(dL/dz_k = p_k - (1-\varepsilon)\,onehot_y[k] - \varepsilon/C\) with \(p=\text{softmax}(logits)\); \(\varepsilon=0\) degenerates to \(p-y\).
- Rows equal to `ignore_index` are all zero **and** are excluded from the mean denominator, which counts valid rows, not N.
- `label_smoothing` must lie in \([0,1)\); `reduction` accepts only `'mean'`/`'sum'`; anything else raises ValueError.
- Subtract the row maximum before exponentiating; never produce `nan` in an ignored row by taking a log of an empty aggregation.

## Example

4 classes, 2 rows, `reduction='mean'`, \(\varepsilon=0\); numbers come from the EXAMPLE_* constants in test_cases.py:

```
logits = [[0, 0, 0, 0],      target = [1, 0]
          [0, 0, 0, 0]]      valid rows = 2
p      = [0.25, 0.25, 0.25, 0.25]
dL/dz  = [[ 0.125, -0.375,  0.125,  0.125],
          [-0.375,  0.125,  0.125,  0.125]]
```

Each row is \((0.25-\text{onehot})/2\): the hit class gives \(0.25-1=-0.75\), halved to \(-0.375\).

## Constraints

- logits are float32 or float64; cross-checked against `torch.autograd.grad(F.cross_entropy(...))` with atol=1e-9 in float64
- Accumulate in float64 internally and return the dtype of logits; when every target is `ignore_index`, return all zeros without a division-by-zero error
