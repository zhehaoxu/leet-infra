# Binary Cross-Entropy Loss (BCE Loss)

## Problem

The binary classification (or multi-label) loss: the input is a **probability** $p\in(0,1)$ that has already gone through sigmoid, and the label is $y\in\{0,1\}$. It measures the discrepancy between the prediction of a single Bernoulli distribution and the ground truth, and is standard equipment for multi-label classification and GAN discriminators.

## Rules

- Implement `solve(pred, target, reduction="mean")`, element-wise:
  $$ \mathcal L_i = -\big[\,y_i\log p_i + (1-y_i)\log(1-p_i)\,\big] $$
- `pred` is a probability (not logits), with the same shape as `target`; `target` is 0/1 (may be float).
- You must **clamp** $p$ (to the open interval) to avoid $\log 0=-\infty$ when $p=0/1$.
- `reduction` is `"mean"` / `"sum"` / `"none"`; an unknown value raises `ValueError`.
- For interior probabilities (away from 0/1) the result must match `F.binary_cross_entropy(pred, target, reduction=...)`.

## Examples

```
Input: pred = [0.9, 0.2, 0.75, 0.4]
      target = [1.0, 0.0, 1.0, 0.0]
reduction="none" → [0.10536, 0.22314, 0.28768, 0.51083]
reduction="mean" → 0.28175
(y=1 keeps only -log p, y=0 keeps only -log(1-p); the closer the prediction to the true label, the smaller the loss)
```

## Constraints

- `pred` supports float32 / float64; the output dtype matches `pred`
- Do not use `F.binary_cross_entropy`; write the formula by hand and clamp before the log
