# Cross-Entropy Loss

## Problem

The core loss for multi-class classification: the model outputs unnormalized **logits** and the labels are integer class indices. Cross-entropy = the KL divergence between the true distribution (one-hot) and the predicted distribution + entropy, equivalently "the negative log probability of the true class" $-\log p_{\text{true}}$. For numerical stability, softmax and log must be merged into log-softmax.

## Rules

- Implement `solve(logits, target, reduction="mean")`:
  - `logits` has shape $(N, C)$ ($N$ samples, $C$ classes);
  - `target` has shape $(N,)$ and holds the true class index $[0,C)$ of each sample;
  - per-sample loss $= -\log\text{softmax}(\text{logits})[\,\text{target}\,]$.
- Must be **numerically stable**: subtract the row maximum first, then take $\exp$/$\log$; do not use `log(softmax(x))` directly.
- `reduction` is `"mean"` (average over the $N$ samples) / `"sum"` / `"none"`; an unknown value raises `ValueError`.
- The result must match `F.cross_entropy(logits, target, reduction=...)`.

## Examples

```
Input: logits = [[2.0, 1.0, 0.1],
                [0.1, 2.0, 1.0],
                [1.0, 0.5, 3.0]]
      target = [0, 1, 2]
reduction="none" → [0.41703, 0.41703, 0.19673]
reduction="mean" → 0.34360
(even if logits reach the thousands, the output stays finite: the effect of the stable formulation)
```

## Constraints

- `logits` supports float32 / float64; `target` holds integer indices
- Do not use `F.cross_entropy` / `F.nll_loss`; implement a stable log-softmax + gather by hand
