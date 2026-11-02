# DFL Distributional Box Regression Ledger

Regressing a distance as a scalar treats annotation noise as ground truth. DFL regresses a
**distribution** instead: each side's distance is discretized into `reg_max+1` bins, the network
emits logits, decoding takes the expectation, and training supervises with "a weighted sum of the
cross-entropies at the two neighbouring bins" — the weights come from where the target sits
inside its bin.

## Interface

```python
def solve(logits, targets=None, stride=1.0, reg_max=7, weights=None) -> dict
```

`logits` has shape `(N, S, reg_max+1)`, where `S` is the number of sides (4 for l/t/r/b in
detection). The ledger is:

| Field | Meaning |
| --- | --- |
| `probs` | `(N, S, reg_max+1)`: softmax probability per bin |
| `dist` | `(N, S)`: expected distance `Σ i·p_i × stride` |
| `bin_index` | tensor of indices `0..reg_max` |
| `left` / `right` | left and right bin indices of the target |
| `w_left` / `w_right` | the two CE weights, always summing to 1 |
| `loss` / `loss_mean` | elementwise loss and its mean |

## Input / Output

```
Input: logits=all zeros (1 side, 8 bins), targets=[[3.25]]
Output: w_left=[[0.75]], w_right=[[0.25]], left=[[3]], right=[[4]],
      loss=[[log 8]], dist=[[3.5]]
```

Under a uniform distribution the expectation is the midpoint of `0..7`, namely `3.5`; the target
`3.25` lies between bin 3 and 4 with weights `0.75 / 0.25`, both CEs equal `log 8`, and the
convex combination is still `log 8`.

## Rules

- **Decoding is an index expectation**: `dist = Σ_{i=0}^{reg_max} i · softmax(logits)_i × stride`.
  Bin indices start at **0**, so the expectation tops out at `reg_max × stride`, not
  `(reg_max+1) × stride`.
- **The loss is a convex combination of two CEs**: `left = int(target)` (truncation toward zero),
  `right = left + 1`, `w_left = right - target`, `w_right = target - left`, and
  `loss = w_left · CE(left) + w_right · CE(right)`. It is not "interpolate the logits and run one
  CE", nor "run one CE against a soft label".
- **An integer target degenerates to a single CE**: for an integer `k`, `w_left = 1` and
  `w_right = 0`, so the right term drops out.
- **Targets live in `[0, reg_max-1]`**: `right` has to be indexable, so `target = reg_max` is
  illegal (the upstream implementation would go out of bounds) and negative targets are rejected.
- **The normalizer is the element count**: `loss_mean = loss.sum() / loss.numel()`, i.e. `N × S`,
  not the number of positives.
- **`weights`** multiply the loss elementwise, applied after the weighted CE.
- **Numerical stability**: logits of `1e4` must still yield a uniform distribution, so softmax
  has to subtract the maximum.
- **Validation**: `reg_max` is a positive integer (`bool` rejected), `logits` is 3-D with last
  dimension `reg_max+1`, `stride` is finite and positive, and `targets` / `weights` match `dist`.

## Hints

- `.long()` truncates toward zero, which equals floor only for positive targets; that is not
  `round`.
- With only `logits` given the call is pure decoding, and no `loss` keys should appear in the
  ledger.
