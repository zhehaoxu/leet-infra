# Adaptive-p Sampling

## Problem

Truncation-based samplers (top-k/p) only "cut the tail"; **Adaptive-p** (llama.cpp's `adaptive-p`) does the opposite —
it **deletes no candidates at all** and instead **reshapes** the whole distribution: a bell-shaped function peaked at the "target probability $p^\star$"
assigns a new logit to every token, so whichever token's original probability is closest to $p^\star$ is pushed to the front of the distribution.

What is more, $p^\star$ is **adaptive**: it maintains an exponential moving average (EMA) of the "historical sampling hit probability". If recent draws have always been
high-probability tokens (meaning the current distribution is too sharp), it **raises** $p^\star$ and flattens the distribution to increase diversity; otherwise it **lowers** it.
Hence this problem carries two state variables, `weighted_sum` / `total_weight`, as explicit inputs and outputs.

> To keep judging **deterministic**, the uniform random number `u` is an explicit input; the EMA state is both input and output.

## Constants

`DISTRIBUTION_WIDTH = 0.3`, `PEAK_LOGIT_VALUE = 5.0`, `SHARPNESS = 10.0`,
`INV_WIDTH = 1 / 0.3`.

## Rules

- Implement `solve(logits, target, decay, weighted_sum, total_weight, u)`: `logits` has shape `(..., V)`,
  `target`/`decay`/`weighted_sum`/`total_weight` are scalars, and `u` lies in $[0,1)$.
- Steps (row by row):
  1. `probs = softmax(logits)` (these **original** probabilities are needed later for the EMA in step 6).
  2. $\text{adapted\_target} = \text{clamp}\big(\text{total\_weight}==0\,?\ \text{target} : 2\cdot\text{target} - \tfrac{\text{weighted\_sum}}{\text{total\_weight}},\ 0,\ 1\big)$.
  3. $\text{dist}_i = |p_i - \text{adapted\_target}| \cdot \text{INV\_WIDTH}$;
     $\text{new\_logit}_i = \text{PEAK} - \text{SHARPNESS}\cdot\dfrac{\text{dist}_i^2}{1+\text{dist}_i}$.
  4. `filtered = softmax(new_logit)` (sums to 1, no candidate removed).
  5. Use `u` to inverse-transform sample on the CDF of `filtered` (original order) to obtain `chosen`.
  6. Update the EMA with the **original** `probs[chosen]`:
     `weighted_sum_new = probs[chosen] + decay*weighted_sum`; `total_weight_new = 1 + decay*total_weight`.
- Return `(filtered, chosen, weighted_sum_new, total_weight_new)`.

## Examples

```
Input: logits = [2, 1, 0.5, -1, -2, -3], target = 0.5, decay = 0.9,
      weighted_sum = 0.0, total_weight = 0.0, u = 0.5
  probs(softmax) = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  total_weight==0 → adapted_target = target = 0.5
  dist = |probs-0.5|/0.3 = [0.3343, 0.9305, 1.2202, 1.5670, 1.6300, 1.6532]
  new_logit = 5 - 10*dist²/(1+dist) = [4.1624, 0.5147, -1.7060, -4.5660, -5.1024, -5.3009]
  filtered = softmax(new_logit) = [0.971617, 0.025311, 0.002747, 0.000157, 0.000092, 0.000075]
  first CDF entry already ≥ u(0.5) → chosen = 0
  EMA: weighted_sum_new = probs[0] + 0.9*0 = 0.600294; total_weight_new = 1 + 0.9*0 = 1.0
Output: filtered = [0.971617, 0.025311, 0.002747, 0.000157, 0.000092, 0.000075], chosen = 0,
      weighted_sum_new = 0.600294, total_weight_new = 1.0
```

## Constraints

- float32 input; `target ∈ [0,1]`, `decay ∈ [0,1]`; `total_weight ≥ 0`; `u ∈ [0,1)`
- Each row of `filtered` sums to 1 (nothing is **zeroed out**; all candidates remain); `chosen` is the position hit by inverse-transform sampling on `filtered`
- The EMA update must use the `chosen` entry of `softmax(logits)` (the original probs), **not** `filtered[chosen]`
- When `total_weight==0`, `adapted_target` degenerates to `target`
