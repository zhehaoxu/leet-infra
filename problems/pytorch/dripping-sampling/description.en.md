# Sampling: Dripping Ultra-High Probability Cut

## Problem

min-p cuts the **tail** below a relative fraction of the peak; its **head–tail dual** cuts the **head** instead: tokens whose probability is so high they carry no information (the clumped spike of the distribution) get dripped away, returning probability mass to the tail and forcing the sampler out of the spike into low-probability exploration. Dripping sets a **relative threshold** from the row's statistics and cuts the whole batch of tokens above it:

\[\text{mode="mean"}:\ \text{thr}=\alpha\cdot\bar p,\qquad
\text{mode="std"}:\ \text{thr}=\bar p+\alpha\cdot\sigma,\qquad
\text{cut}=\{i: p_i>\text{thr}\}\]

where \(\bar p\) and \(\sigma\) are the mean and **population** standard deviation of the softmax probabilities (computed over "valid" entries \(p_i>10^{-35}\) only, so underflow cannot pollute the statistics). The cut fires only if "**at least one token was actually cut** and **the survivors are \(\ge\)`min_keep`**" — otherwise the row is left untouched (better not to cut than to cut to nothing). The surviving set is renormalized into `filtered`, and the explicit uniform `u` inverse-transform-samples on the CDF in original order. CPU, float32, fully deterministic.

## Rules

- Factory `solve(logits, mode, drip_amount, min_keep, u)`; the full contract lives only in the starter comments and is the single judging standard.
- Procedure:
  1. `probs = softmax(logits)`;
  2. statistics over **valid entries** (`probs > 1e-35`) only: \(n\)=valid count, \(\bar p=\sum p_i/n\), \(\sigma=\sqrt{\sum p_i^2/n-\bar p^2}\) (**population** std — denominator \(n\), not \(n-1\));
  3. threshold by `mode`, one of: `"mean"` → \(\text{thr}=\alpha\bar p\); `"std"` → \(\text{thr}=\bar p+\alpha\sigma\); any other `mode` raises;
  4. **cut the head**: `cut = probs > thr` (**equal to the threshold survives**) — by softmax monotonicity the cut is exactly the highest-probability batch;
  5. **backstop**: if `cut` is empty (nothing to cut) **or** survivors \(<\)`min_keep` → the row is untouched, `filtered` equals the raw softmax, `cut_count=0`;
  6. renormalize the survivors into `filtered` (original order; cut positions strictly 0);
  7. `chosen`: inverse transform of `u` on the CDF of `filtered` in original order (first index whose prefix sum is \(\ge u\); fallback to the last index).
- Return `(filtered, chosen, cut_count)`: `cut_count` is an int64 tensor shaped like the batch dims of logits; **unfired rows record 0** (not "would-have-cut count blocked by the backstop").
- Everything is float32 on CPU and deterministic; ready-made samplers are forbidden.

## Example

`logits=[2.0, 1.5, 1.0, 0.5, 0.0, -0.5]` with `mode="std"`, `drip_amount=1.0`, `min_keep=1`, `u=0.55`.

softmax gives \(p=[0.414,\,0.251,\,0.152,\,0.092,\,0.056,\,0.034]\); all entries valid, \(\bar p=1/6=0.1667\), \(\sigma=\sqrt{0.2704/6-0.1667^2}=0.1315\). The `std` threshold is \(0.1667+0.1315=0.2982\) → only token 0 (0.414) exceeds it; cut 1, 5 survivors ≥ 1, fires. After renormalization `filtered=[0, 0.4287, 0.2600, 0.1577, 0.0956, 0.0580]`; the CDF first reaches \(\ge0.55\) at token 2 → `chosen=2`, `cut_count=1`.

Same row with `mode="mean"`: threshold \(=1.0\cdot0.1667=0.1667\) → both token 0 (0.414) and token 1 (0.251) get cut, `cut_count=2`.

```
std  mode:  filtered=[0.0, 0.4287, 0.2600, 0.1577, 0.0956, 0.0580]  chosen=2  cut_count=1
mean mode:  cut_count=2
```

## Constraints

- `mode` accepts only `"mean"`/`"std"`; `drip_amount\ge0`, `min_keep\ge1`; `logits` may carry batch dims (statistics and threshold per row)
- Each row of `filtered` sums to 1 with cut positions strictly 0; `chosen` must lie in the surviving set; floats compared with `atol=rtol=1e-5`, `chosen`/`cut_count` with `torch.equal`
- Four discriminating criteria the judge asserts directly: **perfectly flat** logits (all equal) → nothing is "above the mean" → neither mode fires and `filtered` equals the raw softmax; larger `drip_amount` raises the threshold and cuts **less** (monotone); when survivors fall below `min_keep` the row is untouched with `cut_count=0`; opposite to min-p, what gets zeroed is the **head** while the tail's mass survives
