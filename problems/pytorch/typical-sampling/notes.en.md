# Solution Notes · Typical / Locally Typical Sampling

## The information-theoretic view

A token's **self-information** (surprisal) is $-\ln p$: the less likely it is, the more "information /
surprise" it carries when selected. The **entropy** of the distribution
$H=\mathbb{E}_{p}[-\ln p]=\sum_i -p_i\ln p_i$ is the "average surprise". The typical-set idea from
information theory says that tokens which truly carry information should have a surprisal **close to the
average $H$**.

- surprisal $\gg H$: extremely unlikely (long-tail noise) → drop.
- surprisal $\ll H$: almost certain (the model has already decided) → keeping it carries no "information",
  and dropping it actually makes the output livelier → drop.

Keeping the candidates with the smallest $|-\ln p_i - H|$ amounts to sampling among options that are
"neither outlandish nor boring".

## Truncation rule (matching llama.cpp `typical`)

1. Softmax gives `probs`.
2. `H = -(probs * log(probs)).sum()`.
3. `d_i = abs(-log(probs_i) - H)`.
4. Sort tokens by ascending `d` (ties by ascending original index, `argsort(..., stable=True)`).
5. Accumulate probability in that order, keeping tokens up to and including the one where the cumulative sum
   **first exceeds tau**; if nothing crosses it, keep everything.
6. Map the kept set back to original order and renormalize → `filtered`.
7. Inverse-transform sample (first position $\ge u$ on the original-order CDF of `filtered`) → `chosen`.

## Why the implementation needs care

- **The argmax may be discarded**: when several medium-probability tokens push $H$ noticeably above the
  argmax's surprisal, the argmax's deviation ends up larger than that of the medium-probability tokens, so it
  is not in the earliest-accumulated "typical" set (see the example in the problem statement).
- Tokens with `p=0`: `log(0) = -inf` → infinite surprisal → largest deviation → sorted last → truncated, and
  zeroing them during normalization has no effect; the code uses `clamp_min(1e-30)` to avoid `inf/nan`.
- The normalization divisor is `filtered.sum(-1, keepdim=True)`; since at least 1 token is always kept, the
  denominator is always > 0.

## Parameters

`tau` corresponds to llama.cpp's `typ_p` (a default close to 1.0 means essentially no truncation); smaller
values truncate harder.
