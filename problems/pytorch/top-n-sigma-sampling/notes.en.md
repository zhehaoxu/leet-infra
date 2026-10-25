# Solution Notes · Top-n-sigma Sampling

## Idea: let the threshold follow the shape of the distribution

top-k always keeps `k` entries and top-p always keeps mass `p`; neither looks at how **steep** the
distribution itself is. Top-n-sigma puts the cutoff line at "n standard deviations below the maximum":

$$\text{keep}_i = \big[\, z_i \ge z_{\max} - n\sigma \,\big],\qquad
\sigma = \sqrt{\tfrac1V\sum_i (z_i-\bar z)^2}$$

- If the distribution is very **peaked** (a few logits are high, the rest near $-\infty$) → $\sigma$ is large
  → the threshold drops far, but the high-scoring candidates are also far apart, so the low-scoring long
  tail is cut off by the $n\sigma$ line.
- If the distribution is very **flat** (logits cluster together) → $\sigma$ is small → the threshold sits close
  to max, leaving only the very top few.

This way there is no per-model hand tuning of `k`; a single `n` (llama.cpp defaults to 2.0) adapts.

## Three details for matching llama.cpp

1. **Population standard deviation**: `std = sqrt(acc / valid_count)`, with the element count `V` as the
   denominator, **not** `V-1` (`unbiased=False`). Using the sample variance instead shifts the whole
   threshold and changes the kept set at the boundary.
2. **Operate in logit space**: `mean/std/max` are all computed on the raw logits, and `-INF` slots should be
   excluded. The input here has no mask, so row-wise statistics over the whole row are enough. The
   truncation test is `logit < max - n*std → discard`; negating it gives the kept set.
3. **Softmax first, then filter by logit threshold**: softmax is monotonic, so keeping `logit>=cut` is
   equivalent to keeping those positions and renormalizing, which gives the same result as setting the
   discarded logits to $-\infty$ and applying softmax.

## Sampling

Normalize the kept set to obtain `filtered` (original order, sums to 1), then inverse-transform on the CDF
of `filtered` in its original order (first position $\ge u$) → `chosen`. Because at least the argmax is kept
(it is always $\ge$ max $-n\sigma$), the denominator of `filtered` is always positive and `chosen` always
lies in the kept set.
