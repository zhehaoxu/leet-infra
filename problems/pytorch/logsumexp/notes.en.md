# Explained: logsumexp (Max-Subtraction and Degenerate ±inf Segments)

## Origin

"Subtract the maximum before exponentiating" is the textbook log-sum-exp trick and `scipy.special.logsumexp` exists as a standalone primitive; PyTorch inlines the same idea inside `F.log_softmax` and `F.binary_cross_entropy_with_logits` (the latter using the sister form `log1p(exp(-|x|))`). FlashAttention's tile accumulation keeps the LSE ($\log\sum e$) explicitly as the running softmax denominator, so any "recover the gradient from O/LSE/dO" problem (the backward side of `flash-attention-tile`) requires its LSE to agree with a full-segment logsumexp — this problem is that primitive itself.

## The Stable Form and the Non-Finite Branches

Let $m=\max_i x_i$ inside one segment. The identity

$$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$$

holds provided $m$ is finite. $x_i - m \le 0$ keeps $e^{x_i-m}\in(0,1]$, the sum lands in $[1, n]$ and its log in $[0,\log n]$, so the segment neither overflows nor underflows. The three non-finite cases each need their own value:

| segment content | $m$ | naive stable form | correct result |
| --- | --- | --- | --- |
| contains $+\infty$ | $+\infty$ | $+\infty - (+\infty) = \mathrm{NaN}$ | $+\infty$ |
| all $-\infty$ | $-\infty$ | $-\infty - (-\infty) = \mathrm{NaN}$ | $-\infty$ |
| contains NaN | NaN | NaN | NaN (propagates) |

The reference collapses this into three `where` calls with no per-segment Python branch:

```
m  = xd.amax(dim=axes, keepdim=True)
fm = torch.isfinite(m)                       # only finite m takes the stable path
sm = torch.where(fm, m, torch.zeros_like(m)) # placeholder 0 avoids inf - inf
s  = (xd - sm).exp().sum(dim=axes, keepdim=True)
out = torch.where(fm, sm + s.log(), m)       # fall back to m itself (±inf or NaN)
```

The last step is the crux: `m` *is* the correct answer in all three degenerate cases ($+\infty$, $-\infty$, NaN), so no extra branch per case is needed. Using 0 as the placeholder is deliberate too — it keeps `xd - sm` finite or $\pm\infty$ in those segments, whose `s` values get discarded wholesale anyway.

**Why underflow also matters**: for $x\sim-10^3$, $e^{x}$ is 0 in both float32 and float64, so the naive form returns $\log 0 = -\infty$; the stable form lifts the largest term to $e^0=1$, and even if the remaining terms still underflow to 0 only their negligible contribution is lost while the leading $m$ survives. The judge checks the segments $[1000,1002,999]$ and $[-1000,-1002,-999]$ together, first asserting that the naive form really is non-finite there (so an implementation cannot "copy the right formula without it mattering").

## Core Points and Pitfalls

1. **`dim=None` is an extension**: `torch.logsumexp` has no such spelling, so the judge uses `logsumexp(x.reshape(-1), 0)` as the reference.
2. **`keepdim=True` with `dim=None`** must keep every axis as 1 (giving `(1,1)`); in the implementation that is just `axes = tuple(range(x.dim()))` followed by the same `squeeze` decision.
3. **Sort and dedupe the axes before `.squeeze(dim=axes)`**: otherwise `squeeze` either errors or collapses the wrong axis.
4. **Keep intermediates in float64**: `exp` and `log` amplify relative error in float32; for a large $m$ (the $10^3$ segment) only a 1e-4 tolerance is available, so a pure float32 path lands on the boundary.
5. **Do not build `log_softmax` out of this problem's return value**: this interface only yields the scalar normaliser, while `log-softmax` returns $x - \text{LSE}$ per element; `check_log_softmax_consistency` cross-validates the two.

## How It Differs from Neighbouring Problems

- `log-softmax` / `softmax`: forward normalised outputs; this one only returns the denominator term.
- `cross-entropy-loss` / `bce-with-logits`: they consume logsumexp (or the equivalent stable softplus) internally and examine the loss and reduction conventions.
- `flash-attention-tile`: online-softmax LSE accumulation with block recomputation; this is a single one-shot reduction.
- `jsd-loss` / `kl-divergence`: divergence families whose inputs are already probabilities or log-probabilities.

## Cross-Checking and Complexity

- The oracle is `torch.logsumexp` (with the flattened form for `dim=None`), covering random float32, ×100 scaled, and 3-D inputs, both `keepdim` states, the ±inf/NaN degenerate segments, and five illegal `dim` values.
- Complexity $O(N)$: two reductions (`amax`, `sum`) plus one elementwise `exp`; the judging sizes run in milliseconds.
