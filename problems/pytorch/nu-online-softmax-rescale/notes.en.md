# Notes: Why Two Scalars Suffice, and Why the Four Degenerate Cases Need Their Own Branches

## Origin

Streaming the denominator was spelled out first by Milakov and Girouard in "Online Normalizer Calculation for Softmax" (2018, NVIDIA): the softmax normalizer can be accumulated block by block at the cost of one extra rescale per block. FlashAttention (Dao et al., 2022) turned that into the per-step rescale inside KV blocking. The other equally real line is vocab-parallel training — in Megatron/DeepSpeed `vocab_parallel_cross_entropy` each rank only holds \(\text{vocab}/T\) logits, so the cross-entropy denominator must be fixed with one all-reduce that exchanges (max, sumexp); `shard_size` here is an abstraction of that rank count. Being able to write down "one exponential factor per segment is all we need" and name which rows must branch early says more about having handled sharded logits than reciting FlashAttention pseudocode.

## Why One Exponential Factor Is Enough

Suppose a segment already computed \(l_s=\sum_j\exp(x_j-m_s)\). The whole-row denominator wants \(\sum_j\exp(x_j-M)\), and the two differ by a constant:
$$\sum_j^{(s)}\exp(x_j-M)=\exp(m_s-M)\sum_j^{(s)}\exp(x_j-m_s)=\exp(m_s-M)\,l_s$$
Since \(m_s\le M\), the rescaling factor is always \(\le 1\): it only ever shrinks numbers, never overflows. The final `probs` pass uses \(\exp(x_j-M)\), again with a non-positive argument. Not a single `exp` on this path receives a positive argument, which is exactly why such kernels are numerically safe — not because we divide at the end, but because every step is measured against a maximum that actually exists in the data.

A useful corollary: `probs` are insensitive to the choice of reference. If the reference is wrong and equals \(M'\) instead of \(M\), numerator and denominator are both scaled by \(\exp(M'-M)\) and the ratio is unchanged; \(\texttt{lse}=M'+\log\sum\exp(x_j-M')\) is mathematically the same number too. So overflow-shaped bugs stay invisible in `probs` and only show up in `lse` — which is why the judge watches `lse` separately.

## Four Degenerate Cases Need Separate Branches

- **Rows containing NaN**: take segment maxima with the tensor `max()`, which propagates NaN (that is `Tensor.amax` semantics), so \(m_s\) becomes NaN and the row is routed out early.
- **Rows containing \(+\infty\)**: when \(M=+\infty\) every finite term is absorbed by \(\exp(\text{finite}-\infty)=0\), and all that is left is "split 1 among the \(+\infty\) entries", so counting them first and assigning beats letting IEEE arithmetic produce \(\infty-\infty\).
- **All-\(-\infty\) rows**: here \(m_s=M=-\infty\), so \(m_s-M\) is NaN. Such a row must branch out and emit "all-NaN probs, `lse` \(=-\infty\)" directly instead of splashing NaN through the rest of the row — that is the correct way to express "undefined".
- **One all-\(-\infty\) segment inside a row that has finite values**: \(\exp(-\infty-M)=0\) and that segment's \(l_s\) is itself \(0\), so the product is clean and the segment can simply be skipped.

## Core Ideas and Pitfalls

- **`torch.tensor(python float)` defaults to float32**: use it for the cross-segment reduction and a maximum like \(10^{39}\) — finite, but outside float32 — gets folded into \(+\infty\), the row is then routed down the \(+\infty\) branch and `lse` turns from \(10^{39}\) into `inf` (measured: `torch.logsumexp` returns 1e39 for a float64 input and inf for a float32 input). Do the cross-segment reduction in float64.
- **Python `max(list)` on NaN depends on order**: `max([float('nan'), 1.0])` and `max([1.0, float('nan')])` need not agree, so rows containing NaN must be diverted before any Python-level reduction.
- **Do not feed Python floats through `torch.exp`**: `torch.exp(torch.tensor(v))` falls back into the float32 trap; use `math` for pure scalars.
- **No gluing the row back together**: the judge cannot see what you do in memory, but `exp` of something like \(10^{300}\) becomes inf on the spot, and the tests assert explicitly that `probs` may not contain inf.
- **Cross-`shard_size` differences are last-digit only**: the cut changes summation order, giving roughly \(10^{-16}\) relative disagreement in float64, hence tolerances of 1e-9 (float64) and 1e-5 (float32) rather than bitwise equality.

## How It Differs from Neighboring Problems

`softmax` and `log-softmax` keep the whole row resident and subtract a maximum once. `logsumexp` wants only the denominator, not the probabilities. `reduce-axis` is a plain reduction with no cross-block calibration. `flash-attention-tile` uses exactly this rescale but then multiplies by V and emits attention, so its material is blocked GEMM and masking. This problem's unique angle is merging under a prescribed cut, plus the deliberate convention that \(+\infty\) rows split uniformly instead of following `torch.softmax` into NaN.

## Cross-Check Basis and Complexity

The oracle ignores blocking entirely: one whole-row `torch.softmax(src, 1)` / `torch.logsumexp(src, 1)` in float64, an independent path from the reference solution's "segment statistics plus Python-level merge". NaN is asserted with `isnan`, \(\pm\infty\) short-circuits on `==`, everything else goes through an absolute tolerance. Separate cases cover cut-invariance (every shard size from 1 to cols+2), extreme magnitudes (\(\pm900\), \(\pm1000\)), the four special row states, dtype/shape/empty handling, and invalid arguments. Complexity is \(O(rows\cdot cols)\), with only \(O(\lceil cols/\texttt{shard\_size}\rceil)\) scalars exchanged per row.
