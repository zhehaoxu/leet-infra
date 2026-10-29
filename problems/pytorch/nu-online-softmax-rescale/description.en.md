# Blocked Softmax Merging: Exact Results by Exchanging Only (max, sum)

## Problem Description

When a row of logits is too long and gets cut into consecutive segments — the KV blocks of FlashAttention, or the slice of the vocabulary each rank owns under vocab-parallel training — no single rank can see the whole row, yet the softmax denominator is a whole-row quantity. The fix is to let each segment hand over just two scalars, its maximum and its relative exponential sum, and have the merging side rescale them onto one shared reference maximum with a single exponential factor. This problem asks you to implement exactly that "per-segment statistics + cross-segment rescaling" for a given cut, returning both probabilities and log-sum-exp, while never taking `exp` of a quantity that can overflow.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- No `torch.softmax` / `torch.logsumexp` / `log_softmax`, and no gluing the row back together to compute it in one shot — that would leave no rescaling step at all.
- Each row is cut into consecutive segments of length `shard_size`, the last one may be short; if `shard_size >= cols` the whole row is one segment. For segment \(s\) of row \(r\):

```
m_s = max(seg)                    segment maximum
l_s = sum(exp(seg - m_s))         relative to m_s, so one term is exp(0) = 1
M   = max_s m_s
L   = sum_s l_s * exp(m_s - M)    the one rescaling factor per segment
probs[j] = exp(x[j] - M) / L
lse = M + log(L)
```

- The argument of every `exp` must stay \(\le 0\): subtract \(m_s\) inside a segment, multiply only by \(\exp(m_s-M)\le 1\) across segments, and finally compute only \(\exp(x_j-M)\le 1\). Any \(\exp(900)\) is wrong.
- Per-row special conventions: a row containing NaN → all-NaN `probs` and NaN `lse`; a row with \(k\ge1\) copies of \(+\infty\) → those \(k\) positions get \(1/k\) each, everything else \(0.0\), `lse` \(=+\infty\) (`torch.softmax` returns NaN on such a row; this problem deliberately chooses the uniform split because "don't blow up" is worth more in production); an all \(-\infty\) row → NaN `probs` and `lse` \(=-\infty\); a segment that is all \(-\infty\) inside a row that has finite values → that segment contributes \(0\); a maximum that is finite but beyond float32 range (say \(10^{39}\)) → still an ordinary row, `lse` must stay finite.
- Validation: \(x\) must be a 2-D float32/float64 tensor with `cols >= 1`, and `shard_size` must be an integer \(\ge1\) (`True`, `1.5`, `'2'`, `None` are all invalid), otherwise ValueError. `rows == 0` returns a correctly shaped empty result.

## Example

Write \(\ln 2=0.6931471805599453\) and \(\log 3=1.0986122886681098\), with `shard_size = 2`:

```
x = [[0.0, 0.0, 0.0],
     [0.0, ln2, -inf]]

probs = [[1/3, 1/3, 1/3],
         [1/3, 2/3, 0.0]]
lse   = [log3, log3]
```

In the first row the two segments contribute \(l_1=2\) and \(l_2=1\) with \(M=0\), so \(L=3\). In the second row \(M=\ln2\), the first segment gives \(l_1=e^{-\ln2}+e^{0}=1.5\) and the all-\(-\infty\) second segment is skipped, so the probabilities are \([0.5/1.5,\,1/1.5,\,0]=[1/3,2/3,0]\) and \(\texttt{lse}=\ln2+\log1.5=\log3\). Switching `shard_size` to 1 or 5 changes only the last digit.

## Constraints

- compute internally in float64 and cast the result back to the input dtype; `probs` has the shape of \(x\), `lse` has shape (rows,)
- the oracle is independent of any blocking scheme: one whole-row call to `torch.softmax` / `torch.logsumexp` (float64)
- across `shard_size` values only last-digit agreement is required: tolerance 1e-9 for float64 input, 1e-5 for float32, so do not expect bitwise reproduction
- \(\lceil cols/\texttt{shard\_size}\rceil\) segments per row, running time \(O(rows\cdot cols)\)
