# In Detail: KL Divergence (Forward and Reverse)

## Origin
Kullback & Leibler, *On Information and Sufficiency* (Ann. Math. Statist. 1951). Defined as \(\mathrm{KL}(P\|Q)=\sum P\log(P/Q)\); asymmetric, non-negative, and satisfying Gibbs' inequality. Forward KL (\(\mathrm{KL}(P\|Q)\)) is central to MLE/variational lower bounds; reverse KL (\(\mathrm{KL}(Q\|P)\)) is used in distillation and VAE posterior approximation because of its mode-seeking behavior.

## Core points and pitfalls
1. **Asymmetry is the point of this problem**: both directions must be returned, and they are generally unequal. Grading asserts `forward_kl != reverse_kl` (on the same pair \(P,Q\)).
2. **The \(0\log 0=0\) convention**: terms with \(P_c=0\) are skipped outright; `log(0)` must not produce NaN/Inf. The reference solution uses the mask `P>0` to accumulate only valid terms.
3. **Sum per row, then average over the batch**: not a sum over the whole matrix, but one KL per row and then `mean` over \(N\).
4. **Return a tuple of scalars**: the judge supports tuple return values, both elements are 0-dimensional tensors, compared element-wise with allclose.

## Matching convention
Reference solution `_kl(P,Q)` accumulates with a mask and then `.sum(-1).mean()`; `solve` returns `(_kl(P,Q), _kl(Q,P))`. Grader `_oracle` uses a pure-Python triple loop computing `a*log(a/b)` element by element (only for \(a>0\)), accumulating per row and dividing by the number of rows; the structure differs but the values are equivalent. Assertions are also set for "\(P==Q\) → both directions ≈0", "asymmetry (fwd≠rev)", and "example values".

## Complexity
\(O(N C)\).
