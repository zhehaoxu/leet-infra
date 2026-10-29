# Solution Notes: Soft-Thresholding and One ISTA Step

## Origin
Soft-thresholding as the proximal operator of the L1 norm goes back to Donoho & Johnstone's shrinkage estimators (1994/1995); ISTA as an iterative thresholding method for linear inverse problems with a sparsity constraint was formalised by Daubechies, Defrise & De Mol (2004), with FISTA accelerating it (Beck & Teboulle, 2009, arXiv 0904.0426). Since Wanda, threshold/proximal views are also used to explain why pruned weights need fine-tuning.

## Key points and pitfalls
1. **Gradient orientation**: the gradient of \(\tfrac12\|Wx-y\|_2^2\) with respect to \(x\) is \(W^\top(Wx-y)\), not \(W(Wx-y)\); with \(W\) of shape \((m,n)\) only the left multiplication by \(W^\top\) returns an \((n,)\) vector.
2. **Gradient step before shrinkage**: the order is not interchangeable. Shrinking first and then taking a gradient step yields neither the proximal step nor a sparse point (the next gradient refills it).
3. **Threshold boundary**: \(|z|=\lambda\) maps to 0. `clamp(|z|-lam, min=0)` gives that naturally, while writing `z - sign(z)*lam` flips the **sign** whenever \(|z|<\lambda\) — the classic implementation bug.
4. **Definition of `support`**: indices where the *output* is non-zero — not the set that was shrunk, and never positions clamped to zero.
5. **dtype**: `support` must be an integer index tensor; `-0.0` counts as 0 (`== 0` holds), and the judge compares values.

## Verification
The reference computes `W @ x - y`, `W.T @ r`, then `sign(z)*clamp(|z|-lam, min=0)` and finally `nonzero(...).flatten()`. The judge's oracle re-sums everything scalar by scalar in pure Python and keeps the intermediate \(z\) for the property assertions. Independent checks: `lam=0` matches a hand-computed gradient step bit-for-bit; `lam` above every \(|z|\) gives all zeros with an empty support; every kept component loses exactly `lam` of magnitude while preserving sign; `support` equals the set \(\{|z|>\lambda\}\); \(|z|=\lambda\) is zeroed; `eta=0` performs shrinkage only.

## Complexity
\(O(mn)\) for one matrix–vector product and one transposed product; \(O(n+m)\) memory.
