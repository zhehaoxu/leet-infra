# In Detail: Jensen-Shannon Divergence Loss

## Origin
Lin, *Divergence Measures Based on the Shannon Entropy* (IEEE Trans. Inf. Theory 1991); Menéndez et al. 1997 introduced the general weighted form with \(\beta\). Define \(\mathrm{JSD}_\beta(P\|Q)=\beta\mathrm{KL}(P\|M)+(1-\beta)\mathrm{KL}(Q\|M)\), \(M=\beta P+(1-\beta)Q\). Properties: symmetric (\(\beta=0.5\)), non-negative, bounded above by \(\ln 2\). Modern uses include Hinton distillation and pseudo-label consistency (FixMatch/ReCoNet), among others.

## Core points and pitfalls
1. **Must go through the mixture distribution M**: JSD is not the average of two KL(P‖Q) terms, but the weighted sum of each distribution's KL against the **same** \(M\). Write it as \(\frac12\mathrm{KL}(P\|M)+\frac12\mathrm{KL}(Q\|M)\) (\(\beta=0.5\)).
2. **\(0\log 0=0\) and a safe denominator**: use a mask to accumulate only the terms with \(P_c>0\) (or \(Q_c>0\)). Since \(M\ge\beta P\), any \(P_c>0\) implies \(M_c>0\), so there is no division by zero.
3. **Upper bound \(\ln 2\)**: with disjoint supports (`[1,0]` vs `[0,1]`) it exactly reaches \(\ln 2\), used as a numerical anchor.
4. **Symmetry**: for \(\beta=0.5\), `solve(P,Q)==solve(Q,P)`; grading uses this for cross-validation.

## Matching convention
The reference solution's `_kl(A,B)` accumulates with a mask, and `solve=beta*_kl(P,M)+(1-beta)*_kl(Q,M)`, computed per row and then averaged over the N rows (reusing the same `_kl` as kl-divergence, with M passed in). The grader's `_oracle` constructs M element by element from P and Q in pure Python and then accumulates by definition; the structure differs but the values are equivalent. Assertions are also set for "disjoint = ln2 upper bound", "P==Q → 0", "symmetry", and "example values".

## Complexity
\(O(N C)\).
