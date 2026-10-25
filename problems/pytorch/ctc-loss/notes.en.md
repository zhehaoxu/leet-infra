# In Depth: CTC Loss (Forward Algorithm)

## Origin
Graves, Fernandes, Gomez, Schmidhuber, *Connectionist Temporal Classification* (ICML 2006). Forward variable recursion: the expanded label sequence \(\ell'\) has length \(L'=2L+1\) (blanks inserted at both ends and in the gaps), \(\alpha_t(k)\) is the sum of path probabilities that reach time \(t\) and end at expanded position \(k\), and the loss is \(-\log(\alpha_T(L'-1)+\alpha_T(L'-2))\).

## Core points and pitfalls
1. **Expanded labels \(\ell'\)**: `label_at(k)` is blank at even indices and `labels[k//2]` at odd indices. This is the standard trick that "allows skipping blanks, allows merging repeats".
2. **Two transitions**: position \(k\) can be reached each step from \(k-1\) (emit the current symbol) or from \(k-2\) (skip blank / merge repeats); but \(k-2\) is only allowed when "the current symbol is not blank and differs from the symbol at \(k-2\)" — precisely what prevents misinterpreting `a-a` as two `a`s.
3. **Log-space accumulation**: all sums use a stable `logaddexp`, with `-inf` denoting unreachable states. Plain addition underflows on long sequences.
4. **Terminal merge**: at the end, both the last and second-to-last (blank) positions may be accepting states, and they must be merged with `logaddexp` (for \(L'=1\), i.e. empty labels, only the all-blank path exists).
5. **blank = V-1**: fixed by the problem statement, not passed in from outside.

## Verification approach
The reference solution is a pure forward DP (`_lae` for stable merging, `label_at` for expansion) returning the scalar `-total`. The judge's `_oracle` **brute-force enumerates** all \(V^T\) paths, uses `collapse` (merge consecutive repeats, delete blanks) to filter those equal to the target label, multiplies and sums their probabilities, and takes `-log` — a structure completely different from the DP, exponential but run only for very small \(T,V\), cross-validating the forward algorithm. Two example values, "labels=[0]" and "labels=[0,1]", are also provided as numerical anchors.

## Complexity
Forward DP \(O(T L)\) (each step traverses \(O(L)\) expanded positions); the cross-check oracle \(O(V^T\cdot T)\).
