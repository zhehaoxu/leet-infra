# Deep Dive: Unigram Optimal Subword Segmentation (Viterbi)

## Origin
Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates* (ACL 2018, arXiv:1804.07389). The Unigram LM assigns each subword a probability \(p(x)\), and the segmentation probability of a string is the product of the subword probabilities; encoding takes \(\arg\max_S \prod p = \arg\max_S \sum \log p\), solved with Viterbi/shortest-path DP in \(O(nL)\) (the deterministic decode of SentencePiece `Encode` is exactly this argmax; training estimates \(p\) with EM and uses the unigram probabilities for subword-regularization sampling).

## Core Ideas and Pitfalls
1. **Global optimum ≠ greedy longest**: greedy longest-match at each position can miss a better combination (e.g. splitting one high-frequency long word into two even-higher-frequency words); this problem wants the DP global maximum.
2. **DP recurrence**: `best[j]` = the maximum log probability of the prefix `text[:j]`; `best[j]=max over each vocab word p that can end at j: best[j-len(p)]+logp(p)` (only when `text[j-len:j]==p` and `best[j-len]` is reachable). Store a backpointer recording the best predecessor word.
3. **Reachability**: mark \(-\infty\) when a prefix cannot be segmented; `best[n]==-\infty` means no complete coverage exists → raise `ValueError`.
4. **Unique optimum**: the judge guarantees a unique optimal segmentation, so no tie-breaking rule is needed (a strict `>` update in the implementation suffices); the returned index sequence comes from backtracking.

## Cross-Checking Against the Oracle
The reference solution is a forward DP (`best[]` + `back[]`, backtracked into indices). The judge's `_oracle` **brute-force enumerates** all \(2^{n-1}\) gap-based segmentations, checking one by one whether every piece is in the vocabulary and computing \(\sum\log p\) to take the maximum — a structure completely different from the DP, yet with identical results because the optimum is unique; it runs only on tiny strings (exponential). Additional assertions cover "single-piece coverage", "pure single-character fallback", "uncoverable raises ValueError", and "example values".

## Complexity
DP \(O(n L m)\) (\(n\) string length, try all \(m\) vocab words at each position, compare length \(\le L\)); the cross-check oracle is \(O(2^n)\).
