# Explained: Maximal Marginal Relevance (MMR) Reranking

## Source
Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries* (SIGIR 1998). The paper defines \(P_1(d|q)\) as the similarity to the query and \(P_2(d|s)\) as the similarity to an already-selected document, with \(\lambda\) trading off relevance against diversity; \(\lambda=1\) is pure relevance and \(\lambda=0\) is pure diversity. LangChain's `MMRRetrieval` and sentence selection in text summarization both use it.

## Key points and pitfalls
1. **The penalty term is "the maximum similarity to the selected set"**: \(\max_{s\in S}\text{sim}(d,s)\), not a sum and not similarity to all documents. Redundancy is measured by the worst case.
2. **The first step has no selected set**: the \(\max\) over an empty set is treated as 0, so the score degenerates to \(\lambda\cdot\text{sim}(d,q)\) and the argmax is simply the most relevant document (for \(\lambda>0\)).
3. **sim is cosine**: \(\frac{a\cdot b}{\|a\|\|b\|}\), and a zero norm must be special-cased to 0 (otherwise 0/0=NaN poisons the argmax).
4. **Update \(S\) and remove from the candidates each round**: the selection is a greedy sequence, and a document already picked never comes back.
5. **Ties go to the smallest index**: floating-point equality needs a tolerance; iterating candidates in ascending order and updating only on strict `>` is equivalent.

## Comparison reference
The reference solution first normalizes docs/query, computes `rel=Du@q` and `sim=Du@Duᵀ`, then for \(k\) rounds takes the argmax of `lam*rel[i]-(1-lam)*max(sim[i,S])` over the unselected set (ascending order, tolerance, smallest index on ties). The judging `_oracle` uses a pairwise pure-Python cosine function (independently computing norms, division, returning 0.0 for zero norm) and recomputes every round's scores in a double loop — a different structure from the vectorized implementation, numerically equivalent. There are also assertions that "λ=1 equals relevance top-k", "with λ=0 the first pick is always index 0 and the rest are purely diverse", and "the output has length k with distinct indices".

## Complexity
\(O(nD + k\cdot n^2)\) (similarities can be precomputed; the \(k\) selection rounds each scan \(O(n\cdot|S|)\)).
