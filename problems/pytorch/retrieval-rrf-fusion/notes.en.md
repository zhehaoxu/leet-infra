# Walkthrough: Reciprocal Rank Fusion (RRF)

## Source
Cormack, Clarke, Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009). Production implementations: the `rrf` retriever in Elasticsearch/OpenSearch and LangChain's `EnsembleRetriever` (the weighted default variant). The core formula is \(\text{score}(d)=\sum_{r}\frac{1}{k+r(d)}\), with \(k=60\) near-optimal and insensitive.

## Core and pitfalls
1. **Ranks, not raw scores**: the whole value of RRF is that it needs no calibration — never mix the BM25/dot-product scores of the individual paths in.
2. **1-based ranks plus the `0` sentinel**: the denominator is \(k+\text{rank}\) with rank starting at 1; `0` means absent and must be skipped (otherwise it contributes a phantom \(1/k\)).
3. **The smoothing constant \(k\)** is a given scalar argument (convention 60); do not turn it into a rank offset.
4. **Ties by ascending index**: `torch.argsort(-score, stable=True)` keeps the original column order (ascending indices), matching a hand-written sort on `-score` then index.
5. A document absent from every path has score 0 and ends up last.

## Comparison setup
The reference solution gets the score with `contrib = where(ranks>0, 1/(k+ranks), 0).sum(0)`, then `argsort(-score, stable=True)`. The judge's `_oracle` accumulates `1/(k+r)` with a pure Python double loop (added only when r>0) and orders with `sorted(range(N), key=lambda d:(-sc[d], d))` (different structure, numerically equivalent). Additional assertions: "a single path reproduces that path's ranking", "fully absent documents come last", "the output is a permutation of 0..N-1".

## Complexity
\(O(L\cdot N)\) for scoring plus \(O(N\log N)\) for sorting.
