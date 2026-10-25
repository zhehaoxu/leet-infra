# Retrieval: Reciprocal Rank Fusion (RRF)

## Problem

Hybrid search usually has several retrieval paths — for instance one BM25 keyword path and one vector semantic path — and their score scales differ, so they cannot simply be added together. **Reciprocal Rank Fusion** (RRF, Cormack et al., SIGIR 2009; Elasticsearch/OpenSearch `rrf`) uses only the **ranks** and never the raw scores, so it needs no calibration by construction: a document's fused score is the sum of \(\frac{1}{k+\text{rank}}\) over the paths it appears in, with \(k\) a smoothing constant (the paper recommends \(k=60\)). Documents ranked near the top of several paths get the highest fused score. This problem asks you to implement RRF scoring and re-ranking.

## Rules

Implement `solve(ranks, k)`. Input `ranks` (int64, shape \((L, N)\)): `ranks[l, d]` is the **1-based rank** of document \(d\) in path \(l\); if path \(l\) did not retrieve document \(d\) the entry is `0`. For each document \(d\) compute the fused score \(\text{score}(d)=\sum_{l:\,r_{l,d}>0}\frac{1}{k + r_{l,d}}\). Return the document indices sorted by score in **descending** order (int64, shape \((N,)\)); ties are broken by **ascending** index. \((N,)\) is the number of columns of `ranks` (the full candidate document set).

## Example

\(L=2, N=3, k=60\):

```
ranks = [[1, 2, 0],     # path 0: doc0 rank 1, doc1 rank 2, doc2 not retrieved
         [2, 0, 1]]     # path 1: doc0 rank 2, doc1 not retrieved, doc2 rank 1
score(doc0)=1/61+1/62=0.032522
score(doc1)=1/62       =0.016129
score(doc2)=1/61       =0.016393
Descending: doc0 > doc2 > doc1
out = [0, 2, 1]   shape = (3,)
```

## Constraints

- CPU; `k` is a positive integer (default convention \(k=60\)).
- Ranks are 1-based; `0` means that path did not retrieve the document and contributes no score.
- The fused score uses ranks only and never uses (or invents) any raw retrieval score.
- Ties are broken by ascending document index to guarantee determinism.
