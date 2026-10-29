# Retrieval: ColBERT MaxSim and MaxPooled Aggregation

## Problem

ColBERT's late interaction keeps both the query and the document as **sequences of token vectors** and scores them with MaxSim: every query token finds its most similar document token, takes that cosine, and the per-query-token maxima are summed:

\[ S=\sum_{i=1}^{L_q}\max_{1\le j\le L_d}\cos(q_i,d_j). \]

When a long document is split into chunks for indexing, a tempting but wrong aggregation is "run MaxSim on every chunk and add the chunk scores" — the contribution of one query token gets counted once per chunk, so the score inflates with the number of chunks. Alongside the base MaxSim, this task splits the document into hard chunks of `chunk_size`, computes **the per-query-token maximum cosine inside each chunk** (a per-chunk MaxSim matrix), and aggregates it two ways: MaxPooled (each query token contributes exactly once) and repeated counting (sum over chunks), the latter returned as a judging contrast.

Normalization and degeneracies are the other half: query and document token vectors are L2-normalized row by row, a zero vector stays all-zero and has cosine 0 with everything, and empty documents, empty queries and short tail chunks all need deterministic results.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract.

- **Normalization**: both query and doc rows are divided by their L2 norm; a zero-norm row stays all-zero (use a safe denominator, never produce NaN). The cosine matrix is \(C_{ij}=\hat q_i\cdot\hat d_j\), where hatted rows are normalized.
- **Base score**: \(S=\sum_i \max_j C_{ij}\), a 0-dim floating tensor. With an empty document (\(L_d=0\)) every \(\max\) counts as 0, so \(S=0\); this is legal.
- **Argmax**: also return the index \(\arg\max_j C_{ij}\) per query token, with ties broken toward the **smallest index**; it is `-1` when \(L_d=0\).
- **Chunking**: hard split in token order, chunk \(c\) covering \([c\cdot\text{chunk\_size},\ \min((c+1)\cdot\text{chunk\_size},L_d))\). A short tail chunk is kept, so \(C=\lceil L_d/\text{chunk\_size}\rceil\), and \(C=0\) when \(L_d=0\). Chunks do not overlap.
- **Per-chunk matrix**: shape \((L_q,C)\); entry \((i,c)\) is the maximum cosine of query token \(i\) over chunk \(c\).
- **MaxPooled aggregation**: max inside each chunk, then max over chunks, then sum over query tokens. Since \(\max_c\max_{j\in c}C_{ij}=\max_j C_{ij}\), it is identically equal to the base score — that is exactly what "each query token contributes only once" means.
- **Repeated-count convention**: sum **all** entries of the per-chunk matrix (each query token contributes once per chunk) and return it as a contrast value. With a single chunk it equals the base score; with non-negative cosines it is never below the base score and inflates with more chunks (in the extreme, when every chunk contains the same best document token, it reaches \(C\) times the base score).
- **Empty inputs**: with \(L_q=0\) the base and both aggregate scores are 0, the per-chunk matrix has shape \((0,C)\) and the indices are empty; with \(L_d=0\) the matrix has shape \((L_q,0)\). Both are legal.
- **Error contract**: query or doc not 2-D, mismatched feature dimension, mismatched or non-floating dtype, and `chunk_size < 1` all raise `ValueError`.
- **dtypes**: the per-chunk matrix and the three scalars follow the input floating dtype (float32 / float64); indices are always int64. Fully deterministic, no RNG calls.

## Example

`query = [[1,0],[0,1]]`, `doc = [[1,0],[1,0],[0,1],[-1,0],[0.6,0.8]]`, `chunk_size = 2`. After normalization the cosine matrix is

\[
C=\begin{bmatrix}
1 & 1 & 0 & -1 & 0.6\\
0 & 0 & 1 & 0 & 0.8
\end{bmatrix}.
\]

The document forms 3 chunks: chunk 0 holds tokens 0 and 1, chunk 1 holds tokens 2 and 3, chunk 2 holds only the tail token 4. The per-chunk MaxSim matrix is

```
[[1.0, 0.0, 0.6],
 [0.0, 1.0, 0.8]]
```

- Base score: query token 0 takes \(\max(1,1,0,-1,0.6)=1\) and query token 1 takes \(\max(0,0,1,0,0.8)=1\), so \(S=2\); the argmax is `[0,2]` (the 1s for the first query token tie at tokens 0 and 1, and the convention picks 0).
- MaxPooled: the cross-chunk maxima stay 1 and 1, total 2, identical to the base score.
- Repeated counting: summing every matrix entry gives \(1+0+0.6+0+1+0.8=3.4\). Query token 0 is counted in chunk 0 and chunk 2 (\(1+0.6\)), query token 1 in chunk 1 and chunk 2 (\(1+0.8\)), 1.4 above the true score.

With a single chunk (for example when `chunk_size` is at least the document length) all three scores coincide; the 3.4 versus 2 above is the minimal discriminator between repeated counting and count-once.

## Constraints

- Return a five-tuple: base score, argmax indices, per-chunk MaxSim matrix, MaxPooled aggregate, repeated-count contrast.
- Indices are judged with `torch.equal`; floating values use tolerance \(10^{-5}\) for float32 and \(10^{-12}\) for float64.
- \(L_q,L_d\ge 0\), `chunk_size` \(\ge 1\); zero vectors have cosine 0 and must not produce NaN.
- CPU, deterministic, no RNG; any implementation path is fine (matmul, per-chunk loop, broadcast comparisons).
