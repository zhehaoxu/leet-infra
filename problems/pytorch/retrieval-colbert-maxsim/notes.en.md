# In Depth: ColBERT MaxSim and MaxPooled Aggregation

## Source

- MaxSim / late interaction: Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020, arXiv:2004.12832. Taking the maximum cosine per query token over document tokens and summing over query tokens is the scoring definition in Eq. (1) of the paper.
- Compression and indexing side: Santhanam, Khattab, Saad-Falcon, Potts, Zaharia, *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*, NAACL 2022, arXiv:2112.01488 (residual compression plus denoised supervision).
- The chunk-level "max inside a chunk, then max over chunks" MaxPooled aggregation is a long-document engineering convention for counting each query token once; it is not a separately named operator in the papers above. If a specific industrial implementation must be traced line by line, it is **to be sourced**. This task judges the mechanism defined in the starter docstring.
- Neighbors: `retrieval-mmr` reranks, `retrieval-rag-chunk` only emits a sliding-window index matrix, and `retrieval-pq-codes` compresses vectors into codeword indices; none of them performs MaxSim scoring.

## Full derivation and index convention

**Normalization**: for a row \(x\), take \(n=\sqrt{\sum_k x_k^2}\) and output \(x/\max(n,1)\), which leaves a zero row all-zero. The reference uses `torch.where(norm > 0, norm, 1)` as the safe denominator. A zero vector has cosine 0 with everything and never produces a `0/0` NaN. Normalization is **per row**, not the whole matrix divided by its Frobenius norm.

**Base score**: after normalization \(C=\hat Q\hat D^\top\in\mathbb R^{L_q\times L_d}\) and \(S=\sum_i\max_j C_{ij}\). With \(L_d=0\), \(C\) has shape \((L_q,0)\), every query-token max is defined as 0, so \(S=0\) and `base_idx` is all `-1`; with \(L_q=0\) the sum is empty and \(S=0\).

**Ties pick the smallest index**: `torch.max` returns values only, and `torch.argmax` picks the first occurrence of a repeated maximum; the reference first takes the row maximum `max_val`, then computes `(C == max_val).argmax(dim=1)`. The first `True` in the boolean matrix is necessarily the smallest index, independent of the backend's tie policy.

**Chunking and the tail**: the chunk count is \(C=\lceil L_d/\text{chunk\_size}\rceil\) and chunk \(c\) is the half-open interval \([c\cdot s,\min((c+1)s,L_d))\). A tail chunk may be shorter than \(s\) and must **not** be dropped; when `chunk_size` is at least \(L_d\) the result degenerates to \(C=1\). Chunks do not overlap (unlike the overlapping windows in `retrieval-rag-chunk`).

**MaxPooled is identically the base score**: for a fixed query token \(i\),
\[
\max_{c}\ \max_{j\in \text{chunk }c} C_{ij}
=\max_{j\in[0,L_d)} C_{ij},
\]
because the chunks partition the document indices. Max inside a chunk, then max over chunks, then sum over \(i\), is therefore **the same value** as a direct max over the whole document followed by the sum. In the measured example both are `2.0` and `torch.equal(maxpooled, base)` is `True`. So it is a wrong expectation that "MaxPooled differs from the base score" in this task; what actually differs is the repeated-count convention.

**Repeated counting**: `repeated = Σ_i Σ_c m_{i,c}` with \(m_{i,c}=\max_{j\in c}C_{ij}\). It adds the per-chunk maximum of every query token once per chunk. Measured:
- the example (2 query tokens, 3 chunks) gives `repeated = 3.4` and `maxpooled = base = 2.0`, a difference of 1.4, i.e. 1.7 times the base score;
- an extreme construction: one query token and four identical best document vectors. With `chunk_size=1` all four chunk maxima are 1, so `repeated = 4.0` and `base = 1.0`, exactly \(C\) times;
- the same construction with `chunk_size=4` (a single chunk) gives `repeated = base = 1.0`, the single-chunk degeneration.

With non-negative cosines `repeated ≥ base` always holds. Cosines can be negative, however: if every chunk maximum is negative, summing chunks can be **smaller** than the base score (two single-token chunks with maxima \(-1\) give `base=-1` and `repeated=-2`). That is why the starter says "generally larger", not "always larger".

**The zero-padding trap**: padding the document to \(C\cdot s\), reshaping and taking max puts cosine 0 at padded slots. When all real cosines inside a chunk are negative, 0 wrongly wins the max. Slice by the real chunk boundaries (as the reference does), or fill padded slots with `-inf` before the max.

**Empty inputs and dtype**: with \(L_d=0\) the per-chunk matrix has shape \((L_q,0)\) and all three scalars are 0; with \(L_q=0\) the matrix has shape \((0,C)\) and the indices are empty; both empty gives \((0,0)\). Every floating output follows the input dtype and `base_idx` is always int64. Judging uses \(10^{-5}\) for float32, \(10^{-12}\) for float64 and `torch.equal` for indices.

**Error surface**: non-2-D inputs, mismatched feature dimension, mismatched or non-floating dtype and `chunk_size<1` all raise `ValueError`; empty query/doc are not errors.

## Key points and common mistakes

1. **Normalize per row with a zero-safe denominator**: zero vectors must stay 0 and have cosine 0; a naive `x / x.norm()` yields NaN and poisons the max.
2. **Direction of the max**: each query token takes the max over **all document tokens**, then query tokens are summed. Taking max over query tokens first and summing over document tokens is a different (wrong) score.
3. **MaxPooled is not a new score**: max inside chunks and max across chunks equals the global max, so `maxpooled_score == base_score`. Do not "make it differ" by summing query tokens inside each chunk.
4. **Repeated counting is what inflates**: `repeated` sums every chunk's maximum, so it grows with the chunk count and must fall back to the base score for a single chunk.
5. **Keep the tail, round the chunk count up**: \(C=\lceil L_d/s\rceil\) and the last chunk may be short; dropping it changes both the matrix shape and `repeated`.
6. **Ties pick the smallest index**: `base_idx` must be stable; the first `True` of a boolean matrix or an ascending strict-`>` update both work.
7. **Do not treat padded 0 as a cosine**: with negative cosines, 0 steals the max; slice by boundaries or mask padded slots with `-inf`.
8. **An empty document is not an error**: \(L_d=0\) gives max 0, index `-1`, matrix \((L_q,0)\); \(L_q=0\) gives \((0,C)\).
9. **Keep the dtype**: float64 input must not silently fall back to float32 (judging uses \(10^{-12}\)); non-floating input raises `ValueError`.

## Differences from neighboring problems

- **vs `retrieval-mmr`**: MMR is **reranking** (greedily ordering a candidate set while penalizing redundancy with the selected set) over document-level vectors; this task is **scoring** (query-token × document-token late interaction) and produces no ordering.
- **vs `retrieval-rag-chunk`**: that task only emits the **token index matrix** of sliding-window chunks (`-1` padding); it has no vectors, mask or scores. The chunks here are **non-overlapping hard splits** that take part in cosine maxima and aggregation.
- **vs `retrieval-pq-codes`**: PQ compresses vectors into sub-codebook indices and outputs an integer \((n,m)\) code; this task does no quantization and keeps floating cosines plus scores and a matrix.
- **vs `cosine-similarity`**: that is a single cosine between one pair; here it is an \(L_q\times L_d\) cosine matrix with a double max and aggregation.
- **vs `retrieval-ivfpq-adc`** (if that task exists): ADC uses lookup tables to approximate distances and returns top-k; this is exact floating MaxSim and additionally exposes the "repeated vs count-once" convention gap.
- **vs `retrieval-bm25-score`**: BM25 is sparse bag-of-words scoring; this is late interaction over dense token vectors.

## Cross-check convention and complexity

- **Primary oracle**: `_oracle` recomputes cosines with a pure-Python double loop over query tokens and document tokens (independent norms, division, zero norm returning 0), takes chunk maxima with a `while` loop, then computes all three aggregates explicitly. It is structurally different from the reference's matmul plus per-chunk slicing. Coverage: the example, negative cosines, a zero query row, a zero document row, a tail chunk, a single chunk, empty query, empty document, both empty, float64.
- **Explicit discriminators**: tie-to-smallest-index (duplicate rows, all-negative, zero query), zero vectors without NaN, tail shape \((L_q,3)\), `repeated == base` for a single chunk, four single-token chunks giving `repeated = 2` while `base = 1`, empty-input shapes, a float64 comparison at \(10^{-12}\), and seven `ValueError` branches.
- **Complexity**: the cosine matrix is \(O(L_qL_dD)\) time and \(O(L_qL_d)\) memory; per-chunk maxima are \(O(L_qL_d)\); the output matrix is \(O(L_qC)\). The reference's chunk loop only slices and maxes and does not copy the document. Judging sizes are tiny (\(L_q,L_d\le 9\)), so tolerance is never the bottleneck.
