# In Depth: Retrieval: IVF-PQ Asymmetric Distance Computation

## Source

- Method: Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545. The PQ paper introduces both symmetric distance computation (SDC, the query is encoded too) and asymmetric distance computation (ADC, the query stays floating point and only database codes are decoded); this task implements ADC.
- Engineering reference: the Faiss official documentation for `IndexIVFPQ` / `IndexIVF::search` (coarse quantizer + inverted lists + `nprobe`) and PQ LUT scoring. Faiss does not prescribe this task's padding, tie or dtype details; the starter docstring governs those.
- CPU, fully deterministic: cell selection only sorts distances, scoring only looks up and adds table entries, and no RNG is called.

## Full derivation and index convention

**Coarse selection.** Let \(C\in\mathbb R^{C\times D}\) be the centroid matrix and \(q\in\mathbb R^D\) a query. For every cell \(j\),
\[
s_j=\|q-c_j\|_2^2=\sum_{t=0}^{D-1}(q_t-c_{j,t})^2 .
\]
Sort cells by \((s_j,\ j)\) ascending and take the first \(n_{\text{probe}}\). The reference uses `torch.argsort(..., stable=True)`: a stable sort preserves the original index order of equal keys, and the original index *is* the cell id, so ties automatically go to the smaller cell. Measured: `torch.argsort(torch.tensor([2.,1.,1.,0.,1.]), stable=True)` returns `[3, 1, 2, 4, 0]`, i.e. the three 1s keep their relative order 1, 2, 4.

**LUT.** Subspace \(j\) covers feature range \([j d,(j+1)d)\), and its \(t\)-th codeword is \(\text{codebooks}[j,t]\in\mathbb R^d\). The query is not quantized and is sliced directly:
\[
\text{LUT}[j,t]=\bigl\|q_{j d:(j+1)d}-\text{codebooks}[j,t]\bigr\|_2^2
=\sum_{u=0}^{d-1}\bigl(q_{j d+u}-\text{codebooks}[j,t,u]\bigr)^2 .
\]
The LUT has shape \((m,K)\), one per query.

**ADC distance.** Document \(i\) has code row \(z_i=(z_{i,0},\dots,z_{i,m-1})\) with \(0\le z_{i,j}<K\). The PQ approximation concatenates per-subspace codewords, hence
\[
\hat d(q,x_i)=\sum_{j=0}^{m-1}\bigl\|q_{j d:(j+1)d}-\text{codebooks}[j,z_{i,j}]\bigr\|_2^2
=\sum_{j=0}^{m-1}\text{LUT}[j,z_{i,j}].
\]
This is a **sum of \(m\) terms**, not an average, and not "reconstruct the full vector and compute one Euclidean distance" (mathematically equal, but the latter repeats a \(D\)-dimensional subtraction, which is exactly what the LUT amortizes).

**Sorting.** The candidate key is the pair \((\hat d,\ \text{id})\). The reference first stably sorts by id, then stably sorts by distance; equal distances keep the preceding id-ascending order, yielding lexicographic order. torch has no direct lexsort, so two stable passes are the shortest route. With globally unique ids this is "distance ascending, id ascending".

**Padding.** Let \(N_q\) be the number of candidates in the selected cells for a query and \(t=\min(k,N_q)\). `ids` is always \((Q,k)\): the first \(t\) slots hold real ids and the rest \(-1\); `dists`' first \(t\) slots hold real distances and the rest \(+\infty\). With `k=0` the result is \((Q,0)\). Shapes therefore do not depend on candidate counts, and the empty case has one uniform representation.

**dtype.** Inputs accept float32/float64 only and all three dense tensors must match. Queries, centroids and codebooks are cast to float64 for squared differences and sums; distances are cast back to the query dtype, and `ids` is always int64.

**Complexity.** Cell selection \(O(QCD)\); LUT construction \(O(QmKd)=O(QKD)\); scanning candidates \(O(m)\) lookups/adds each, \(O(QNm)\) total. Space is \(O(QmK)\) for LUTs and \(O(Nm)\) for concatenated codes.

## Key points and common mistakes

1. **Never mix subspace codebooks**: the \(j\)-th query slice may only index `LUT[j]`, and `LUT[j]` may only be built from `codebooks[j]`. Treating the \(m\) codebooks as one big codebook, or always using codebook 0, is the classic shape-correct / value-wrong bug.
2. **ADC means the query is not quantized**: do not encode the query to a codeword and compare codeword-to-codeword — that is SDC. The LUT here is built from the raw, sliced query.
3. **Accumulate in float64**: measured on a random \(D=32,m=4,K=16,C=8\) case with coordinates scaled by 10 (distances 2370.5864–11989.4717), a pure float32 LUT + sum differs from float64 accumulation followed by a cast by at most \(9.76562\times10^{-4}\) absolute, mean \(1.91154\times10^{-4}\). That exceeds the usual \(10^{-5}\) float32 absolute tolerance, which is why the reference accumulates in float64.
4. **nprobe ties choose the smaller cell**: implemented by the stable sort. The test puts four centroids at distance 1 from the query, populates only cells 2/3, and requires `nprobe=2` to select cells 0/1 and return all padding, while `nprobe=4` returns `[20,30]`. Picking the larger cells on a tie fails immediately.
5. **top-k ties choose the smaller id**: in the example ids 1 and 2 both have ADC distance exactly 100, and the output must be `[0,1,2]`, not `[0,2,1]`. The two stable sorts guarantee this.
6. **Padding is not omission**: fewer than \(k\) candidates still returns \((Q,k)\), with `-1` / `+inf` tails; `k=0` is \((Q,0)\), not an error.
7. **Skip empty cells**: a cell with \(n_c=0\) may exist in the inverted table; filter it before `torch.cat` (an empty chunk is harmless, but treating it as a candidate corrupts counts).
8. **nprobe boundaries**: `nprobe=0` probes nothing; `nprobe>C` means \(C\). Measured: the example gives identical output for `nprobe=5` and `nprobe=2`.
9. **Global ids are unique by contract**: this task does not deduplicate; a repeated id would appear twice and the behaviour is undefined. Tests never build such data.
10. **C=0**: there are no centroids and no lists; `argsort` over a length-0 dimension returns empty and the result is all padding. Do not raise just because `min(nprobe,C)=0`.

## Differences from neighboring problems

- **vs `retrieval-ivf-assign`**: that task only assigns each vector to its nearest coarse centroid and returns a centroid index; this task selects \(n_{\text{probe}}\) cells for a *query* and then scores documents with PQ codes, with the LUT and top-k as the core.
- **vs `retrieval-pq-codes`**: that is **encoding** (raw vector in, per-subspace argmin code out, shape \((n,m)\) integers); here codes are inputs and the task only **decodes and scores**, with no training or encoding.
- **vs `retrieval-hnsw-greedy`**: that walks a graph hop by hop and candidates come from adjacency; this is an exhaustive scan over inverted lists, with no graph and no local optimum.
- **vs `retrieval-nprobe-merge` (same batch)**: that task stores full vectors and focuses on probe order, cross-cell deduplication taking the minimum distance, and candidate-count statistics, with no PQ; this task does not deduplicate and does not expose the cell sequence — its core is the per-subspace LUT sum and ADC top-k. The two share the same `nprobe` selection rule but have completely different downstream scoring.
- **vs `retrieval-kmeans-lloyd`**: that trains coarse centroids; here centroids and codebooks are given.

## Cross-check convention and complexity

- **Reference structure**: broadcast the LUT to \((Q,m,K)\) in float64, concatenate ids/codes of the selected cells per query, gather with `lut[qi][arange(m), codes]`, sum over \(m\), then two stable sorts and write the first \(t\) slots.
- **Independent oracle**: `_oracle` uses pure-Python nested loops (query × cell × document × subspace × dimension) accumulating squared differences, explicit `sorted(range(C), key=(dist, cell))` for cells and `list.sort(key=(dist, id))` for ranking. This is structurally different from the reference's "LUT gather + two stable sorts" and specifically catches subspace misalignment, wrong LUT indices and wrong tie order.
- **Cases**: the example (measured `EXAMPLE_*`, including the nprobe=1 padding branch); 5 random float32 draws × `nprobe∈{0,1,2,4,9}` × `k∈{0,1,3,10}`; 3 random float64 draws × `nprobe∈{0,1,3,4,7}` × `k∈{0,2,6}`; top-k truncation and huge-k padding; four equidistant codewords testing id tie-break; nprobe cell tie-break; k=0, nprobe=0, all lists empty, C=0, Q=0; 10 invalid-input groups.
- **Tolerances**: indices with `torch.equal`; float32 distances `atol=1e-5, rtol=1e-6`; float64 `atol=1e-9, rtol=1e-12`; padding must be exactly `inf`.
- **Measured log** (inline scripts under `.venv-eval/bin/python`, calling the reference directly):
  - example `nprobe=2` → `ids [[0,1,2]]`, `dists [[0.,100.,100.]]`; `nprobe=1` → `ids [[0,1,-1]]`, `dists [[0.,100.,inf]]`; `nprobe=0` → all `-1`/`inf`; `nprobe=5` equals `nprobe=2`.
  - `k=0` → `ids`/`dists` shape `(1,0)`; all lists empty → all padding.
  - float32 vs float64 accumulation: max absolute difference \(9.76562\times10^{-4}\) (distance range 2370.5864–11989.4717).
  - `torch.argsort(torch.tensor([2.,1.,1.,0.,1.]), stable=True)` → `[3,1,2,4,0]`.
- **Complexity**: \(O(QCD+QKD+QNm)\) time; \(O(QmK+Nm)\) temporary space.
