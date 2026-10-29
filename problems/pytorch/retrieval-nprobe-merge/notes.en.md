# In Depth: Retrieval: nprobe Multi-Probe and Cross-Cell Merge

## Source

- Structure: the coarse-quantizer + inverted-list design of Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545, and the Faiss official documentation for `IndexIVF` and the `nprobe` parameter (`IndexIVF::search` visits the first `nprobe` lists by coarse distance). Faiss's multi-probe is "coarse-rank the cells, then scan the selected lists"; this task isolates the ordering, deduplication and statistics of that flow.
- CPU, fully deterministic: cell selection only sorts distances, merging only uses a dict/sort and a minimum, with no RNG.

## Full derivation and index convention

**Cell score and order.** For query \(q\) and centroids \(c_j\),
\[
s_j=\|q-c_j\|_2^2,\qquad
\pi=\operatorname{argsort}\bigl(s_0,\dots,s_{C-1}\bigr)\ \text{(stable, i.e. ties keep the smaller cell id)}.
\]
The first \(P=\min(n_{\text{probe}},C)\) entries form the probe sequence and are scanned as \(\pi_0,\pi_1,\dots,\pi_{P-1}\). Stability matters: when two cells are equidistant, the original index order is cell-id order, so a stable sort puts the smaller id first. Measured with centroids \(c=[(0,2),(1,0),(-1,0),(0,1)]\) and \(q=(0,0)\), whose squared distances are \(4,1,1,1\): `nprobe=4` gives \([1,2,3,0]\) and `nprobe=2` gives \([1,2]\) — neither starts at 0 nor uses an arbitrary order.

**Distance of one posting.** Posting \(i\) in cell \(c\) stores the full vector \(v_{c,i}\), and its distance is
\[
d_{c,i}=\|q-v_{c,i}\|_2^2=\sum_{t=0}^{D-1}(q_t-v_{c,i,t})^2 .
\]
No residual coding and no PQ approximation: the stored vector is the document itself.

**Union and deduplicate to the minimum.** Let the set of scanned postings be
\[
\mathcal P=\{(c,i): c\in\pi_{0:P}\}.
\]
For a global id \(g\) occurring in several postings, define
\[
D(g)=\min\{d_{c,i}:\ \text{id}(c,i)=g\}.
\]
The output contains each id once as \((g,D(g))\). An equivalent implementation sorts all postings lexicographically by \((\text{id},\text{dist})\); the first row of each id then carries the minimum distance. The reference achieves this with two stable sorts (distance first, id second): within one id the distances are already ascending, so the first row is the minimum. The pure-Python oracle instead uses a `dict` with an explicit `min`.

**Ranking and truncation.** Unique candidates are sorted by \((D(g),g)\) ascending and the first \(k\) are taken. In the reference the unique ids are already ascending, so one more stable sort by distance keeps tied distances in id order. The number of real slots is \(t=\min(k,\#\{g\})\), with `ids=-1` and `dists=+inf` tails.

**Statistics identities.** With \(N_{\text{post}}\) scanned postings and \(N_{\text{uniq}}\) distinct ids,
\[
N_{\text{dup}}=N_{\text{post}}-N_{\text{uniq}},\qquad
N_{\text{ret}}=\min(k,N_{\text{uniq}}).
\]
`n_duplicates` counts removed *occurrences*, not ids that had duplicates; `n_returned` depends only on \(k\) and the unique candidate count.

**dtype and padding.** Queries, centroids and inverted vectors must all be float32/float64; squared differences accumulate in float64 internally and distances are cast back to the query dtype. `cells`, `stats` and `ids` are int64 throughout.

**Complexity.** Cell selection \(O(QCD)\); scanning \(O(D)\) per posting, \(O(QND)\) total; dedup sorting \(O(N\log N)\). Space \(O(QC+N)\) excluding inputs.

## Key points and common mistakes

1. **Probe order is not cell-id order**: in the example the distances to cells 1/2 are 9/1, so `nprobe=2` yields `[0,2]` and skips cell 1. A dedicated case with centroid distances \(4,1,1,1\) requires `nprobe=2` to produce `[1,2]`.
2. **Ties choose the smaller cell id**: achieved by a stable sort; `torch.argsort(..., stable=True)` preserves the original index order of equal keys. Measured: `argsort([2,1,1,0,1], stable=True)` returns `[3,1,2,4,0]`.
3. **Deduplication must take the minimum, not the first or last occurrence**: in the example id 3 appears first in cell 0 with distance 0 and later in cell 2 with distance 4, where the first happens to be the minimum; a second test has id 1 three times in one cell with distances 9, 0, 1, so the answer is 0 — first-wins gives 9 and last-wins gives 1, both fail.
4. **Meaning of the statistics columns**: `n_postings` includes duplicates, `n_unique` is the number of distinct ids, `n_duplicates` is their difference, and `n_returned=min(k,n_unique)`. Measured: the example gives `[4,3,1,3]`; `nprobe=1` gives `[2,2,0,2]`; `nprobe=5` (clamped to \(C=3\)) also scans id 8 in cell 1 and gives `[5,4,1,3]`.
5. **nprobe boundaries**: `nprobe=0` makes `cells` shape \((Q,0)\) and all four statistics zero; `nprobe>C` means \(C\) and `cells` is the full cell order; `nprobe=1` takes only the nearest, smallest-id cell. All three were measured.
6. **k=0 leaves intermediates untouched**: `ids`/`dists` have shape \((Q,0)\), but `cells` is still `[[0,2]]` (example) and `stats` is still `[4,3,1,0]` — scanning and deduplication happen, only the result list is empty.
7. **Accumulate in float64**: measured on a random \(D=32\) case with 5 cells of 150 vectors and coordinates scaled by 10 (distances 2574.5815–11984.5059), a pure float32 distance differs from float64 accumulation plus a cast by at most \(9.76562\times10^{-4}\) absolute, mean \(2.13542\times10^{-4}\), above the \(10^{-5}\) float32 judging tolerance.
8. **Padding and empty candidates**: fewer than \(k\) candidates still yields \((Q,k)\) with `-1`/`+inf` tails; an entirely empty inverted table still returns a padded result, not an empty tensor.
9. **C=0**: no centroids and no lists, so `cells` is \((Q,0)\), `stats` is all zeros and the result is all padding; do not raise because of an empty dimension or a division by zero.
10. **Ids may repeat but must be non-negative**: duplicates are the whole point of the task; a negative id collides with the `-1` sentinel and raises `ValueError`.

## Differences from neighboring problems

- **vs `retrieval-ivfpq-adc` (same batch)**: that task stores PQ codes and its core is the per-subspace LUT sum (ADC), with no deduplication and no exposed cell sequence; this task stores full vectors and its core is probe order, cross-cell deduplication to the minimum distance, and candidate-count statistics, with plain squared Euclidean distances.
- **vs `retrieval-ivf-assign`**: that task only assigns each vector to its nearest coarse centroid and returns an index; this task selects \(n_{\text{probe}}\) cells on the *query* side and merges postings inside them, adding order, deduplication and statistics.
- **vs `retrieval-hnsw-greedy`**: that walks a graph hop by hop and can stop in a local optimum; this is a deterministic scan of selected inverted lists with no graph and no early stopping.
- **vs `retrieval-rrf-fusion`**: that fuses **ranks** from several retrieval routes with no distances or cells; this fuses **distances** of one query across several inverted lists and explicitly deduplicates.
- **vs `retrieval-kmeans-lloyd`**: that trains coarse centroids; here they are given.

## Cross-check convention and complexity

- **Reference structure**: stable-sort cells in float64; broadcast distances per cell; `torch.cat`, then two stable sorts (distance, then id) and take each id's first row to deduplicate; stable-sort the unique candidates by distance; compute statistics from tensor sizes.
- **Independent oracle**: `_oracle` computes distances with pure-Python nested loops over cells and postings, deduplicates with a `dict` and `min`, and sorts with `sorted(best.items(), key=(dist,id))`. This is structurally different from the reference's "concatenate + two stable sorts + boolean mask" and catches probe-order, deduplication-direction and statistics mistakes.
- **Cases**: the example (measured `EXAMPLE_*`, including the nprobe=1 branch); 5 random float32 draws × `nprobe∈{0,1,2,4,9}` × `k∈{0,1,3,10}`; 3 random float64 draws × `nprobe∈{0,1,3,4,8}` × `k∈{0,2,6}`; random ids drawn with replacement to create duplicates within and across cells; dedup-to-min (including "neither first nor last is the minimum"); cell tie order; nprobe boundaries; top-k truncation and huge k; all-empty, k=0, C=0, Q=0; 9 invalid-input groups.
- **Tolerances**: `ids`/`cells`/`stats` with `torch.equal`; float32 distances `atol=1e-5, rtol=1e-6`; float64 `atol=1e-9, rtol=1e-12`.
- **Measured log** (inline scripts under `.venv-eval/bin/python`, calling the reference directly):
  - example `nprobe=2` → `ids [[3,7,5]]`, `dists [[0.,0.,1.]]`, `cells [[0,2]]`, `stats [[4,3,1,3]]`; `nprobe=1` → `ids [[3,7,-1]]`, `dists [[0.,0.,inf]]`, `cells [[0]]`, `stats [[2,2,0,2]]`.
  - example `nprobe=5` (\(C=3\)) → `cells [[0,2,1]]`, `stats [[5,4,1,3]]`; `nprobe=0` → `cells` shape `(1,0)`, `stats [[0,0,0,0]]`.
  - tied centroid distances \(4,1,1,1\): `nprobe=4` → `cells [[1,2,3,0]]`; `nprobe=2` → `[[1,2]]`.
  - same-cell duplicate id 1 at distances 9/0/1 → `ids [[1,-1]]`, `stats [[3,1,2,1]]`.
  - float32 vs float64 accumulation: max absolute difference \(9.76562\times10^{-4}\) (distance range 2574.5815–11984.5059).
- **Complexity**: \(O(QCD+QND+N\log N)\) time; \(O(QC+N)\) extra space.
