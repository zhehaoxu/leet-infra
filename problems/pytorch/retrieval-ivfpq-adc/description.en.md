# Retrieval: IVF-PQ Asymmetric Distance Computation

## Problem

IVF-PQ is the core of Faiss `IndexIVFPQ` (Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545): a coarse quantizer partitions the space into Voronoi cells, each cell keeps an inverted list, and documents inside a list are stored not as raw vectors but as PQ codes over \(m\) subspaces. At query time only a few nearest cells are visited, and candidates are scored with **Asymmetric Distance Computation (ADC)** — the query stays in full precision and is never quantized, while each database code is turned into an approximate distance through a lookup table.

This task chains "pick cells" and "ADC scoring" into one deterministic function. Three things must all be right:

- each query independently picks `nprobe` cells by **squared Euclidean distance** to the coarse centroids, breaking ties by the smaller cell id;
- every subspace has its own LUT, and a document's approximate distance is the **sum** of its \(m\) per-subspace LUT entries — never mix codebooks across subspaces;
- final ranking is **(distance ascending, global id ascending)** top-k, with `-1` / `+inf` padding when there are fewer than \(k\) candidates.

CPU, fully deterministic, no RNG calls.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract.

- **Input shapes**: queries are a \((Q,D)\) floating tensor, \(Q\) may be 0; centroids are \((C,D)\); codebooks are \((m,K,d)\) with \(D=m\cdot d\). The inverted lists are two lists of length \(C\): item \(c\) holds the cell's global ids \((n_c,)\) int64 and codes \((n_c,m)\) int64, and \(n_c\) may be 0. All three dense tensors must share one dtype, float32 or float64 only.
- **Cell selection**: for each query compute \(\|q-c_j\|^2,\ j=0\dots C-1\), sort ascending and take the first `nprobe`; ties go to the smaller cell id. `nprobe=0` probes nothing, `nprobe>C` means all \(C\) cells.
- **ADC distance**: for subspace \(j\) and codeword \(t\),
  \[
  \text{LUT}[j,t]=\bigl\|q_{j\cdot d:(j+1)\cdot d}-\text{codebooks}[j,t]\bigr\|^2,
  \]
  and a document whose code row is \(\text{codes}[i]\) gets
  \[
  \hat d(q,x_i)=\sum_{j=0}^{m-1}\text{LUT}\bigl[j,\text{codes}[i,j]\bigr].
  \]
  The query side is not quantized; distances accumulate in float64 internally and are cast back to the query dtype on output.
- **Output**: `ids` is \((Q,k)\) int64, `dists` is \((Q,k)\) with the query dtype. Each row lists real candidates first, ordered by (distance ascending, global id ascending); when there are fewer than \(k\), trailing `ids` slots are \(-1\) and trailing `dists` slots are \(+\infty\). With `k=0` both tensors have shape \((Q,0)\).
- **Degeneracy**: with `nprobe=0`, all selected cells empty, or \(C=0\), the row has no real candidate (all padding if \(k>0\)). Empty cells are skipped and consume no candidate slot.
- **Global-id convention**: one id appears at most once in the whole inverted table; this task does not deduplicate, and repeated ids are outside the contract.
- **Error contract**: wrong dimensions, \(D\neq m\cdot d\), mismatched or unsupported dtypes, list lengths different from \(C\), id/code row-count mismatch, ids not 1-D int64, codes not \((n_c,m)\) int64, a negative id, a code outside \([0,K)\), `nprobe<0`, or `k<0` all raise `ValueError`.

## Example

Take \(D=2,m=2,d=1,K=2,C=2\):

- centroids: \(c_0=(0,0)\), \(c_1=(10,10)\);
- both subspace codebooks are \(\{[0],[10]\}\);
- cell 0: id 0 with code \((0,0)\) (approximates \((0,0)\)), id 1 with code \((1,0)\) (approximates \((10,0)\));
- cell 1: id 2 with code \((0,1)\) (approximates \((0,10)\)), id 3 with code \((1,1)\) (approximates \((10,10)\)).

Query \(q=(0,0)\), `nprobe=2`, `k=3`:

| doc | code | subspace 0 dist | subspace 1 dist | ADC dist |
|---|---|---|---|---|
| 0 | (0,0) | 0 | 0 | 0 |
| 1 | (1,0) | 100 | 0 | 100 |
| 2 | (0,1) | 0 | 100 | 100 |
| 3 | (1,1) | 100 | 100 | 200 |

The squared distances from \(q\) to the two centroids are 0 and 200, so both cells are probed. Sorting by (distance, id), ids 1 and 2 tie at distance 100 and the smaller id comes first:

```
ids   = [[0, 1, 2]]
dists = [[0.0, 100.0, 100.0]]
```

With `nprobe=1` only cell 0 is probed, candidates are ids 0/1, and `k=3` pads the third slot:

```
ids   = [[0, 1, -1]]
dists = [[0.0, 100.0, inf]]
```

## Constraints

- Return the pair `(ids, dists)`: `ids` is \((Q,k)\) int64 and `dists` is \((Q,k)\) with the query dtype.
- Judging compares ids with `torch.equal`; floating distances use `atol/rtol` \(10^{-5}/10^{-6}\) for float32 and \(10^{-9}/10^{-12}\) for float64, and padding must be exactly \(+\infty\).
- \(Q,C\) may be 0 and \(n_c\) may be 0; `nprobe` and `k` must be non-negative integers.
- No random numbers; any implementation path is fine (per-query loops, concatenating cells then vectorizing, or LUT gathers), but the result must match the ordering contract exactly.
