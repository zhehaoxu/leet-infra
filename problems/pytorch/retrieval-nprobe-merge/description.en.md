# Retrieval: nprobe Multi-Probe and Cross-Cell Merge

## Problem

IVF query cost and recall are both controlled by one knob: **nprobe**, the number of inverted lists visited per query (Faiss `IndexIVF`'s `nprobe`; the coarse-quantizer + inverted-list structure of Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545). Larger nprobe scans more candidates and improves recall at higher latency. This task implements only the three deterministic steps of a multi-probe query:

1. rank cells by **squared Euclidean distance** from the query to the coarse centroid and take the first nprobe (ties go to the smaller cell id);
2. **probe in that order**, take the union of the postings in those lists, and keep each global id only once with the **minimum** distance over all its occurrences;
3. sort the deduplicated candidates by **(distance ascending, global id ascending)** and take top-k.

To make the intermediate process observable, besides the top-k result the function also returns the **selected cell sequence** and **candidate-count statistics** (postings scanned, unique ids left after deduplication, duplicates removed, results returned). Inverted lists store full vectors, so a distance is just the squared Euclidean distance between the query and a stored vector; no PQ approximation is involved.

CPU, fully deterministic, no RNG calls.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract.

- **Input**: queries are \((Q,D)\) floating tensors (float32/float64, \(Q\) may be 0); centroids are \((C,D)\); the inverted lists are two lists of length \(C\), item \(c\) being \((n_c,D)\) full vectors and \((n_c,)\) int64 global ids, with \(n_c\) possibly 0. The same id may appear multiple times in one cell or across cells.
- **Probe order**: for each query compute \(\|q-c_j\|^2\), sort ascending and take the first \(P=\min(\text{nprobe},C)\) cells, breaking ties by smaller cell id. `nprobe=0` gives \(P=0\), probing nothing; `nprobe>C` gives \(P=C\). Probing must follow this order, not cell-id order.
- **Per-posting distance**: \(\|q-v\|^2\), summing squared differences over dimensions, accumulated in float64 internally and cast back to the query dtype on output.
- **Merge and deduplicate**: collect postings in probe order; if the same global id appears multiple times, keep it once with the minimum distance. If no postings are scanned, all four statistics columns are 0.
- **Top-k**: sort unique candidates by (distance ascending, global id ascending) and take the first \(k\), padding when fewer exist.
- **Four outputs**:
  - `ids`: \((Q,k)\) int64, trailing real slots padded with \(-1\);
  - `dists`: \((Q,k)\) with the query dtype, padding \(+\infty\);
  - `cells`: \((Q,P)\) int64, each row the actual probe sequence; shape \((Q,0)\) when `nprobe=0`, and \(P=C\) when `nprobe>C`;
  - `stats`: \((Q,4)\) int64 with columns `[n_postings, n_unique, n_duplicates, n_returned]`, where `n_postings` counts scanned postings including duplicates, `n_unique` counts distinct ids, `n_duplicates = n_postings - n_unique`, and `n_returned = min(k, n_unique)`.
- **Degeneracy**: with `k=0`, `ids`/`dists` have shape \((Q,0)\), while `cells` and `stats` still reflect the actual probing (`n_returned=0`); with \(C=0\), `cells` is \((Q,0)\) and `stats` is all zeros.
- **Error contract**: wrong dimensions, \(D\le 0\), mismatched or unsupported dtypes, list lengths different from \(C\), wrong vector shape or dtype, ids not 1-D int64, id/vector row-count mismatch, a negative id, `nprobe<0`, or `k<0` all raise `ValueError`.

## Example

Query \(q=(0,0)\), centroids \(c_0=(0,0)\), \(c_1=(3,0)\), \(c_2=(1,0)\), with squared distances 0, 9, 1. With `nprobe=2` the probe order is **cell 0 → cell 2** (cell 2 is closer than cell 1).

Inverted lists:

| cell | postings: id / vector | squared distance to q |
|---|---|---|
| 0 | 7 / (0,0); 3 / (0,0) | 0; 0 |
| 1 | 8 / (3,0) | 9 (not probed) |
| 2 | 3 / (2,0); 5 / (1,0) | 4; 1 |

Scanning in order yields 4 postings: `(7,0), (3,0), (3,4), (5,1)`. Id 3 occurs twice and keeps the minimum distance 0, leaving 3 unique candidates `(3,0), (7,0), (5,1)`; ids 3 and 7 tie on distance, so id ascending puts 3 first. With `k=3`:

```
ids   = [[3, 7, 5]]
dists = [[0.0, 0.0, 1.0]]
cells = [[0, 2]]
stats = [[4, 3, 1, 3]]   # 4 postings scanned, 3 unique, 1 duplicate removed, 3 returned
```

With `nprobe=1` only cell 0 is probed, candidates are ids 3/7, and `k=3` pads the third slot:

```
ids   = [[3, 7, -1]]
dists = [[0.0, 0.0, inf]]
cells = [[0]]
stats = [[2, 2, 0, 2]]
```

## Constraints

- Return the quadruple `(ids, dists, cells, stats)` with exactly the shapes and dtypes above; `cells` and `stats` are int64 throughout.
- Judging compares `ids`, `cells` and `stats` with `torch.equal`; floating distances use `atol/rtol` \(10^{-5}/10^{-6}\) for float32 and \(10^{-9}/10^{-12}\) for float64.
- \(Q,C,n_c\) may all be 0; `nprobe` and `k` must be non-negative integers.
- No random numbers; any implementation path is fine, but the probe order, the deduplicate-to-minimum rule, and the tie ordering must all hold.
