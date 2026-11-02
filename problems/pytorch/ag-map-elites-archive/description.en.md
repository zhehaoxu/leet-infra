# MAP-Elites Quality-Diversity Archive Ledger

An evolutionary search that keeps a single best solution collapses onto one peak. MAP-Elites
replaces it with a **grid archive**: behaviour descriptors cut the solution space into cells and
each cell keeps only the best solution that landed in it — diversity is preserved without giving
up local optima. The archive core of program-evolution systems such as AlphaEvolve and OpenEvolve
is exactly this grid.

## Interface

```python
def solve(descriptors, scores, bins, ranges) -> dict
```

`descriptors` is `(N, D)` of behaviour descriptors (program length and latency, say), `scores` is
`(N,)` of fitness, `bins` is the per-dimension cell count (a scalar means the same for every
dimension), and `ranges` holds each dimension's `(lo, hi)`. The ledger is:

| Field | Meaning |
| --- | --- |
| `cells` | `(N,)`: flat cell id per candidate, `-1` when out of range |
| `accepted` | `(N,)`: whether the candidate became the cell's new elite |
| `elite_score` | `(total cells,)`: current elite score per cell, `-inf` when empty |
| `elite_index` | `(total cells,)`: candidate index of the elite, `-1` when empty |
| `occupancy` / `best_score` / `best_index` | filled cells, global best score and its index |

## Input / Output

```
Input: descriptors=[[0.0, 0.0], [0.9, 0.9], [1.0, 1.0], [0.5, 0.5]],
      scores=[1.0, 2.0, 3.0, 0.5], bins=2, ranges=[(0, 1), (0, 1)]
Output: cells=[0, 3, 3, 3], accepted=[True, True, True, False],
      elite_score=[1.0, -inf, -inf, 3.0], elite_index=[0, -1, -1, 2], occupancy=2
```

`[1.0, 1.0]` equals the upper bound and clamps into the last cell `3`; its 3.0 strictly beats the
2.0 already there, so it replaces it. `[0.5, 0.5]` also lands in `3`, but 0.5 loses to 3.0 and is
rejected.

## Rules

- **Cell assignment**: `slot = floor((v - lo) / (hi - lo) * bins[k])`; for `v == hi` this yields
  `bins[k]`, clamped to `bins[k] - 1` so a legal boundary solution is never dropped.
- **Out-of-range candidates are discarded**: `v < lo` or `v > hi` never enters the archive,
  `cells` records `-1` and `accepted` is false. Discard, not clamp.
- **Flat ids are row-major**: `flat = ((i0·b1) + i1)·b2 + i2 …`, with the last dimension varying
  fastest.
- **Only strict improvements replace**: a candidate takes a cell only when `score >` the current
  elite score; ties keep the **earlier** arrival (the archive has memory, so "newest" is not
  "best").
- **Empty cells and the global best**: empty cells report `-inf` and `-1`; the global best is the
  maximum elite score, and ties resolve to the **earlier candidate index**.
- **Validation**: `bins` is a `>=1` integer or a list of `>=1` integers of length `D`; `ranges`
  has length `D` with `lo < hi` each (equality rejected); `scores` has length `N` and no `NaN`;
  `N = 0` is legal.

## Hints

- Cell assignment uses `floor`, not rounding: `0.5` belongs to cell 1 when `bins=2`.
- `accepted` and "landed in a cell" are different things: a candidate can land in a cell and still
  be rejected for a lower score.
