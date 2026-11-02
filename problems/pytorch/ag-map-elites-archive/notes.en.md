# notes (ag-map-elites-archive)

## Source

- Paper: MAP-Elites (arXiv 1504.04909, Mouret & Clune). The original algorithm is "grid plus one
  elite per cell": a solution maps to a cell of the behaviour-descriptor space and replaces the
  incumbent when the cell is empty or the newcomer is better.
- Engineering counterpart: the `archive` of AlphaEvolve / OpenEvolve (a `programs` dict keyed by
  cell) and QD variants (CVT-MAP-Elites, CMA-ME). This problem takes only the "grid plus strict
  improvement" layer, with no CVT partitioning or gradient archive.
- There is **no single official source to pin** here (implementations disagree on descriptor
  normalization and out-of-range handling), so the four rules — cell formula, discard-on-out-of-
  range, strict improvement, row-major flat id — are written into the statement and used as the
  contract.

## Normalizations in this problem

- Cell: `slot = floor((v - lo)/(hi - lo) * bins)`, `v == hi` clamps to `bins - 1`, and
  out-of-range candidates are discarded. These two are where implementations diverge most, so both
  are pinned explicitly.
- Flat ids are row-major with the last dimension varying fastest, so `elite_score` follows the
  Cartesian order of `bins`.
- Replacement requires **strictly greater**; ties keep the earlier arrival, and `best_index` also
  resolves ties to the earlier candidate.
- `accepted` marks only "became the new cell elite" and is kept distinct from "landed in a cell".
- Validation gates: the scalar `bins` must also be `>=1` (the first draft missed this and
  `bins=0` fell through to an `IndexError`), `ranges` requires `lo < hi`, `scores` must have no
  `NaN`, and `N = 0` is legal.

## Mutations

All 14 mutants are CAUGHT:

```
m-slot-clamp-drop            upper bound not clamped      → boundary check
m-slot-ge                    clamp condition becomes >    → boundary check
m-outside-gate               out-of-range not discarded   → out-of-range check
m-strict-improve             ties also replace            → tie check
m-flat-order                 flat id reversed             → multi-dim check
m-bins-len                   list length unchecked        → validation check
m-bin-zero                   per-item >=1 unchecked       → validation check
m-range-eq                   lo == hi allowed             → validation check
m-best-tie                   tie keeps the later one      → best-tie check
m-best-index                 records the cell, not the index → best_index assertion
m-nan-gate                   NaN scores allowed           → validation check
m-bins-bool                  scalar bool accepted         → validation check
m-occupancy                  occupancy counted as all cells → occupancy assertion
m-cells-discard-zero         out-of-range recorded as 0   → out-of-range check
```

## Overlap check

- `ag-got-operations-ledger`(723) covers operation-chain selection and merging, not a grid archive.
- `ag-reflexion-memory-ledger`(725) and `ag-memorybank-forget-ledger`(722) are memory and decay,
  with no behaviour-cell routing.
- `ag-lats-uct-ledger`(724) is UCT selection inside a tree and has no elite-replacement archive.
- The only subject here is "grid routing plus per-cell elite replacement".
