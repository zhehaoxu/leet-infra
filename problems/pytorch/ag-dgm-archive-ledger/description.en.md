# Darwin Godel Machine: Parent Selection and Admission in Archive Evolution

DGM lets a coding agent rewrite its own codebase, re-runs the benchmark afterwards, and **only a
self-modification that actually scores better enters the archive**. The next round draws parents from
the archive by "score times rarity", and each parent picks one self-improvement direction (fill empty
patches, fix stochasticity, fix over-long context, or attack one unresolved issue). This problem turns
one generation of `jennyzzt/dgm` into a pure function ledger: candidate table, parent probabilities,
entry dispatch, compile filter, admission gate and the full-eval threshold.

## Interface

```python
def solve(commits, children, method="score_child_prop", size=2,
          update="keep_all", noise_leeway=0.1, num_issues=60,
          num_full_eval=500, parent_picks=None, pad_picks=None,
          draws=None, baseline=None) -> dict
```

`commits` is the archive in order (must contain `"initial"`), `children` is this generation's
offspring, and each record is `{"id", "score", "resolved", "unresolved", "empty", "submitted",
"parent", "metadata_ok", "context_exceeded"}`. The result carries `candidates`, `probabilities`,
`parents`, `entries`, `draws_used`, `compiled`, `rejected`, `archive` and `threshold`.

## Input and Output

```
Input:  commits=[initial 0.30, a 0.42, b 0.51, c 0.61, d 0.20]
     children=[e 0.65, f 0.90(empty patch)]
     method="best", size=2
Output: parents = ["d", "initial"]           # ascending: the two worst parents
     compiled = ["e"]；rejected = [{id:"f", reason:"no_patch"}]
     archive  = [..., "e"]
```

`best` is the counter-intuitive one: the official code writes `sorted(candidates, key=score)[:size]`,
i.e. **ascending**, so it picks the parents with the **lowest** scores. Copied verbatim, not fixed.

## Rules

- **Candidate table**: only commits with `metadata_ok` enter, in archive order. `children_count` counts
  **in-archive parent links**; when the parent is missing from the table the link is silently dropped
  (the child itself stays a candidate).
- **Parent probabilities**: `sigmoid(10·(score − 0.5))`, normalized; `score_child_prop` divides each
  weight by `1 + children_count` before renormalizing. `random` is uniform, `no_darwin` always takes the
  last candidate.
- **`best`**: ascending by score, first `size` entries; if short, `pad_picks` indexes into the already
  selected list to pad.
- **Entry dispatch**: each parent walks three gates in order, consuming coins from `draws`; a gate that
  **short-circuits does not consume** its coin:
  1. only if `empty >= 0.1 · (empty + resolved + unresolved)` take the first coin; `< 0.25` yields
     `solve_empty_patches`;
  2. unconditionally take the second coin; `< 0.25` yields `solve_stochasticity`;
  3. only when `context_exceeded` is true take the third coin; `< 0.25` yields `solve_contextlength`;
  4. otherwise: `unresolved == 0` records `skipped` (no fourth coin), else take one coin and index
     `min(int(coin · unresolved), unresolved − 1)`.
- **Compile filter**: check `metadata_ok` / `resolved + unresolved > 0` / `submitted >= num_issues` in
  order; the first failure records `no_metadata` / `no_patch` / `not_enough_tests` (no stacking).
- **Full-eval threshold**: the pool always gets the **initial score unconditionally**, then the scores of
  archive members with `submitted >= 0.9 · num_full_eval` (so a full-eval `initial` is counted twice);
  take the second highest (the only element if there is one), then `max(threshold, 0.4)`.
- **Admission**: `keep_all` takes every compiled child; `keep_better` only takes
  `score >= initial − noise_leeway` (equality passes).
- **Validation**: `size >= 1`, `num_issues >= 0`, `num_full_eval >= 1`, finite `noise_leeway` (`bool`
  rejected), `method`/`update`/`baseline` in their enumerations, coins inside `[0, 1)`, an archive that
  contains `initial` with unique ids, unique `children` ids, `parent_picks` of length `size`, and enough
  coins — all `ValueError`.

## Hints

- The gates short-circuit: when the `empty` ratio is too low the first coin never exists, so the second
  gate takes what you thought was the first. Miscount one coin and every later entry is shifted.
- `children_count` only counts in-archive parent links; this generation's `children` do not contribute —
  it is "how often this commit has been a parent", not "how many children it has".
- Counting `initial` twice in the threshold pool is real official behaviour (unconditional append, then
  one more inside the loop), not a typo. Copy it.
