# notes (ag-dgm-archive-ledger)

## Sources

- Official implementation `jennyzzt/dgm`: `DGM_outer.py` (337 lines) and `utils/evo_utils.py` (127 lines).
  - `choose_selfimproves`: walks `archive` to build the candidate table (`accuracy_score`, the three id
    lists, `children_count`). When the parent is not in the table, `candidates[parent_commit]` raises
    `KeyError` and the outer `except` swallows it — but `candidates[commit]` was already assigned, so the
    child **stays a candidate**, it just does not contribute a count.
  - Four parent strategies: `no_darwin` takes `commits[-1:]`; `score_prop` normalizes
    `1/(1+exp(-10*(score-0.5)))`; `score_child_prop` additionally multiplies by `1/(1+children_count)`;
    `best` is `sorted(..., key=accuracy_score)[:min(size, len)]` — **ascending**, i.e. it picks the
    worst-scoring parents, padded with `random.choices(parent_commits, k=...)`.
  - Entry dispatch: `len(empty_ids) >= 0.1*num_total` and `random.random() < 0.25` →
    `solve_empty_patches`; else `random.random() < 0.25` → `solve_stochasticity`; else
    `any_exceeding_context_length(...) and random.random() < 0.25` → `solve_contextlength`; if nothing
    hits and `unresolved_ids == 0` → `continue` (no entry at all), else `random.choice(unresolved_ids)`.
    Because `and` short-circuits, a failed ratio check means that coin is never thrown — the subtlest
    layer of this problem.
  - `filter_compiled` → `is_compiled_self_improve`: the three id keys must exist,
    `resolved + unresolved > 0`, and `total_submitted_instances >= num_swe_issues[0]` (**only index 0**;
    passing `[small, medium]` silently ignores the second element).
  - `update_archive(keep_better)`: gate is `get_original_score() - noise_leeway`, admitted when
    `score >= gate`.
  - `get_full_eval_threshold`: the initial score enters the pool unconditionally, then the loop over
    `archive` (**which contains `initial`**) adds every score with `submitted >= num_full_eval*0.9`, so
    `initial` is counted twice; take `sorted(desc)[1]`, or `[0]` when there is only one element, then
    `max(threshold, 0.4)`.
- Paper: arXiv 2505.22954 (Darwin Godel Machine). The paper only gives the loop; every detail follows
  `DGM_outer.py`.

## Normalizations

- **Every random step is explicit**: parent sampling becomes a `parent_picks` index sequence (the
  probabilities are still emitted as a ledger field), `best` padding uses `pad_picks`, gate coins come
  from an ordered `draws` stream, and the unresolved pick is `int(coin * n)`.
- **Two-pass table build**: the official code walks once, so a parent that appears after its child loses
  that link to a `KeyError`. This problem builds all candidates first and then walks the links, making the
  count order-independent. The divergence is stated here and the single-pass behaviour is no longer
  asserted by any test.
- **`random.choices`** is not reproduced bit-for-bit; the contract is "candidate order + weights +
  explicit picks". Probabilities are outputs, not judgements.
- **`entry_ids` become counts + indices**: the official code picks an issue id, this problem picks an
  index (`int(coin*n)`), which is deterministic but not identical to `randbelow`.
- The `num_swe_issues[0]`-only quirk is not turned into a trap (`num_issues` is a plain scalar) but is
  recorded here for reference.
- Wall clock, thread pool, Docker and timeouts are all dropped: only pool management and judgement stay.

## Mutations

34 mutations, all caught (no equivalent mutant):

```
m-sig-scale             sigmoid drops the factor 10                    → probability ratio
m-sig-shift             sigmoid drops the 0.5 shift                    → probability ratio
m-prop-no-norm          probabilities not normalized                   → sum-to-one
m-prop-no-child         score_child skips the child divisor            → two-parent case
m-best-desc             best flipped to descending                     → picked-value case
m-no-darwin-first       no_darwin takes the first entry                → baseline case
m-gate-ratio-half       ratio gate 0.1 -> 0.5                          → gate case
m-gate-ratio-strict     ratio gate `>=` -> `>`                         → equality boundary
m-gate-empty-threshold  first-gate coin threshold 0.25 -> 0.5          → gate case
m-gate-ctx-always       third gate ignores context_exceeded            → coin consumption
m-gate-unresolved-neg   skip condition becomes < 0                     → skip case
m-entry-index-shift     index shifted by one                           → index case
m-entry-index-ceil      index becomes coin*(n-1)                       → index case
m-compiled-resolved-only  ignores unresolved in the emptiness check    → filter case
m-compiled-issues-le    test-count check uses `<=`                     → off-by-one case
m-compiled-meta         metadata not checked                           → filter case
m-threshold-first       threshold takes the max, not the second        → threshold case
m-threshold-no-floor    threshold not clamped to 0.4                   → threshold case
m-threshold-no-double   pool does not pre-seed the initial score       → double-count case
m-threshold-submitted-le  full-eval predicate uses `<=`                → 449/450 case
m-keep-all-always       always behaves like keep_all                   → admission case
m-keep-gate-strict      admission `>=` -> `>`                          → equality boundary
m-keep-gate-leeway      gate adds the leeway instead of subtracting    → admission case
m-keep-uncompiled       uncompiled children also admitted              → filter case
m-cand-meta             ineligible commits enter the candidate table   → counting case
m-count-no-parent-check parent link counted without a membership check → counting case
m-int-bool              int check accepts bool                         → validation case
m-real-bool             real check accepts bool                        → scalar boundary
m-real-finite           finiteness unchecked                           → validation case
m-draws-range           coins not checked against < 1                  → scalar boundary
m-size-min              size allows 0                                  → scalar boundary
m-method-check          method unchecked                               → validation case
m-picks-len             picks length unchecked                         → validation case
m-draws-exhaust         running out of coins does not raise            → exhaustion case
```

## Pitfalls

- The first draft assumed three coins per parent and was wrong every time: when the `empty` ratio fails,
  the first gate never throws, so the real count is two. The `draws_used` assertions are what exposed it.
- `m-keep-gate-strict` survived the first round: `0.30 - 0.1` is `0.19999999999999998` in floating
  point, so a child at `0.20` passes even with `>`. Only a **binary-exact** pair such as
  `0.50 - 0.25 = 0.25` distinguishes `>=` from `>`.
- `m-draws-range` survived because `draws=[1.0]` also raised `ValueError` in the mutant — via
  "not enough coins". Two coins are needed to separate "out of range" from "ran out".
- `m-size-min` survived because `size=0` with `parent_picks=[0]` trips the length check first; empty
  picks are required.
- The "child before parent" case was originally asserted; after switching to a two-pass build the
  behaviour changed, so the case was deleted rather than left as a false expectation.

## Duplicate checks

- `ag-funsearch-island-ledger`(748): islands, clustering, temperature sampling and weak-island reset.
  This problem is a **single archive lineage** with parent selection and admission — no clusters,
  islands or clonal resets.
- `ag-map-elites-archive`(745): a behaviour-descriptor grid of elites; here the structure is the
  commit parent/child relation.
- `ag-reflexion-memory-ledger`(725), `ag-memorybank-forget-ledger`(722): reflection and forgetting
  curves, no archive admission.
- `ag-lats-uct-ledger`(724), `ag-tot-bfs-ledger`: selection rules on trees/graphs, without the
  "self-modify + benchmark compare + compile filter" triad.
- `ag-self-instruct-filter`(747): instruction pool filtering and dedup, no lineage and no admission gate.
