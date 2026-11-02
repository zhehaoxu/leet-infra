# notes (ag-funsearch-island-ledger)

## Sources

- Official implementation `google-deepmind/funsearch`, `implementation/programs_database.py`
  (297 lines) and `implementation/config.py`:
  - `ProgramsDatabase.__init__`: one `Island` per slot plus three parallel lists
    `_best_score_per_island` / `_best_program_per_island` / `_best_scores_per_test_per_island`.
  - `_register_program_in_island`: registers into the island first, then updates the island best via
    `_reduce_score`.
  - `register_program`: `island_id is None` broadcasts to every island; then checks the wall-clock
    `reset_period` to decide whether to reset.
  - `reset_islands`: `np.argsort(best + randn * 1e-6)[:num_islands // 2]` picks the weak half, clears
    them, draws a founder with `np.random.choice(keep_islands_ids)`, then clones via
    `_register_program_in_island`.
  - `Island.register_program`: keys clusters by `_get_signature`; a new key creates
    `Cluster(score, program)`.
  - `Island.get_prompt`: `temperature = init * (1 - (num_programs % period) / period)`,
    `_softmax(cluster_scores, temperature)`, `functions_per_prompt = min(len(clusters), fpp)`,
    `np.random.choice(..., p=probabilities)` with replacement, then numbering after
    `np.argsort(scores)`.
  - `Cluster.sample_program`: `normalized = (lengths - min) / (max + 1e-6)`,
    `softmax(-normalized, 1.0)`.
- The papers are only there to establish that this is FunSearch's mechanism: arXiv 2306.12424
  (FunSearch) and the island variant of 2506.13131. They do not spell out the pool management, so
  `programs_database.py` is the sole authority.

## Normalizations

- **Every random step is an explicit input**: the `island` of a `prompt` event, the `clusters` /
  `programs` sampling outcome, and the `founders` are all parameters. No `np.random` remains.
- **`np.argsort` is unstable** → version ordering is normalized to `(cluster score, sampling position)`
  ascending. Python's `sorted` is already stable, so this matches the natural behaviour.
- **Reset noise**: the official code adds `1e-6` Gaussian noise to break ties; this problem normalizes
  to `(island best, island id)` ascending, i.e. ties go to the smaller id.
- **The argmax renormalization in `scipy.special.softmax`**
  (`result[argmax] = 1 - sum(others)`) moves the value by ~1e-17 in float64, while the tests compare
  at 1e-12 relative tolerance, so that step is **not** part of the contract (noted here for reference).
- **Wall-clock `reset_period`** → an explicit `{"kind": "reset"}` event; `_last_reset_time` is not
  part of the ledger.
- **Code rewriting** (`rename_function_calls`, docstring replacement, `Program.__str__`) is out of
  scope: only version names and docstrings are emitted.
- Pinned boundaries: a prompt from an empty island raises (otherwise it would silently emit an empty
  prompt); with `num_islands=1`, `1 // 2 == 0` so `founders` must be empty; a founder must be in the
  kept half **and** hold a program — both checks are needed because a reset island may also hold one.

## Mutations

35 caught, 1 proven equivalent and dropped:

```
m-sig-unsorted          cluster key uses dict value order, not test-name order  → multi-test case
m-sig-desc              cluster key sorted descending                          → multi-test case
m-reduce-max            cluster score takes max instead of the last-inserted key → headline case
m-reduce-first          cluster score takes the first key                      → insertion-order case
m-reduce-lastkey        cluster score takes the lexicographically last key      → insertion-order case
m-cluster-score-overwrite  later members overwrite the cluster score            → frozen-score case
m-best-ge               ties also replace the island best                       → tie case
m-num-programs-stuck    counter never increments (temperature never decays)     → temperature case
m-broadcast-first-only  broadcast only reaches island 0                        → broadcast case
m-temp-fixed            temperature pinned at the initial value                 → temperature case
m-temp-plus1            phase shifted by one                                    → temperature case
m-take-no-min           no clamp to the cluster count                           → few-cluster case
m-version-plus2         version number off by one, plus                      → version assertion
m-header-doc-index      header docstring points at the wrong version            → version assertion
m-sort-version-desc     versions sorted by descending score                     → ordering case
m-prog-denom-nudge      denominator drops the 1e-6                              → ratio case
m-prog-denom-span       denominator becomes max-min                             → ratio case
m-prog-sign             minus sign dropped (long programs favoured)             → monotonicity case
m-reset-min-one         resets one island even when num_islands=1               → single-island case
m-reset-strongest       resets the strongest half                               → reset case
m-reset-keep-swapped    kept and reset halves swapped                           → reset case
m-reset-founder-any     founder not checked against the kept half               → new case
m-reset-founder-empty   founder allowed to hold no program                      → new case
m-reset-count-check     founders length unchecked                               → negative cases
m-reset-keep-counter    reset does not clear the counter                        → reset case
m-prompt-empty-ok       empty island allowed                                    → empty-island case
m-prompt-len-check      only compares lengths, not the expected count           → count case
m-int-bool              int check accepts bool                                  → validation case
m-real-bool             real check accepts bool                                 → validation case
m-real-finite           finiteness unchecked                                   → validation case
m-temp-nonpos           rejects negatives but allows zero                       → validation case
m-kind-check            kind unchecked                                          → unknown-kind case
m-scores-empty          empty scores allowed                                    → validation case
m-softmax-ignore-temp   probabilities ignore the temperature                    → temperature ratio case
m-softmax-no-norm       probabilities not normalized                            → sum-to-one case
```

Equivalent mutant (excluded): `normalized = (x - lo)/denom` → `x/denom`. Softmax is invariant to a
constant shift and the denominator only depends on `max`, so subtracting `min` shifts every logit by
the same constant and the output is bit-identical (verified with a full `_eq` comparison).

## Pitfalls

- The first version of `check_softmax_over_clusters` used `exp(2.0 / 0.5)` as the expected ratio and
  ignored that the temperature had already been shaved from `0.5` to `0.49996…` by
  `num_programs % period`; the ratio differs in the 5th digit. Read the actual temperature from
  `rec["temperature"]`.
- `m-prog-no-min` is only caught because the temperature is recorded in the ledger; otherwise softmax
  shift invariance would turn it into an equivalent mutant.
- `-inf - (-inf)` is `nan`, so the comparison helper must test equality **before** applying a
  tolerance, or the empty-island best score fails the cross-check.
- A broadcast seed gives every island a program, so the "kept island holds nothing" branch only
  becomes observable with a case that registers into a subset of islands; likewise the "founder points
  at a reset island" branch needs a reset island that actually holds a program.

## Duplicate checks

- `ag-map-elites-archive`(745): a quality-diversity grid archive keyed by behaviour descriptors, one
  elite per cell. This problem clusters by **score vector**, samples by temperature and resets islands,
  and keeps every cluster member rather than a single elite.
- `ag-self-instruct-filter`(747): filtering and similarity dedup over a linearly growing pool. Here the
  pool is sharded into islands with resets and cloning, and the sampling probability is an explicit
  temperature softmax.
- `ag-generative-memory-retrieval`, `ag-memorybank-forget-ledger`: memory retrieval and decay, no
  evolutionary pool.
- `opt-*` and `inf-cache-aware-routing`: scheduling / routing, no cluster-island-reset structure.
- `kv-morphkv-island-fusion` only has "island" in its name; it fuses KV pages and is unrelated.
