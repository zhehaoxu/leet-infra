# notes (ag-zep-invalidation-ledger)

## Sources

- Paper: Zep: A Temporal Knowledge Graph Architecture for Agent Memory
  (arXiv 2501.13956), §2.3 introduces the bi-temporal model: `t_valid` /
  `t_invalid` are event time, `t_created` / `t_expired` are transaction time, and
  a superseded fact gets `t_invalid` set to the new fact's `t_valid`.
- Implementation: `getzep/graphiti` main (fetched 2026-11-02). Section by section:
  - `graphiti_core/utils/maintenance/edge_operations.py`
    - `_dedupe_extracted_edge` (lines 905-1089): empty-pool early return, fast
      path, continuous index translation, duplicate out-of-range handling,
      `resolved` takes the first valid duplicate, timestamps only for new edges.
    - `_apply_temporal_invalidation` (715-745): stamp `expired_at` for an edge
      that already says it is history, then scan ascending candidates for the
      first later `valid_at` that supersedes the new edge, then hand off to
      `resolve_edge_contradictions`.
    - `resolve_edge_contradictions` (748-783): two skip predicates and one
      invalidate predicate.
    - `_extract_edge_timestamps` (786-831): the "any timestamp already set →
      skip the whole extraction" gate.
  - `graphiti_core/utils/maintenance/dedup_helpers.py::_normalize_string_exact`
    (40-43): `re.sub(r'[\s]+', ' ', name.lower()).strip()`.
  - `graphiti_core/utils/datetime_utils.py`: `ensure_utc` reads naive datetimes
    as UTC and converts other zones to UTC; `utc_now()` returns a fresh instant.
  - `graphiti_core/edges.py`, `EntityEdge`: `expired_at` is documented as
    "datetime of when the node was invalidated", sitting next to `invalid_at`.

## Source semantics (pinned one by one)

1. **Two pairs, two jobs**: `invalid_at` records event time (the moment the fact
   stopped being true), `expired_at` records transaction time (this ingest's
   `now`). The example and the `writes` stream both separate them.
2. **Empty pool comes first**: `if len(related_edges) == 0 and
   len(existing_edges) == 0` returns `candidates=None` immediately — the fast
   path is not tried and nothing is invalidated.
3. **The fast path sits after the empty check**: the first edge in
   `related_edges` (never `existing_edges`) whose normalized text matches and
   whose endpoints match is reused. It returns `duplicate_edges` as an **empty
   list** (`return resolved, [], None`), not `[i]`, and `candidates=None` means
   contradiction handling is skipped entirely.
4. **Continuous indices**: `related` occupies `0..n-1`, `existing` is offset by
   `invalidation_idx_offset`. The two out-of-range rules differ — duplicates use
   `len(related_edges)` only, contradictions use
   `len(related)+len(existing)-1`. Negative indices are out of range for both.
5. **`resolved` takes the first valid duplicate**: the official loop is
   `for ...: resolved = ...; break`, so later valid duplicates only lengthen
   `duplicate_edges`.
6. **A duplicate hit keeps the old timestamps**: the code comment says
   "duplicated edges retain their existing timestamps", and
   `_extract_edge_timestamps` is called only when
   `resolved_edge.uuid == extracted_edge.uuid`.
7. **Extraction gate**: `if edge.valid_at is not None or edge.invalid_at is not
   None: return` — either field kills the whole extraction. Per-field judgement
   happens one level down as `if valid_at:` (empty string skipped).
8. **Step 1 only looks at `resolved`**: `if resolved.invalid_at and not
   resolved.expired_at` stamps `expired_at = now`. Once `expired_at` is set, step
   2 is skipped wholesale.
9. **Step 2 has three strictness points**:
   - the sort key `(c.valid_at is None, ensure_utc(c.valid_at))` puts missing
     `valid_at` **last** and is **stable** (equal values keep their order);
   - the predicate `candidate_valid_at > resolved_valid_at` is **strictly
     greater**;
   - a hit `break`s, so among several later candidates only the earliest after
     sorting supersedes.
10. **The sort is in place**: `invalidation_candidates.sort(...)` mutates the
    very list that is then passed to `resolve_edge_contradictions`, so
    `invalidated_edges` comes out in the **sorted** order. This is the observable
    that a "copy before sorting" implementation loses, and the ledger pins it
    with `sorted_candidate_uuids`.
11. **Step 3's three predicates**:
    - `edge.invalid_at <= resolved.valid_at` → skip (inclusive);
    - `resolved.invalid_at <= edge.valid_at` → skip (inclusive);
    - `edge.valid_at < resolved.valid_at` → invalidate (strictly less).
    The official code writes `if A or B: continue / elif C:`; this solution writes
    two `if ... continue` plus one `if`, which is equivalent because A/B
    short-circuit.
12. **`expired_at` is preserved**: `edge.expired_at = edge.expired_at if ... is
    not None else utc_now()` never rewrites an existing expiry.
13. **Repeated candidates dedupe themselves**: on the second pass
    `edge.invalid_at` already equals `resolved.valid_at`, so the inclusive first
    predicate trips and `invalidated` never holds the same uuid twice.
14. **A self-referencing candidate cannot kill itself**: the third predicate
    requires strict inequality, and the values are equal.
15. **`now` is taken once per call**: the official code grabs `utc_now()` at the
    top of `_apply_temporal_invalidation` and reuses it for every `expired_at`.
    Here it is the explicit `now` argument, so every expiry written in one call
    is necessarily identical.

## On the B predicate (`resolved.invalid_at <= edge.valid_at`)

It reads like defensive code, but it **does** change outcomes on its own. When
`resolved.valid_at` and `resolved.invalid_at` form an inverted interval
(invalidation earlier than validation — a real product of LLM date extraction),
a candidate whose `valid_at` lands in `[resolved.invalid_at, resolved.valid_at)`
is skipped by B and would be invalidated by C without it. The
`check_new_edge_dead_first` checkpoint covers both this case and its equality
boundary.

By contrast, the `invalid_at` written in step 2 (a later candidate's `valid_at`,
which is `> resolved.valid_at`) can never make B fire together with C — the two
are mutually exclusive. That is exactly why testing only "the new edge got
superseded" scenarios would not exercise B.

## Deviations from the official implementation

1. **`utc_now()` made explicit**: the official code reads the wall clock, which
   cannot be judged; here `now` is required and raises `ValueError` when missing.
2. **Future-tense normalization is out of scope**: `temporal_edge_utils.py`'s
   `normalize_future_temporal_bounds` rewrites assertions such as
   "will / plans to / by Friday" relative to the reference time after extraction.
   That belongs to the **extraction** side and is orthogonal to invalidation; its
   regex surface is large enough to be a problem of its own. This problem sticks
   to "parse and write", and the statement says so.
3. **LLM, embeddings and attribute extraction are all out**: `decisions` and
   `timestamps` are arguments.
4. **The `edges` snapshot and the `writes` audit stream are additions**: upstream
   only has in-memory objects and DB writes, no ledger. They exist so that "which
   edge was written with what, when" can be checked line by line.

## Judging and mutation

41 mutants: 40 substantive ones all CAUGHT, 1 equivalent
(`m-package-order`, replacing `for uid in order` with `sorted(order)` when
building `edges`).

Why it is equivalent: `edges` is a dict, and Python dict equality ignores
insertion order; the `order` variable is used nowhere else in the solution
(`state` is the source of values). Under the mutant, `random_cross`'s
cross-check over 60 random fixtures and every hard assertion still pass, which
confirms the ordering is unobservable.

Covered substantive branches (mapping one-to-one to mutant names): empty path
`and`/`or`, empty path dropping timestamps, the fast path's three conditions
(source / target / normalized text), the fast path wrongly scanning `existing`,
duplicate bounds using `max_idx` instead of `len(related)`, dropping the negative
check, first-vs-last duplicate, contradiction bounds `>`/`>=`, negative indices
mapping, `elif` turned into `else`, extracting timestamps for merged edges, the
extraction gate, the empty-string check, field write order, step 1's
`expired is None` gate, step 2's `>`/`>=`, `break` vs continue, sort direction /
missing sort / `None` ordering, step 2's "sort only when not expired" gate, each
of step 3's three predicates with its equality direction, `continue` vs `pass`
after A, deleting B, dropping the `resolved.valid` guard, `expired_at`
preservation, aliasing the candidate list so the original order gets rewritten,
the `field` suffix, ISO formatting, naive timestamps, the `now` / `decisions`
required-argument gates, the direction of `resolved_is_new`, and whether
`candidate_uuids` uses the original or sorted order.

## Traps for solvers

- Writing `sorted(candidates)` (which returns a new list) instead of sorting in
  place — both `sorted_candidate_uuids` and `invalidated` diverge from upstream,
  especially given that step 1's stamp suppresses sorting entirely.
- Writing `expired_at = now` unconditionally in step 3, overwriting a candidate's
  existing expiry.
- Implementing the fast path as "same fact text" while dropping the endpoint
  checks, or scanning `existing_edges` as well.
- Using the combined length as the duplicate bound, which turns `existing`-segment
  indices into duplicates.
- Writing step 2's `>` as `>=`, so a candidate with an equal `valid_at`
  supersedes the new edge.
