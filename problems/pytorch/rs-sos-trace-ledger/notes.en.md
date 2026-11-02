# notes (rs-sos-trace-ledger)

## Sources

- Paper: Stream of Search (arXiv 2404.03683). The claim is "treat the search process as
  language": a symbolic searcher plays Countdown and serializes states, operations and
  forward/backward moves into a text trace, which is then used to train an LM. The paper
  gives the idea and the curves; the **line format, the id scheme and the parsing rules**
  are not in it.
- Code: `kanishkg/stream-of-search` main (the paper's official implementation).
  - `src/countdown_dfs.py::dfs` (6-65): 19-26 successor enumeration, 28-34 `threshold`
    filter, 35 sort, 39 id write, 41-45 goal test, 50-52 recursion, 54-57 sibling
    hand-off, 60-63 backtrack hint.
  - `src/countdown_utils.py::combine_nums` (7-20), `CountdownNode` (22-31),
    `sum_heuristic` (35-38), `mult_heuristic` (40-45), `simple_rating` (57-60),
    `parse_trajectory` (77-140), `metric_fn` (143-165).
  - `src/countdown_bfs.py` is the other route (beam + `random.shuffle`), non-deterministic
    and therefore unused here.
- The encoding side reproduces `dfs` verbatim: running the official module on 400 random
  instances gives traces that are **byte-identical** (0 diffs), and the verdicts plus
  ratings line up with `parse_trajectory` / `metric_fn` case by case.

## Traps in the upstream code

1. **`while open_set` turns exactly once per call**: the root call seeds an empty list and
   each recursive call passes a one-element `new_set` that is popped immediately. It is
   really a recursive DFS that expands on entry; the loop is just a wrapper.
2. **`open_set.sort(key=lambda x: -x[0])` + `pop()`** takes the **smallest** heuristic;
   on ties `CountdownNode.__lt__` only compares `heuristic`, so comparing the second
   element of `(h, node)` is always `False`, ties read as equal, and `sort` is stable —
   **ties keep generation order**.
3. **`node_index` always equals `enumerate`'s `g`**: it is incremented at the end of the
   loop body and a hit returns before the increment. So the "position" in an id is the
   post-sort position.
4. **`threshold` filters before sorting and numbering**: pruned branches never take an id,
   and numbering re-packs after pruning.
5. **The goal test happens before descending**: a single number equal to `target` writes
   `equal` and stops the whole tree; a single number that differs writes `unequal` and
   **does not descend** (so no `Generated` line either).
6. **The sibling hand-off writes the parent's state**: after a failed branch it writes
   `Moving to Node #parent` and `Current State: <parent numbers>` (the code uses
   `new_node.parent.*`), not the sibling's state.
7. **The `if open_set:` backtrack block is dead code**: the local `open_set` is always
   empty after the pop, so that `Moving to Node #next` line is never written. A trace may
   therefore end abruptly on an `unequal` line.
8. **`round(b // a, 0)` in `combine_nums` is a no-op**: `//` already yields an int, so
   `round` changes nothing; this problem writes `b // a` directly (an equivalent mutant
   proves it).
9. **Subtraction and division are always larger-minus-smaller / larger-over-smaller**: a
   result can never be negative and never a fraction, so a trace never shows `3-5=-2` or
   `3/2=1.5`.
10. **`a+b` and `a*b` always appear**, even when they duplicate another result (`2+2=4` and
    `2*2=4` both stay); division is added as a fourth item only when it divides evenly.
11. **Three "which one" choices in `parse_trajectory`**: the goal is the **first** match,
    the goal step is the **last** `Exploring Operation` before it, and the parent state is
    the **last** `Current State` before it.
12. **The empty `Operations` list trap**: `Operations: []` is captured as an empty string
    by `\[(.*?)\]`, and `"".split(", ")` yields `[""]`, not `[]`; replaying therefore fails
    to split the first "operation" → `Could not split operation into lhs, rhs`. **A
    two-number instance that hits on the first move scores 0**, the most counter-intuitive
    rule here.
13. **The available pool only grows**: `[n for n in available if n not in used_numbers]`
    compares an `int` against a list of `str`, always true, so nothing is ever removed; the
    pool is "first line's numbers + every earlier result". The same number can be reused
    without complaint.
14. **`metric_fn`'s `max_nodes` only knows 2–5 numbers**: with 1 or ≥6 numbers the variable
    is never assigned and the call raises `UnboundLocalError`.
15. **`simple_rating` is "line count + 1"**: the `+1` is the root, which only writes a
    `Current State` line and never counts as `Exploring Operation`.
16. **The full-tree formula is too small**: `3c2×4 × 2c2×4 = 48` assumes exactly four
    operations per pair, but "division only when exact" and duplicate `+`/`*` results make
    the real count larger, so a **valid trace can score a negative rating**, caught by
    `max(..., 0.0)`.
17. **`get_target_nums` (63-74) is an uncalled duplicate** (it repeats the first lines of
    `parse_trajectory`); this problem does not reproduce it.

## Normalizations in this problem

- **`eval(lhs)` → an explicit two-operand evaluator**: upstream calls `eval`; this problem
  only accepts the `a<op>b` form (`+ - * /`, plain non-negative decimals on both sides) and
  routes anything else to `Error in evaluating operation ...`. The encoder only ever emits
  that form, so this **narrows the interface** rather than changing semantics; `eval`'s
  exception text is also environment-dependent, and narrowing it is what makes the fallback
  verdict judgeable. Division is true division (`3/2=1` is invalid).
- **The `Couldnt parse last op` branch is deleted**: the upstream regex `(.*?=\d+)`
  guarantees a numeric right-hand side, so `int(...)` cannot fail and the branch is
  unreachable.
- **`[-1]` on an empty `Current State` list is normalized to an empty list**: upstream
  raises `IndexError` when a trace has a goal but no `Current State` line; here it degrades
  to an empty operation list (the encoder can never produce such input, so this only
  affects hand-fed traces).
- **Unparsable number lists fold into `Invalid input: ...`**: upstream's
  `[int(n) for n in ...]` raises `ValueError`; here it shares the "bad first line" verdict.
- **A first line whose number count is outside 2–5 gives `max_nodes = None` and rating
  0**: upstream raises `UnboundLocalError`; here it becomes a judgeable `None`.
- **Encoder-side argument checks are explicit**: an unknown `heuristic`, a `nums` count
  outside 2–5, a negative number, or `mult` with `target < 2` → `ValueError`. With `trace`
  given, none of this runs.

## Mutations

59 mutations: **58 substantive ones all CAUGHT**, one judged equivalent, with evidence:

- `m-mult-guard-one` (`target < 2` → `target < 1`): the only difference is `target == 1`,
  where the loosened guard lets `_mult_heuristic` build an empty `factors` list and `min()`
  raise `ValueError: min() arg is an empty sequence`. Both paths hand the caller a
  `ValueError`, and the contract only says "`mult` with `target < 2` → `ValueError`".
  42 cases (`target ∈ {0,1}` × `nums` combinations) show 0 difference in exception type.

First-round survivors that needed extra cases (each was masked by another guard):

- `m-sum-div-plus1` (`/ len(nums)` → `/ (len+1)`): every distance is scaled by the same
  factor, so the **sort order is unchanged**; only the absolute `threshold` can see it. The
  added case puts the threshold between `sum/len` and `sum/(len+1)`
  (`solve(6, [1,2,3], "sum", 2)`, where the correct implementation keeps no child at all).
- `m-goal-last` (`goals[0]` → `goals[-1]`): the encoder stops at the first hit, so there is
  only ever one `equal` line. The added hand-fed trace has **two** `equal` lines, the
  second under a different sibling.
- `m-eval-floordiv` (`a / b` → `a // b`): the encoder only writes division when it is
  exact, where both agree. The added trace uses a rounded write-up `3/2=1`.
- `m-eval-accept-any` (dropping the `isdigit` double-check): only non-two-operand left
  sides are affected. The added trace uses `3 + 4=7` (with spaces), where `int("3 ")` slips
  through and wrongly yields `Valid path.`.
- `m-head-slack` (`re.match` → `re.search`): the added trace prefixes the first line with
  `log: `.
- `m-no-clamp` (dropping `max(..., 0.0)`): needs a **valid trace whose score exceeds the
  full tree**. Brute-forcing three-number instances found `solve(18, [2, 4, 4], "mult")`:
  valid, `score` 51 > 48.

Writing the cases also hit three snags (all fixed):

- Operation order cannot be pinned on an ordinary instance — the sort reorders by
  heuristic, which looks like "multiply before add". Only a **tied** instance
  (`2+4=6` and `2*4=8` are equidistant in `7:[2,4]`) reveals generation order.
- In a hand-fed trace the state before the first step is the root with `Operations: []`,
  which trips trap 12 first; to test later branches the goal step's parent must **carry one
  operation**.
- `parse_trajectory` checks "the goal step's right side equals `target`" before replaying,
  so an arithmetic-error case must keep the goal step's right side **equal** to `target`
  and leave the mistake in an earlier step.

## Overlap check

- vs `ag-lats-uct-ledger` (order 724): that one is a numeric **selection/backup** ledger
  (Q/N/c updates, PUCT scores); this one is **text trace encoding/decoding** plus a rating,
  with no selection formula.
- vs `ag-toollm-dfs-ledger` (order 728): that one schedules tool calls over DFS (stack
  backtracking, dedupe pruning); it neither emits nor parses a trace.
- vs `rs-ttrl-pseudo-reward-ledger` / `rs-self-consistency-vote`: those aggregate samples
  and votes; this one serializes a single deterministic search.
- No problem in the library covers "trace serialization / replay / node rating".
