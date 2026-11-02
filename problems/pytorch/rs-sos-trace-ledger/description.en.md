# Stream of Search Trace Serialization Ledger

Stream of Search (arXiv 2404.03683) treats the **search process itself** as language: a
symbolic searcher plays Countdown and serializes every state, operation and
forward/backward move into a **text trace**, which is then used to train a language model.

This problem keeps the deterministic half of that pipeline as an auditable book: a DFS
driven by a pure heuristic **encodes** `(target, nums)` into a trace, the upstream parser
**decodes** that trace back into an operation sequence and replays it step by step, and the
result folds into a 0–1 rating from "nodes spent / nodes in a full tree". It mirrors
`kanishkg/stream-of-search`'s `src/countdown_dfs.py::dfs` and `src/countdown_utils.py`
(`combine_nums`, the two heuristics, `parse_trajectory`, `metric_fn`).

Implement `solve(target, nums, heuristic="sum", threshold=None, trace=None)`.

## Inputs

- `target`: goal number, non-negative integer.
- `nums`: initial numbers, 2–5 of them, non-negative integers.
- `heuristic`: `"sum"` or `"mult"`, see below.
- `threshold`: when given, only children whose heuristic is `<= threshold` survive.
- `trace`: when given, **this text is decoded as-is** and `target` / `nums` /
  `heuristic` / `threshold` take no part; `encoded` is then `False`.

The two heuristics (both act on the number list *after* a move):

- `sum`: with one number left it is `|n - target|`; otherwise the **mean** distance to
  `target` (divided by the number of numbers).
- `mult`: collect every divisor of `target` in `[2, target]`, then sum each number's
  distance to its **nearest divisor**. `target < 2` has no divisors → `ValueError`.

## Encoding: one merge

Merging two numbers `a` and `b` yields at most four results, in a **fixed order**:

1. `a+b`
2. `a*b`
3. larger minus smaller: `b-a` when `a<=b`, else `a-b` (never negative)
4. larger divided by smaller: only when it **divides evenly** — `b/a` when `a<=b` and
   `a != 0` and `b % a == 0`; `a/b` when `a>b` and `b != 0` and `a % b == 0`

Each result reads `<left><op><right>=<result>` with plain decimal integers. The merged
number list is "the unused numbers **in their original order**" followed by "the new
result **appended at the end**".

## Encoding: child order and ids

- A level enumerates every pair `(i, j)` with `i < j`, and inside a pair the four
  operations above.
- `threshold` filters first (`<=` keeps), **then** the level is sorted, **then** it is
  numbered — pruned branches never consume an id.
- The sort key is the child's heuristic value, **stably** ascending: ties keep generation
  order.
- An id reads `<parent id>,<position in this level>`, the root is `0`, and the position is
  the 0-based index **after** filtering and sorting.

## Encoding: what each line says

Entering a node writes the node itself first:

```
Current State: <target>:[<numbers...>], Operations: [<operations...>]
```

Numbers and operations use Python literal form (`[1, 2, 3]`, `['2*3=6']`, a space after
each comma, single quotes around operations); the root's operation list is `[]`.

Children are then tried in sorted order; for each child:

- First write `Exploring Operation: <op>, Resulting Numbers: [<merged numbers...>]`.
- A child with one number left: if it equals `target`, write
  `<n>,<target> equal: Goal Reached` and **stop the whole search at once**; otherwise
  write `<n>,<target> unequal: No Solution` and **do not descend**.
- Otherwise write `Generated Node #<id>: <target>:[<numbers...>] Operation: <op>`, then
  `Moving to Node #<id>`, then recurse; if the recursion reports a hit, return all the way
  out.
- When the child is not the last of its level, two more lines precede the next one:
  `Moving to Node #<parent id>` and the **parent's own** `Current State` (not the
  sibling's state).

Every line ends with a newline, including the last one. If the whole tree is exhausted
without a hit the trace simply stops — the upstream `if open_set:` backtrack hint at the
end of `dfs` is **dead code** under recursion, so a trace may end on an `unequal` line with
no closing marker.

## Decoding: verdicts

Decode the trace back into an operation sequence and replay it:

1. Take the **first line** of `trace.strip()`; it must start with
   `Current State: <digits>:[<digits>]` (anchored at the beginning). `target` only accepts
   unsigned digits and the number list is split on `", "`.
2. Find the **first** `<n>,<n> equal: Goal Reached` in the whole text.
3. Take the last `Exploring Operation: <op>, Resulting Numbers: [...]` **before** it as
   "the step that reached the goal"; its right-hand side must equal the first line's
   `target`.
4. Take the `Operations: [...]` of the **last** `Current State` before it (that is the
   goal step's parent), split it into operations, append the goal step itself, and replay
   in order: each left-hand side must evaluate correctly and every number used must be
   "currently available".
5. The available pool **only grows**: it starts as the first line's numbers, gains each
   result, and **never drops** a consumed number — so the same number can be reused
   without complaint.

The verdict is exactly one of these strings:

```
Invalid input: Cannot find the initial state in the first line.
No goal reached statement found.
No operations found leading to the goal.
Invalid path: Final operation does not result in target.
Could not split operation into lhs, rhs
Invalid operation: <op>
Invalid operation: <op>, number <n> not available in [<pool...>]
Error in evaluating operation <op>: <reason>
Valid path.
```

- Only `Valid path.` counts as valid.
- A left-hand side must be the two-operand form `a<op>b` (`+ - * /`, both sides plain
  non-negative decimals); anything else lands in `Error in evaluating operation ...`.
- Division is **true division**, so a rounded write-up like `3/2=1` is
  `Invalid operation`.
- The pool is printed as "first line's numbers + every earlier result", in order, keeping
  duplicates.

## Rating

- `explored` = how many times `Exploring Operation` appears; `score` = `explored + 1`
  (the `+1` is the root).
- `max_nodes` looks up **the number of numbers parsed from the first line**: 2→4, 3→48,
  4→1152, 5→46080; any other count gives `None`.
- Valid with a table entry: `rating = max(1 - score / max_nodes, 0.0)`; otherwise
  `rating = 0.0`. So a search that explores more than a full tree is clamped to 0, and a
  two-number instance can be clamped to 0 even when it hits on the first move.

## Output

```
{
  "trace": "<the whole trace>",
  "encoded": true,
  "found": true,
  "states": 2,
  "moves": 1,
  "explored": 2,
  "node_ids": ["0,0"],
  "ops": ["2*3=6", "1*6=6"],
  "final": {"value": 6, "target": 6, "equal": true},
  "verdict": "Valid path.",
  "score": 3,
  "max_nodes": 48,
  "rating": 0.9375
}
```

- `states` / `moves` / `explored` count occurrences of `Current State` /
  `Moving to Node` / `Exploring Operation`, all counted on the trace text.
- `node_ids` holds the id of every `Generated Node #<id>`, in written order.
- `ops` holds every `Exploring Operation` operation, in written order.
- `final` is parsed from the **last** `equal: Goal Reached` / `unequal: No Solution` line
  as `{value, target, equal}`; `None` when there is none.
- `found` only asks whether `Goal Reached` appears in the text.

## Example

```
led = solve(6, [1, 2, 3], heuristic="sum", threshold=6)
led["trace"]
# Current State: 6:[1, 2, 3], Operations: []
# Exploring Operation: 2*3=6, Resulting Numbers: [1, 6]
# Generated Node #0,0: 6:[1, 6] Operation: 2*3=6
# Moving to Node #0,0
# Current State: 6:[1, 6], Operations: ['2*3=6']
# Exploring Operation: 1*6=6, Resulting Numbers: [6]
# 6,6 equal: Goal Reached
led["ops"]        # ["2*3=6", "1*6=6"]
led["node_ids"]   # ["0,0"]
led["score"]      # 3
led["rating"]     # 1 - 3/48
```

Counter-example: with two numbers a first-move hit produces a perfectly legal trace, yet
the goal step's parent (the root) has an empty operation list, `"".split(", ")` yields one
empty string, and the verdict becomes `Could not split operation into lhs, rhs` with
rating 0 — that is the upstream behaviour, not a typo.

## Constraints

- Pure function: no network, no file I/O, no mutation of the inputs; identical inputs must
  produce an identical ledger.
- `rating` uses ordinary division; comparisons use a `1e-12` tolerance.
- An unknown `heuristic`, a `nums` count outside 2–5, a negative number or `target`, or
  `mult` with `target < 2` → `ValueError`. These checks run **only while encoding**.
- With `trace` given, none of the argument checks run and the trace is never modified.
