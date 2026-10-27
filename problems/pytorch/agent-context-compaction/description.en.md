# Agent Infrastructure: Context Window Compaction under a Priority Budget

## Problem

The longer an agent runs, the fuller the context: system prompt, full history, every tool schema, the scratchpad, intermediate tool results. It hits the window limit fast. Truncating the tail is the easiest fix and the worst one — it throws away the system constraints and tool contracts, and the model immediately starts misusing tools. Production implementations therefore pick segments by priority: pinned ones first, then a greedy fill within the leftover budget.

(LangChain's `trim_messages`, Claude Code's auto-compact and OpenHands' condenser are all dials on the same idea.)

## Task

Implement `solve(segments, budget, pinned_priority=100)` returning `(kept, used)`:

- `segments`: `[(name, tokens, priority)]` in chronological order; `budget`: total tokens the window allows; `pinned_priority`: any segment with priority `>=` this value is **pinned** (must be kept).
- Pinned segments consume the budget first; if their total already exceeds `budget` → `ValueError`.
- The rest are tried greedily in `(priority desc, original index asc)` order: keep it if it fits, otherwise **skip it** (skipping is not terminating — later, smaller segments must still be tried).
- `kept` is returned in ascending **chronological** index order, and `used` is the token total of the kept segments.

## Example

```
segments = [
    ("system", 40, 100),
    ("history", 120, 10),
    ("tools", 60, 60),
    ("scratchpad", 30, 40),
]

solve(segments, 200) -> ([0, 2, 3], 130)
```

The fill order is system(40) → tools(60) → scratchpad(30) → history(120, skipped because it no longer fits); 40+60+30=130 tokens are kept and the chronological indices are `[0, 2, 3]`.

## Constraints

- **Pinned segments are incompressible**: they carry the system constraints and the tool contract, so dropping them changes the task itself; overflowing the budget must be an explicit error, not a silent drop.
- **Skip the big block and keep trying**: `break` lets one oversized history segment eat every remaining slot, while `continue` actually fills the budget.
- **Return chronological order, not priority order**: assembling the prompt must preserve conversational time order; priority is only used for selection.
- **Equal priorities go to the lower index**: otherwise the same input yields different keep-sets across implementations.
- **Invalid input raises `ValueError`**: `segments` not a `list`, a negative `budget`, negative or non-integer `tokens`, or a segment that is not a triple.
