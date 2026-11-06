# Memory Ledger and Deterministic Compaction

## Background

Loop traces grow without bound. Writing them into a ledger and compacting deterministically at a threshold keeps long dialogues bounded, replayable and reproducible. The summary text is built from the entries themselves, with no model call.

## Task

The loop comes from frozen `scaffold.react_loop`; **the ledger's append / compact / total_chars are this step's missing block and must be implemented in this file**. Finally return stop info, the ledger, the compaction count and total characters.

Implement `solve(script, tools, policy)`.

## Input

See the example and rules below; every "model output" or "tool result" is an explicit argument (fixture). The sandbox has no model, no network, no randomness and no clock.

## Output

A dictionary that can be compared field by field; see the example for keys and values.

## Rules

- Each trace item becomes `{"turn": step, "role": type, "text": ...}`: `action` uses `action:<tool>`, `final` uses `final:<answer>`, anything else uses `noop:`.\n- After each append call `compact(ledger, keep_last, max_chars)`: when the entry count exceeds `keep_last`, fold the oldest entries into one `role="summary"` entry whose text is `[N]<joined>` truncated to `max_chars*4`.\n- Return `{"terminated_by", "final", "ledger", "compactions", "ledger_chars"}`; `ledger_chars` is the sum of final ledger text lengths.\n- `policy` supplies `max_steps`, `keep_last`, `max_chars`.

## Example

```python
policy = {"max_steps": 6, "keep_last": 2, "max_chars": 24}
solve(script, tools, policy)
# {"terminated_by": "final", "final": "answer-1",
#  "ledger": [{"turn": 1, "role": "summary", "text": "[1]action:search"}, ...],
#  "compactions": 1, "ledger_chars": 43}

```

## Constraints

- Only earlier official modules are injected (`tool_schema`, `react_loop`); implement the three ledger functions yourself, keep no module-level ledger, and make repeated calls agree.\n- This step's reference implementation is frozen as `scaffold/memory_ledger.py` for step 05; `keep_last=0` compacts all entries and `max_chars` is a positive integer.
