# Solution Notes · ReAct Loop

## Why this layer deserves its own problem

Model "capability" is not judgeable; the runtime "protocol" very much is — fixed text in, fixed tuple out. It is also where production incidents concentrate: the parser misses a new format, a hallucinated tool name crashes the loop, the run only stops when `max_steps` expires, a misplaced `Observation` makes the model repeat itself. Reduced to a pure function, every edge case can be pinned by a handful of assertions.

## Four termination sources, not one

1. The model emits a `Final Answer` (the normal exit).
2. A turn has neither an Action nor an answer — the model is thinking but no longer acting, and looping again just burns budget.
3. Two consecutive turns share the same `(action, input)` — that is not progress, that is a loop.
4. `max_steps` is exhausted, or the script runs out — the outer backstop.

Implementing only 1 and 4 is the common mistake, and it silently drains the budget whenever the model forgets its Action. Case 3 is subtle: both halves must match; the same tool with a different argument is normal multi-step investigation.

## Parsing is harder than it looks

- **Whitespace after the colon**: both `Action Input: Beijing` and `Action Input:Beijing` must yield `Beijing`, so split on the colon and strip; fixed-column slicing breaks.
- **Case**: models occasionally emit `action:`; matching should be case-insensitive (this problem normalises to lowercase).
- **Distractor lines**: a `Thought` may quote "Action: it should be weather" as an example. Anchoring on the line prefix filters out nearly all such false positives — which is what real implementations do.
- **`Action` without `Action Input`**: real frameworks raise and let the caller retry; this problem raises `ValueError`, because silently passing `""` to the tool produces a much nastier incident.

## Unknown tools and the loop guard

Both are "let the agent correct itself" rather than "crash the runtime": the error observation goes back to the model and the next turn usually fixes it. The guard differs in that it fires **before** execution — a repeated action is not really invoked again (it may have side effects, e.g. placing the same order twice), only recorded.

## What steps counts

`steps` is the number of model turns consumed, therefore:

- a first-turn answer gives `steps == 1` with no observations;
- a repeat guard on turn 2 gives `steps == 2` while the tool ran once;
- `max_steps == 0` consumes nothing, useful for testing "can we still answer after the budget is gone".

## Related problems

- `agent-tool-call-schema`: the validation layer once Action Input becomes JSON arguments.
- `agent-task-dag-schedule`: from single actions to multi-task dependency orchestration.
- `agent-context-compaction`: longer loops fill the context, and the compaction policy decides whether the run survives.
