# Scripted Multi-Turn End-to-End Acceptance

## Background

The first four blocks are individually gradable, but an agent's value lies in the whole data flow. The final step replays a scripted multi-turn dialogue and asserts each turn's tool-call sequence, failure breakpoint, final answer and ledger size -- still entirely explicit arguments.

## Task

Assemble `run_dialog` from the four frozen modules: run the loop per turn, count valid/invalid calls, compute the retry breakpoint, append and compact the ledger, then aggregate a multi-turn report.

Implement `solve(turns, tools, policy)`.

## Input

See the example and rules below; every "model output" or "tool result" is an explicit argument (fixture). The sandbox has no model, no network, no randomness and no clock.

## Output

A dictionary that can be compared field by field; see the example for keys and values.

## Rules

- Each `turn` has `user`, `script` (fixture replies) and `outcomes` (fixture tool results).\n- A per-turn report holds `turn`, `user`, `tool_calls` (valid calls only), `invalid_calls`, `breakpoint`, `breaker_open`, `final`.\n- The ledger appends `"<comma-joined tool names>|<final or empty>"` each turn and compacts with `keep_last` / `max_chars`; `policy.retry` is passed to the step-04 planner.\n- Return `{"turns", "final_answer", "total_tool_calls", "compactions", "ledger_chars"}`; `final_answer` is the last turn's `final`.

## Example

```python
turns = [{"user": "find a",
          "script": [{"thought": "x",
                      "action": {"tool": "search", "args": {"query": "a"}}},
                     {"thought": "y", "final": "A"}],
          "outcomes": ["ok"]}]
solve(turns, tools, policy)
# {"turns": [{"turn": 1, "user": "find a", "tool_calls": ["search"],
#             "invalid_calls": 0, "breakpoint": "success",
#             "breaker_open": False, "final": "A"}],
#  "final_answer": "A", "total_tool_calls": 1,
#  "compactions": 0, "ledger_chars": 8}

```

## Constraints

- Passing this step unlocks the project overview page; progress is recorded separately in `data/project_progress.json`.\n- No training and no convergence judgement; no model, network or clock calls, and repeated calls must agree.
