# Retry Backoff and Circuit Breaker Budget

## Background

Tools fail. Handing failure to real time or random jitter would destroy determinism. Backoff must be a **computed numeric plan**: every outcome comes from a fixture, with no sleep and no clock.

## Task

Reuse the frozen loop to obtain this round's valid tool sequence, then compile `outcomes` into a retry/backoff/breaker plan under `policy`.

Implement `solve(script, tools, outcomes, policy)`.

## Input

See the example and rules below; every "model output" or "tool result" is an explicit argument (fixture). The sandbox has no model, no network, no randomness and no clock.

## Output

A dictionary that can be compared field by field; see the example for keys and values.

## Rules

- Read `outcomes` in order: `"ok"` succeeds immediately; anything else is a failure and appends `base_delay_ms * factor**len(delays)` to the plan.\n- If cumulative backoff would exceed `budget_ms`, stop at `budget`; if attempts run out, stop at `attempts`; if the fixture runs out first, stop at `outcomes`.\n- When the number of recent failures within `window` reaches `breaker_threshold`, open the breaker and stop at `breaker`; the breaker takes priority over backoff.\n- Return `{"actions", "terminated_by", "retry"}`; `retry` holds `attempts`, `delays`, `total_delay_ms`, `breaker_open`, `succeeded`, `stopped_by`.

## Example

```python
policy = {"max_steps": 6, "max_attempts": 4, "base_delay_ms": 10,
          "factor": 2, "budget_ms": 100,
          "breaker_threshold": 2, "window": 3}
solve(script, tools, ["error", "error", "ok"], policy)
# {"actions": ["search"], "terminated_by": "final",
#  "retry": {"attempts": ["error", "error"], "delays": [10],
#            "total_delay_ms": 10, "breaker_open": True,
#            "succeeded": False, "stopped_by": "breaker"}}

```

## Constraints

- `actions` contains only tool names from trace items with `valid=True`, in order; breaker/budget affects the plan only, never the trace.\n- Do not use `time`, `random` or `sleep`; repeated calls must agree.
