# L4 Design Review: Why Stateful Judging Is Forbidden

## Background

The last step is not about writing logic but about the whole picture: can you state the six stages in order, and which decisions are explicitly forbidden in this project? The answer is fully deterministic and still auto-gradable.

## Task

Use the frozen `scaffold.dialog_runner` stage vocabulary and audit function to decide whether a decision list and a data-flow trace are compliant.

Implement `solve(decisions, trace)`.

## Input

See the example and rules below; every "model output" or "tool result" is an explicit argument (fixture). The sandbox has no model, no network, no randomness and no clock.

## Output

A dictionary that can be compared field by field; see the example for keys and values.

## Rules

- The official stage order is `validate → loop → ledger → retry → dialog → review`.\n- A forbidden decision (`replay_user_code`, `judge_convergence`, `call_llm`) gives `reason="forbidden"`; a wrong order gives `reason="order"`; a trace differing from the official stages gives `reason="trace"`; full compliance gives `reason="ok"`.\n- Priority: forbidden items > order > trace.\n- Return `{"accepted", "reason", "stages", "trace_len"}`; `stages` is always the official six, `trace_len = len(trace)`.

## Example

```python
stages = ["validate", "loop", "ledger", "retry", "dialog", "review"]
solve(stages, stages)
# {"accepted": True, "reason": "ok",
#  "stages": [...], "trace_len": 6}
solve(stages + ["replay_user_code"], stages)
# {"accepted": False, "reason": "forbidden", ...}

```

## Constraints

- This is L4 design review, not a quiz UI: it is still graded as an ordinary problem returning a dictionary from `solve`.\n- Do not rewrite the official stage vocabulary; repeated calls must agree.
