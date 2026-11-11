# ReAct Loop State Machine

## Background

Model replies are not reproducible, but 'given this sequence of replies, how should the loop behave' is fully decidable. Externalise model output into the `script` argument and the loop is a pure state machine.

## Task

Consume at most `max_steps` fixture replies and produce a per-step trace, a stop reason and the final answer; each action is checked with the frozen step-01 validator.

Implement `solve(script, tools, max_steps)`.

## Input

See the example and rules below; every "model output" or "tool result" is an explicit argument (fixture). The sandbox has no model, no network, no randomness and no clock.

## Output

A dictionary that can be compared field by field; see the example for keys and values.

## Rules

- Each `script` item has `thought`; an `action` key means a tool call, a `final` key means a terminating reply (even an empty string terminates).\n- Trace items have `type` = `action` / `noop` / `final`; action items also carry `tool`, `args`, `valid`.\n- `valid = tool name is in tools and validate(spec, args)["ok"]`; invalid actions are recorded but never executed.\n- `terminated_by` is `final` (a final reply was seen) / `budget` (budget consumed and the script still has items) / `script_end` (script exhausted first); `steps_used = len(trace)`.

## Example

```python
tools = {"search": {"name": "search", "strict": True,
                    "params": {"query": {"type": "str", "required": True}}}}
script = [{"thought": "need info",
           "action": {"tool": "search", "args": {"query": "a"}}},
          {"thought": "done", "final": "answer-1"}]
solve(script, tools, 6)
# {"trace": [...], "steps_used": 2,
#  "terminated_by": "final", "final": "answer-1"}

```

## Constraints

- Official modules are injected read-only: `from scaffold.tool_schema import validate`; never import your own earlier code.\n- Do not mutate `script` / `tools`; repeated calls with identical arguments must return identical results.
