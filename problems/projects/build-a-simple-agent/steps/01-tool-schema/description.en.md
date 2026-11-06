# Tool Schema Validator

## Background

The first step of an agent is not 'let the model talk' but turning 'is this tool call legal' into an assertable pure function. Model output is a fixture; the validator takes only `spec` and `args`.

## Task

Validate `args` against `spec`, fill in defaults, report three classes of problems, and return a normalized call dictionary.

Implement `solve(spec, args)`.

## Input

See the example and rules below; every "model output" or "tool result" is an explicit argument (fixture). The sandbox has no model, no network, no randomness and no clock.

## Output

A dictionary that can be compared field by field; see the example for keys and values.

## Rules

- Return `{"ok", "tool", "missing", "unexpected", "type_errors", "normalized"}`; the three problem lists are sorted.\n- Missing required keys go to `missing`; with `strict=True` extra keys go to `unexpected`; wrong-typed keys go to `type_errors` and are excluded from `normalized`.\n- Declared params with a `default` are filled in when absent; `type` supports `any/str/int/float/number/bool/list/dict`, and `bool` is not an `int`.\n- `ok = name is non-empty and all three problem lists are empty`; `normalized` contains only declared, correctly typed keys.

## Example

```python
spec = {"name": "search", "strict": True,
        "params": {"query": {"type": "str", "required": True},
                   "top_k": {"type": "int", "required": False, "default": 3}}}
solve(spec, {"query": "hello"})
# {"ok": True, "tool": "search", "missing": [], "unexpected": [],
#  "type_errors": [], "normalized": {"query": "hello", "top_k": 3}}

```

## Constraints

- Pure function: do not mutate inputs, read global state, or raise; treat empty `args` as `{}`.\n- The sandbox has only the standard library plus torch/numpy; no tensors are needed here.
