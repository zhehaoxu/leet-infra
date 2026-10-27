# Agent Infrastructure: Tool-Call Argument Validation and Normalisation

## Problem

With function calling, a model's "action" is no longer a string but structured arguments: `{"city": "Beijing", "unit": "c"}`. Those JSON payloads are **generated**, so they drift — a required field goes missing, an integer is emitted as a string, an enum becomes a synonym, an undeclared key sneaks in. Hand the dict straight to the tool and the failure surfaces downstream as a `TypeError` or, worse, as silently wrong behaviour.

Every agent framework therefore has a schema layer (OpenAI `strict: true`, LangChain's Pydantic wrapper, MCP's `inputSchema`). This problem turns that layer into a pure function: validate against the schema, repair what is repairable (`3.0 → 3`), fill defaults, and report errors in a stable order.

## Task

Implement `solve(schema, arguments)` returning `(normalized, errors)`:

- **Required**: a missing field in `required` reports `$.<name>: missing required property`; if that property carries a `default`, it is filled instead of reported.
- **Types**: `string` / `boolean` are strict; `integer` accepts `int` and integral `float` and normalises to `int`, rejecting `bool`; `number` accepts `int` / `float` and normalises to `float`, rejecting `bool`; `array` recurses via `items` (path `$.tags[i]`); `object` recurses via nested `properties` / `required` (path `$.meta.days`).
- **Enum**: `enum` matches on both type and value, otherwise reports `$.<name>: value <repr> is not one of <list>`.
- **Defaults**: a missing non-required property with a `default` is filled in.
- **Extra keys**: with `additionalProperties` set to `False`, report `$.<name>: unexpected property`; otherwise keep them as-is.
- Fields that fail validation are omitted from `normalized`; error order is "missing required, then property insertion order, then extra keys".

## Example

```
schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "default": 1},
        "unit": {"type": "string", "enum": ["c", "f"]},
    },
    "required": ["city", "unit"],
}
arguments = {"city": "Beijing", "unit": "c", "days": 3.0}

solve(schema, arguments) -> ({"city": "Beijing", "days": 3, "unit": "c"}, [])

solve(schema, {"city": 42, "unit": "k"}) -> ({"days": 1}, [
    "$.city: expected string, got int",
    "$.unit: value 'k' is not one of ['c', 'f']",
])
```

## Constraints

- **`bool` subclasses `int`**: `isinstance(True, int)` is true, so `integer` must exclude `bool` explicitly, otherwise `True` reaches the tool as 1.
- **Enums need types too**: `0 == 0.0 == False` all hold in Python, so comparing with `==` alone lets `True` match `enum=[0, 1]`; `type` must be compared as well.
- **The most common drift**: writing `3` as `3.0`, or `true` as `"true"`. The former is repairable (normalise to `int`), the latter is not (report and let the model retry) — that boundary must be drawn deliberately.
- **Error paths must be locatable**: `$` is the root, `.` a field, `[i]` an array index, concatenated through nesting.
- **Invalid input raises `ValueError`**: `schema` / `arguments` not a `dict`, `properties` not a `dict`, `required` not a `list`.
