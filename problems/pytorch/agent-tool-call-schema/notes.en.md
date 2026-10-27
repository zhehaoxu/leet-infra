# Solution Notes · Tool Argument Validation

## Why it deserves its own problem

This layer fails in a very uniform way: either too loose (`"3"` travels all the way into `range()` and explodes hundreds of lines later) or too strict (the model omits one optional field and the whole turn is retried). Production tuning turns the same few knobs: which types are repairable, whether defaults are filled, whether extra keys are errors or dropped. Written as a pure function, those knobs become test cases.

## Validation order decides the error text

Three classes of problem can appear at once: a missing field, a bad type, an extra key. If errors were reported by iterating `arguments`, the same input could produce differently ordered messages depending on dict insertion order, and the model's retry would see a jittering list. So the order is fixed:

1. missing `required` fields (in `required` order);
2. present properties (in `properties` insertion order, with nested paths built level by level);
3. extra keys (in `arguments` insertion order).

In other words the schema shape drives the error order, and the argument shape only matters in the last stage.

## Repairable versus not

- **Repairable**: `integer` receiving `3.0`. JSON does not distinguish integers from floats, so `3.0` and `3` are the same thing in the wild and normalising to `int` is a fix.
- **Not repairable**: `"3"` receiving `integer`. It *can* be coerced, but the failure rate is high (`"3.0"`, `"3kg"`, `"three"`), and a silent conversion hides the error.
- **Must reject**: `bool` receiving `integer`. In Python `True == 1`, so whoever calls `isinstance` first turns a "switch" into a "quantity" — the most classic incident of all.

## The enum trap: `==` is not enough

With `enum=[0, 1]`, a `True` matches if you only compare `==`. JSON Schema means "the value equals one of the entries and the types are compatible", so the check is `type(item) is type(value) and item == value`. Likewise once `number` normalises to `float`, `0.0` no longer matches the ints `[0, 1]` — harsh-looking, but it makes "the enum is integer literals" unambiguous.

## Two details about defaults

- **A missing optional property is filled**, so downstream code need not write `args.get("k", 1)` everywhere;
- **A required property that carries a `default` is treated as filled**: in a real schema that is a mistake, but since a default exists, absorbing it beats forcing a whole retry.

## Related problems

- `agent-react-loop`: a validation failure becomes an error observation back to the model — exactly that problem's "unknown tool" path.
- `agent-context-compaction`: tool schemas consume context, so more tools mean a tighter budget.
- `agent-task-dag-schedule`: arguments resolved first, then multi-task orchestration.
