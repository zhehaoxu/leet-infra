# Step Notes

## Done Before

The project opening page already shows the milestone dependency graph; this is the first link, with no earlier official module.

## Introduced Here

Define 'call legality' as six field-by-field comparable return items, with L2 invariants (normalization is an idempotent fixed point, the output key set is bounded) covering inputs that cannot be enumerated.

## What the Next Step Does With It

The ReAct loop in the next step calls `from scaffold.tool_schema import validate` to decide whether each action may execute; your implementation is never replayed.
