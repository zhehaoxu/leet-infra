# Solution Notes · Capacity Dispatch and Token Dropping

## Meaning of capacity_factor

$C = \lceil \alpha \cdot T \cdot k / E \rceil$. Under an ideally uniform dispatch each expert receives about $T \cdot k / E$ tokens, so $\alpha=1$ is "exactly enough for the average"; $\alpha<1$ deliberately drops a fraction; $\alpha>1$ allows skew.

The Mixtral/GShard papers commonly use $\alpha \in \{1.0, 1.25\}$.

## position vs probs drop policy

Megatron offers two:
- `position` (this implementation): keep the first $C$ tokens in their original batch order. The semantics are intuitive and reproducible, and this is Shazeer OLNN's original definition.
- `probs`: take the top-$C$ of routing_probs along dim=0 (the token dimension), keeping the $C$ most confident assignments. What gets dropped is the "less certain" assignments.

Grading uses position.

## Why drop before renorm / drop after renorm?

This implementation first **renorms** across the $k$ selected values (summing to 1) and then drops by position. This means the total weight "summed after dropping" is less than 1. This is the standard top-k → renorm → drop order, consistent with Megatron.

If you wanted "drop then renorm", the weight of a dropped slot would be redistributed to the retained slots, which is different semantics. **This implementation does not renorm** a second time.

## dropped is an integer

`int(routing_map.sum() - final_map.sum())`. The test assertions use a Python `int`, not a 0-dim tensor.
