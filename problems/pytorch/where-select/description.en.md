# Where Conditional Selection (Broadcasting Ternary)

## Problem Statement

`torch.where(condition, x, y)` is an element-wise ternary operation: where `condition` is `True` it takes $x$, otherwise it takes $y$. The three operands may have different shapes, but they must be able to **broadcast** to a common shape.

$$ \text{out}[i] = \begin{cases} x[i] & \text{condition}[i] = \text{True} \\ y[i] & \text{otherwise} \end{cases} $$

## Rules

- Implement `solve(condition, x, y)`, where `condition` is a boolean tensor (or one interpreted by truth value), and `x`/`y` may be tensors or scalars.
- `condition`, `x`, and `y` are aligned to a common shape by the **standard broadcasting rules**, then selected element-wise.
- dtype follows the type promotion of `x`/`y` (mixing integers and floats yields a float output).
- Behaves the same as `torch.where`.

## Example

```
Input: condition=[True,False,True,False], x=[1,2,3,4], y=[-1,-2,-3,-4]
Output: [1, -2, 3, -4]      # take x where condition is true, y where it is false
```

## Constraints

- Any broadcastable shape; a broadcasting failure must raise an error
