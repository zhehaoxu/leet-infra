# Padding Operator (Constant Padding)

## Problem

Fill elements at tensor boundaries. `torch`/`F.pad`'s `padding` is a sequence given **starting from the last dimension, in pairs (left, right)**, whose order is the opposite of intuition and is the most common pitfall.

$$ \text{padding} = (\ell_{-1}, r_{-1}, \ell_{-2}, r_{-2}, \dots) $$

That is, the **last two elements** pad the last dimension, the next two moving backwards pad the second-to-last dimension, and so on. This level implements only the `constant` mode (new positions are filled with `value`).

## Rules

- Implement `solve(x, padding, mode="constant", value=0.0)`.
- `padding`'s length must be even, and its elements are interpreted in the "last dimension first, (left, right)" order above; each dimension's new length = original length + left + right.
- Only `mode="constant"` is supported; any other `mode` must be rejected with an error.
- Must match `F.pad(x, padding, mode="constant", value=value)`.

## Examples

```
Input: x=[[1,2],[3,4]], padding=(1,1,1,1)   # last dim W 1 on each side, second-to-last dim H 1 top/bottom
Output: [[0,0,0,0],
      [0,1,2,0],
      [0,3,4,0],
      [0,0,0,0]]

Input: x=[[1,2],[3,4]], padding=(1,0)        # only pad 1 column on the left of the last dimension
Output: [[0,1,2],
      [0,3,4]]
```

## Constraints

- float32/float64 input; each element of `padding` is $\ge 0$
