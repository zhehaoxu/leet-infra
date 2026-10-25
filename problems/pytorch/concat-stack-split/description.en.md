# Concat (along an existing dimension)

## Problem

`torch.cat` joins several tensors end to end along an **existing dimension**: the lengths along that dimension add up, and all remaining dimensions must match exactly. It **does not add a dimension** — this is exactly the key difference from `torch.stack` (which creates a new dimension).

$$ \text{out}[i_0,\dots, i_{dim},\dots] = \text{t}_k[\dots, i_{dim}-\text{offset}_k, \dots] $$

## Rules

- Implement `solve(tensors, dim=0)`; `tensors` is a group of tensors with the same rank and consistent shapes.
- The output has the same rank as the inputs; the length of dimension `dim` equals the sum of the corresponding lengths of all tensors; **every other dimension must be equal**, otherwise an error should be raised.
- The concatenation order is the list order; the order affects the result.
- Support negative `dim` (`-1` means the last dimension).
- Match `torch.cat(tensors, dim)`.

## Examples

```
Input: a=[[1,2]], b=[[3,4],[5,6]], dim=0
   dim0 length 1+2=3, dim1 stays 2
Output: [[1,2],
      [3,4],
      [5,6]]     # shape (3,2), still 2-D (not stack's (2,1,2))
```

## Constraints

- All inputs share the same dtype and the same rank; float32/float64
