# Collective Communication: Reduce-Scatter

## Problem

**Reduce-Scatter** = first perform an element-wise **reduction** over all ranks' data (summation here), then **slice and distribute** the reduction result: rank `i` receives only the `i`-th segment. It is the first half of Ring All-Reduce and the collective behind "each GPU keeps only $1/W$ of the gradients" in ZeRO.

> Collectives are modeled as pure functions: the input is a list of "each rank's local data before communication" (the $r$-th item is rank $r$'s tensor), and the output is a list of "each rank's data after communication" (the $r$-th item is the segment rank $r$ receives).

## Rules

- Implement `solve(inputs)`, where `inputs` is a list of length $W$ and `inputs[r]` is rank `r`'s one-dimensional tensor; all ranks have **equal length**, and the length is divisible by $W$.
- Two steps:
  1. **Reduce**: element-wise sum $T = \sum_{r=0}^{W-1} \texttt{inputs[}r\texttt{]}$ (length $L$).
  2. **Scatter**: split $T$ evenly into $W$ segments (`chunk = L / W` each); rank `i` receives the `i`-th segment
     $T[i\cdot\text{chunk} : (i+1)\cdot\text{chunk}]$.
- Return a **list of length $W$**, where the `i`-th item is the segment assigned to rank `i` (**ranks get different results**, which is exactly the difference from All-Gather).
- Do not modify the input tensors.

## Examples

```
Input: inputs = [ [1,2,3], [4,5,6], [7,8,9] ]   # W=3, each length 3, chunk=1
  Reduce  T = [1+4+7, 2+5+8, 3+6+9] = [12, 15, 18]
  Scatter rank0←T[0:1]=[12], rank1←T[1:2]=[15], rank2←T[2:3]=[18]
Output: [ [12], [15], [18] ]
```

## Constraints

- `1 ≤ W ≤ 64`; each rank is one-dimensional, of equal length, and the length is divisible by $W$
- Return `list[torch.Tensor]` of length $=W$, each shaped $(L/W,)$
- Equality is checked segment by segment with `torch.equal`; note the rank order: the `i`-th item corresponds to the `i`-th segment of $T$
