# Gather Index Selection

## Problem

`gather` selects values element-wise from `input` using the values in `index` along the specified dimension `dim`, and the output has the same shape as `index`. It is the basis for top-k, positional gathering, and similar scenarios.

Taking a 3-D example with `dim=1`:

$$ \text{out}[i][j][k] = \text{input}[i][\ \text{index}[i][j][k]\ ][k] $$

## Rules

- Implement `solve(input, index, dim)`; `index` has the **same number of dimensions** as `input`, and the length of `index` along each non-`dim` dimension is $\le$ the corresponding dimension of `input`; `index` is an integer tensor.
- Output shape = `index.shape`.
- **dim semantics trap**: only the axis `dim` is replaced by `index` for value selection; all other axes keep their own coordinates. For example, with `dim=0`, $\text{out}[i][j][k]=\text{input}[\text{index}[i][j][k]][j][k]$.
- Support negative `dim` (`-1` means the last dimension).
- Exactly consistent with `torch.gather`.

## Examples

```
Input: input=[[10,20,30],[40,50,60]], index=[[2,0,1]], dim=1
   row 0 selects columns by index [2,0,1] → [30,10,20]
Output: [[30,10,20]]
```

## Constraints

- float32/float64 data + int64 indices
- Every value of `index` lies within the corresponding dimension
