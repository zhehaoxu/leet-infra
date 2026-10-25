# Fully Connected Linear Layer (Linear / Affine Layer)

## Problem

The linear layer is the most common network building block: $y = xW^\top + b$. Do not be fooled by the one-line `nn.Linear` call—
internally it is "each output unit = the dot product of the input vector with that unit's weight row, plus the bias". Understanding its
**weight orientation convention** (`weight` has shape `(out_features, in_features)`, not the transposed one) is the first lesson in reading code, deriving gradients, and writing fused kernels.

## Rules

- Implement `solve(x, weight, bias=None)` with `x` of shape `(..., in_features)`,
  `weight` of shape `(out_features, in_features)`, and `bias` of shape `(out_features,)`.
- Output shape `(..., out_features)`: take the dot product of the last dimension of the input with each `weight[o]`.
- Support `x` with arbitrary leading batch dimensions (e.g. `(N, L, in)`).
- The result must match `F.linear(x, weight, bias)` (i.e. `x @ weight.T + bias`).

## Examples

```
Input: x = [[1, 2, 3],
            [4, 5, 6]]
       weight = [[1, 0, 2],
                 [-1, 3, 0]], bias = [10, -1]
  row 0: [1·1+2·0+3·2+10, 1·(-1)+2·3+3·0-1] = [17, 4]
  row 1: [4·1+5·0+6·2+10, 4·(-1)+5·3+6·0-1] = [26, 10]
Output: [[17, 4],
         [26, 10]]
```

## Constraints

- float32 inputs; `bias` may be `None`
- The key points are the `(out, in)` storage convention of `weight` and the "dot product + bias" semantics
