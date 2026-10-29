# Einsum Spec Evaluator: Label String to Multiply-Reduce

## Problem Description

`torch.einsum` uses a single "label string" to express matrix product, batched matrix product, transpose, outer product and per-axis summation in one notation: a letter names an axis, letters shared by several operands are batch/dimensions, and letters that appear in the inputs but not in the output are summed away. `torch.einsum("bhqd,bhkd->bhqk", q, k)` in attention code is its most common appearance.

$$\text{out}[\mathcal{O}] = \sum_{c \in \mathcal{C}} \prod_{k} T_k[\text{slice aligned by labels}]$$

where $\mathcal{C}$ is the set of letters "present in the inputs but absent from the output" (contracted axes) and $\mathcal{O}$ is the output label string. This problem implements a restricted subset of that grammar: explicit `->`, 1–3 operands, no repeated letter within an operand (no trace/diagonal), no ellipsis.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Semantics must match `torch.einsum` pointwise inside the supported subset: `"ij,jk->ik"` is a matrix product, `"bij,bjk->bik"` a batched one, `"bhqd,bhkd->bhqk"` attention scores, `"ij->ji"` a transpose, `"ij->i"` a row sum, `"i,i->"` a dot product.
- When every output letter is contracted away (e.g. `"i,i->"`) the result is a **0-dimensional** tensor; a pure permutation such as `"ij->ji"` has no contraction at all.
- A letter appearing twice inside one operand (`"ii->i"`) is a diagonal/trace operation, **not supported** here and must raise ValueError; likewise `"..."`, a missing `->`, an output letter absent from the inputs, a label count that disagrees with the tensor rank, and more than 3 operands all raise ValueError.
- float64 accumulation internally is fine; the returned dtype equals that of the first operand (float32).

## Example

$X$ is $3\times 2$, $Y$ is $2\times 3$, spec is `"ij,jk->ik"` (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
X = [[1, 2],      Y = [[1.0, 0.0, 2.0],
                      [0.5, 1.0, 0.0]]
     [3, 4],
     [5, 6]]
out(ij,jk->ik) = [[2.0, 2.0, 2.0],
                  [5.0, 4.0, 6.0],
                  [8.0, 6.0, 10.0]]
```

For $\text{out}[0,2]$: $j$ is the eliminated contraction letter, so $\text{out}[0,2]=\sum_j X[0,j]\,Y[j,2]=1\cdot 2.0+2\cdot 0.0=2.0$.

## Constraints

- 1–3 operands, each with at most 4 axes; inputs are float32 random tensors, compared against `torch.einsum` with atol=1e-5, rtol=1e-5
- Calling `torch.einsum` / `torch.tensordot` is forbidden; the judge probes each illegal spec and asserts ValueError
