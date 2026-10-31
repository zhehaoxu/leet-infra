# MoE: Ragged Grouped Matrix Multiply with Offsets, Forward and Backward

## Task

Implement `solve`. Expert weights are stacked into one $[E,K,N]$ tensor, tokens have already been sorted by expert id into $[S,K]$, and the only thing lining the two sides up is a **prefix-sum `offs`**. Produce the grouped matmul output, the gradient with respect to the input, the gradient with respect to the weights, and how many rows each group actually consumed.

This is the shared substrate of every fast MoE execution path: `torch.nn.functional.grouped_mm`, `torch._grouped_mm` and the various Triton/CUTLASS grouped GEMMs all accept the same "sort by expert, describe the ragged boundaries with offsets" convention, and the fallback used when none of them is available (wrong dtype, no kernel, `torch.compile` with dynamic shapes) is exactly the naive "loop over $E$, slice, call `torch.mm`" version. What an interviewer probes here is not arithmetic but **boundary conventions**: is `offs` inclusive or exclusive, does an empty group get skipped, who owns the rows the offsets never reached, is the backward split strictly symmetric with the forward one. The problem stops at a pure function: no all-to-all, no actual permute/unpermute traffic, no autograd graph — gradients are just another computation you implement.

## Input

- `x`: CPU float64 `[S,K]`, non-empty 2-D, finite. Rows are sorted by expert id; group $e$ occupies one contiguous run of rows.
- `weight`: CPU float64 `[E,K,N]`, non-empty 3-D, finite. `weight[e]` is the linear layer of expert $e$.
- `offs`: CPU int32 or int64 `[E]`, non-empty 1-D. **Each element is the row index at which its group ends** (prefix sum, left-inclusive/right-exclusive), so the sequence must be non-decreasing, every element $\ge 0$, and $\mathit{offs}[E-1]\le S$.
- `grad_output`: CPU float64 `[S,N]`, non-empty 2-D, finite. Incoming gradient, row order matching `x`.

Dimension agreement is required: `x.size(1) == weight.size(1)`, `weight.size(2) == grad_output.size(1)`, `grad_output.size(0) == x.size(0)`, `offs.size(0) == weight.size(0)`. Non-contiguous tensors are accepted; inputs are never mutated.

## Exact Rules

Write $\mathit{offs}[-1]\equiv 0$ for notation only — it is not an element of the input. Group $e$ covers the **half-open** row interval

$$
\mathit{start}_e=\begin{cases}0,&e=0\\ \mathit{offs}[e-1],&e>0\end{cases}\qquad
\mathit{end}_e=\mathit{offs}[e]\qquad \mathit{count}_e=\mathit{end}_e-\mathit{start}_e .
$$

Four conventions, applied literally:

- **Forward**: $\mathit{out}[\mathit{start}_e:\mathit{end}_e]=x[\mathit{start}_e:\mathit{end}_e]\ @\ \mathit{weight}[e]$.
- **Empty group** ($\mathit{count}_e=0$) is skipped outright: it produces no values and $\mathit{grad\_weight}[e]$ stays all zeros. For an empty group on its own, "multiply anyway" and "skip" give identical results (a zero-row matmul never reads the weight); what the judge can catch is picking the wrong `weight` index for a **non-empty** group, or getting the interval partition itself wrong.
- **Unassigned tail rows**: the rows in $[\mathit{offs}[E-1],\,S)$ belong to no expert. Their forward output rows and input-gradient rows **must be 0**, and they contribute nothing to any $\mathit{grad\_weight}$.
- **Backward** (strictly symmetric with the split):
$$
\mathit{grad\_input}[\mathit{start}_e:\mathit{end}_e]=\mathit{grad\_output}[\mathit{start}_e:\mathit{end}_e]\ @\ \mathit{weight}[e]^{\top},\qquad
\mathit{grad\_weight}[e]=x[\mathit{start}_e:\mathit{end}_e]^{\top}\ @\ \mathit{grad\_output}[\mathit{start}_e:\mathit{end}_e].
$$

`counts[e]` is simply $\mathit{count}_e$, obtained by differencing `offs`; you never need to count rows.

Invalid input (dtype/shape/device/non-finite/`offs` not integral, decreasing or out of range/dimension mismatch) raises `ValueError`.

## Returns

`(out, grad_input, grad_weight, counts)`:

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `out` | float64 `[S,N]` | grouped matmul result, zero in the tail rows |
| `grad_input` | float64 `[S,K]` | gradient wrt `x`, zero in the tail rows |
| `grad_weight` | float64 `[E,K,N]` | gradient wrt `weight`, zero for empty groups |
| `counts` | int64 `[E]` | rows consumed per group, summing to $\mathit{offs}[E-1]$ |

Floats compared with `rtol=atol=2e-11`, `counts` bit-for-bit.

## Example

$S=6$, $E=3$, $K=2$, $N=2$:

```python
x   = [[1, 2], [0, -1], [2, 1], [1, 0], [-1, 2], [3, 3]]   # S=6, K=2
w0  = [[1, 0], [2, 1]]   w1 = [[1, 1], [1, 1]]   w2 = [[0, 2], [-1, 1]]   # E=3, N=2
offs = [2, 2, 5]
grad = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [7, 7]]

# counts       [2, 0, 3]
# out          [[5, 2], [-2, -1], [-1, 5], [0, 2], [-2, 0], [0, 0]]
# grad_input   [[1, 2], [0, 1], [2, 0], [0, -2], [4, 2], [0, 0]]
# grad_weight  e0: [[1, 0], [2, -1]]   e1: [[0, 0], [0, 0]]   e2: [[4, 0], [1, 5]]
```

Group by group:

- **e0** takes rows $[0,2)$: $[1,2]\cdot[[1,0],[2,1]]=(1+4,\ 0+2)=(5,2)$ and $[0,-1]$ gives $(-2,-1)$. On the gradient side $\mathit{grad\_weight}[0]=x_{0:2}^{\top}\mathit{grad}_{0:2}$: entry $(0,0)=1\cdot1+0\cdot0=1$, entry $(1,1)=2\cdot0+(-1)\cdot1=-1$.
- **e1** has $\mathit{offs}[1]=\mathit{offs}[0]=2$ → empty, `counts[1]=0`, $\mathit{grad\_weight}[1]$ all zeros. Note that $w_1$ is never read for this input: replacing it with anything at all leaves all four return values unchanged.
- **e2** takes rows $[2,5)$ (three rows): $[2,1]\cdot w_2=(-1,5)$, $[1,0]\cdot w_2=(0,2)$, $[-1,2]\cdot w_2=(-2,0)$. $\mathit{grad\_input}$ uses $w_2^{\top}$: row 2 is $(1,1)\cdot w_2^{\top}=(1\cdot0+1\cdot(-1),\ 1\cdot2+1\cdot1)=(2,0)$.
- **Row 5** lies in $[\mathit{offs}[2],S)=[5,6)$ and no expert claims it: `out` row 5 is $[0,0]$, `grad_input` row 5 is $[0,0]$, and its gradient $[7,7]$ reaches no $\mathit{grad\_weight}$ either.

Every number above is an integer multiply-accumulate, so the example is fully hand-checkable.
