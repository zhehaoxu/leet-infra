# Auxiliary-Loss-Free Grouped Routing and Bias Updates

## Task

Implement `solve`: use sigmoid affinities plus a per-expert bias that **participates only in selection, never in weighting**, select groups first, take top-k within selected groups, and normalize by the raw sigmoid scores. Optionally update the bias from this batch's expert load.

This is a restricted numerical model of DeepSeek-style auxiliary-loss-free balancing. The problem defines grouping, tie order, and the bias-update rule explicitly. It is not a step-by-step reproduction of any training framework and performs no training or gradient update.

## Inputs

- `x`: CPU float64 `[T,D]`.
- `gate_w`: CPU float64 `[E,D]`; router logits are `x @ gate_w.T`.
- `bias`: CPU float64 `[E]`, one selection bias per expert.
- `group_count`: Python `int` dividing E.
- `top_groups`: Python `int`, `1 <= top_groups <= group_count`.
- `k`: Python `int`, `1 <= k <= top_groups * group_size`.
- `gamma`: Python float, `>= 0`.
- `update_bias`: Python `bool`, default `False`.
- Experts are partitioned into contiguous equal groups; each group has `group_size = E / group_count` experts and must contain at least 2.
- `T,D,E=1..8`; inputs are valid, noncontiguous tensors are accepted, and no input is mutated.

Raise `ValueError` for invalid structure, out-of-range `k`, or `gamma < 0`.

## Exact rules

Sigmoid scores and choice scores:

$$
s_i=\sigma(xW_i^{\mathsf T}),\qquad c_i=s_i+b_i.
$$

A group $g$ scores as the sum of its **two largest** $c_i$ values. Each group has at least two experts, so nothing is counted twice. Choose `top_groups` groups by descending group score; exact ties go to the lower group index.

Within selected groups only, take the k experts with largest $c_i$; exact ties go to the lower expert index. **The bias does not affect weighting**:

$$
a_j=\frac{s_{i_j}}{\sum_{\ell=1}^{k}s_{i_\ell}}.
$$

Returned `weights` is a normalized `[T,k]` tensor, not a sparse `[T,E]` matrix.

Load is the number of batch-wide selections per expert. Let $\bar L=Tk/E$. The bias changes only when `update_bias=True`:

$$
b_i'=\begin{cases}
b_i+\gamma, & L_i<\bar L,\\
b_i-\gamma, & L_i>\bar L,\\
b_i, & L_i=\bar L.
\end{cases}
$$

Overloaded experts decrease their bias and underloaded experts increase it. With `update_bias=False`, return a new tensor numerically equal to the input `bias` but sharing no storage with it.

## Return value

Return the seven-item `tuple`:

```text
(weights, indices, scores, group_scores, selected_groups, load, new_bias)
```

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `weights` | float64 `[T,k]` | Normalized raw sigmoid scores |
| `indices` | int64 `[T,k]` | Selected experts in descending choice order |
| `scores` | float64 `[T,E]` | Raw sigmoid scores |
| `group_scores` | float64 `[T,G]` | Sum of each group's top two choice scores |
| `selected_groups` | int64 `[T,top_groups]` | Chosen groups in descending group score |
| `load` | int64 `[E]` | Batch-wide selection counts |
| `new_bias` | float64 `[E]` | Updated bias, or an unchanged copy |

Compare floats with `rtol=2e-11, atol=2e-11`.

## Example

```python
x = [[1.0]]
gate_w = [[1.0], [0.9], [-0.5], [-0.6]]
bias = [0.0, 0.0, 0.0, 0.0]
group_count = 2
top_groups = 1
k = 2
gamma = 0.1
update_bias = True
```

Group 0 (experts 0 and 1) has top-2 sum `sigmoid(1)+sigmoid(0.9)≈1.4420`, while group 1 has `≈0.7319`, so group 0 is selected. Output shown as tensor contents:

```python
weights = [[0.5070, 0.4930]]
indices = [[0, 1]]
scores = [[0.7311, 0.7109, 0.3775, 0.3543]]
group_scores = [[1.4420, 0.7319]]
selected_groups = [[0]]
load = [1, 1, 0, 0]
new_bias = [-0.1, -0.1, 0.1, 0.1]
```

Changing expert 2's bias to `2.0` makes group 1's top-2 sum larger and selects only experts 2 and 3; weights still normalize the raw sigmoid scores, so the bias never appears in them.
