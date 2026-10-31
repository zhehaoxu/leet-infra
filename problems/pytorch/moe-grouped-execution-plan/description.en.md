# MoE: Grouped-Execution Sort Plan and Unsort-Merge

## Task

Implement `solve`: compile the table of "which experts each token picked" into the index plan a grouped matrix multiply (grouped mm) actually needs, run the per-expert computation in that sorted layout, then unsort and merge it back into one output per token.

This is a restricted numerical model of the expert-side `grouped_mm` / `batched_mm` implementations in MoE code: the sort, the offset table, the gather/inverse-permutation pair, the sentinel-row handling and the merge strategy are all fixed explicitly by the statement. The inside of an expert is abstracted as one element-wise diagonal product (`hidden[t] * diag[e]`) because this problem tests **slot algebra**, not matmuls; there is no backward pass, no CUDA graph and no real kernel scheduling.

## Inputs

- `hidden`: CPU float64 `[T,D]`, non-empty.
- `expert_ids`: CPU int64 `[T,k]`; legal values are $0,\dots,E$ where $E$ **is itself the sentinel** (the slot was dropped, or its expert is not on this device).
- `routing_weights`: CPU float64 `[T,k]`, same shape as `expert_ids`; not renormalized.
- `diag`: CPU float64 `[E,D]`, its second dimension must match `hidden`; `E = diag.shape[0]`, the expert count is never passed separately.
- `T,k,E = 1..8`, `D = 1..4`.

Noncontiguous tensors are accepted and no input is mutated. Raise `ValueError` for out-of-range ids (negative or greater than $E$), shape mismatch, wrong dtype, non-CPU tensors, wrong rank, or an empty dimension.

## Exact rules

Slots are numbered token-major: slot $s = t\,k + j$ is the $j$-th choice of token $t$, and there are $S = Tk$ slots.

**Sort**: order the slots by the pair (expert id, slot id) ascending, **stably**, giving `perm`; the sentinel id is the largest, so sentinels **naturally fall to the tail** and the valid rows are exactly the first $\mathit{valid}$ rows:

$$
(i<j)\Rightarrow\bigl(e_{\mathit{perm}[i]}<e_{\mathit{perm}[j]}\ \text{or}\ (e_{\mathit{perm}[i]}=e_{\mathit{perm}[j]}\ \text{and}\ \mathit{perm}[i]<\mathit{perm}[j])\bigr),\qquad
\mathit{token\_rows}[i]=\bigl\lfloor \mathit{perm}[i]/k \bigr\rfloor .
$$

`token_rows[i]` says which token's hidden row $i$ gathers; below, $\mathit{sorted\_experts}[i]=e_{\mathit{perm}[i]}$ is a derived quantity and is not returned.

**Offset table**: `counts[e]` counts only slots with $0\le e<E$ (the sentinel is excluded), and `offsets` is the **inclusive** prefix sum of `counts`, of length $E$:

$$
\mathit{counts}_e=\#\{s: e_s=e\},\qquad \mathit{offsets}_e=\sum_{e'\le e}\mathit{counts}_{e'},\qquad \mathit{offsets}_{E-1}=\mathit{valid}=S-\#\{s:e_s=E\}.
$$

**Grouped output**: with $\hat e=\min(\mathit{sorted\_experts}[i],\,E-1)$ (clamp first so the gather index is in range),

$$
\mathit{grouped}[i]=\begin{cases}
h_{\mathit{token\_rows}[i]}\odot \mathit{diag}_{\hat e}, & \mathit{sorted\_experts}[i]<E,\\[2pt]
\mathbf{0}, & \text{sentinel} .
\end{cases}
$$

Sentinel rows must be entirely 0, regardless of which expert the clamped index happens to select.

**Unsort and merge**: `inv_perm` satisfies $\mathit{inv\_perm}[\mathit{perm}[i]]=i$; weights are applied in sorted order, then the result is restored to token-major order and summed over $k$:

$$
\mathit{out}_t=\sum_{j=0}^{k-1}\bigl(\mathit{grouped}\odot w_{\mathit{perm}}\bigr)\bigl[\mathit{inv\_perm}[\,tk+j\,]\bigr].
$$

The merge is "unsort → reshape → sum", not `index_add_`: the former accumulates in the fixed order $j=0,\dots,k-1$.

## Return value

Return a seven-item `tuple`, with $S=Tk$:

```text
(out, grouped, perm, inv_perm, token_rows, counts, offsets)
```

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `out` | float64 `[T,D]` | The merged token output |
| `grouped` | float64 `[S,D]` | Sorted per-expert output, with sentinel rows all 0 |
| `perm` | int64 `[S]` | Which original slot row $i$ came from |
| `inv_perm` | int64 `[S]` | Which sorted row slot $s$ landed on |
| `token_rows` | int64 `[S]` | Which token sorted row $i$ gathers |
| `counts` | int64 `[E]` | Slots per expert (sentinel excluded) |
| `offsets` | int64 `[E]` | Inclusive prefix sum of `counts` |

Compare floats with `rtol=2e-11, atol=2e-11` and index tensors bit for bit.

## Example

```python
hidden = [[1.0, 2.0],
          [3.0, -1.0]]
expert_ids = [[1, 3],
              [0, 1]]          # E = 3, so id 3 is the sentinel
routing_weights = [[0.5, 0.2],
                   [0.25, 1.0]]
diag = [[0.5, -1.0],
        [2.0, 0.0],
        [-1.0, 3.0]]

# perm        = [2, 0, 3, 1]
# token_rows  = [1, 0, 1, 0]
# inv_perm    = [1, 3, 0, 2]
# counts      = [1, 2, 0]
# offsets     = [1, 3, 3]
# grouped     = [[1.5, 1.0], [2.0, 0.0], [6.0, 0.0], [0.0, 0.0]]
# out         = [[1.0, 0.0], [6.375, 0.25]]
```

The four slot ids are `[1, 3, 0, 1]`. Sorting by (expert, slot) gives slot 2 (expert 0), slot 0 (expert 1), slot 3 (expert 1), slot 1 (expert 3 = sentinel), hence `perm = [2,0,3,1]` and `token_rows = perm // 2 = [1,0,1,0]`.

`counts`: one slot for expert 0, two for expert 1, none for expert 2, and the sentinel is not counted; `offsets = [1,3,3]`, so the first 3 rows are valid.

The three valid `grouped` rows are `hidden[1]*diag[0] = [1.5, 1.0]`, `hidden[0]*diag[1] = [2.0, 0.0]` and `hidden[1]*diag[1] = [6.0, 0.0]`; the fourth row is the sentinel and is all 0.

Multiplying by the sorted weights `[0.25, 0.5, 1.0, 0.2]` (that is `flat_w[perm]`) gives `[[0.375,0.25],[1.0,0.0],[6.0,0.0],[0,0]]`; after unsorting with `inv_perm`, token 0 sums `[1.0,0.0] + [0,0] = [1.0,0.0]` and token 1 sums `[0.375,0.25] + [6.0,0.0] = [6.375,0.25]`. Checking per token directly agrees: token 0's only live path is expert 1, `0.5 * [2,0] = [1,0]`, and the sentinel path must contribute 0 even though its weight is 0.2.
