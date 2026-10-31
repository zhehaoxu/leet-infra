# NSA branch dispatch ledger: GQA shared scores, top-n block selection and gated aggregation

DeepSeek's NSA (Native Sparse Attention, arXiv 2502.11089) remaps each query position's KV into three **independent** branches — compression (cmp), selection (slc), and sliding window (win) — aggregated by learned gate weights. This problem tests the dispatch ledger: how GQA-group shared scores are computed, how top-n block selection ranks, how branch KV lengths are counted, and how the gates aggregate. The softmax/matmul of attention itself is out of scope (given as intermediate values).

## Pinned simplifications

- Compression block length l, compression stride d, and selection block size l' are all equal (l = d = l'). Paper Eq. (9)'s cross-granularity score mapping then degenerates to p_slc = p_cmp; selection blocks correspond one-to-one with compression blocks.
- L tokens generated (numbered 0..L-1); compression block i covers tokens [i·l, (i+1)·l); only blocks fully inside the causal range ((i+1)·l ≤ L) exist, so the usable block count is n_blocks = L // l.
- Selection operates on complete blocks only (Eq. 12 concatenates whole blocks); the unfinished tail is covered by the sliding-window branch.

## Rules (each maps to the paper)

1. **Group-shared scores (Eq. 10)**: query heads in a GQA group share KV, so block selection must be group-consistent — p_shared[i] = Σ_h p_cmp_heads[h][i], a **sum** (not mean, not max).
2. **Top-n selection (Eq. 11)**: rank p_shared in descending score (rank = 1 is highest); blocks with rank ≤ top_n are selected. Ties break by ascending block index (stable). The selected count is n_sel = min(top_n, n_blocks).
3. **Branch KV lengths (size terms of Eq. 6)**: cmp = n_blocks·l; slc = n_sel·l; win = min(window, L). N_t = their sum (the paper requires N_t ≪ t to keep sparsity).
4. **Gated aggregation (Eq. 5)**: output[j] = g_cmp·o_cmp[j] + g_slc·o_slc[j] + g_win·o_win[j]. The three branches have independent KV (the paper's anti-shortcut design); a zero-gate branch contributes nothing.

## Input

```python
solve(p_cmp_heads, L, l, top_n, window, gates, branch_outputs)
```

- `p_cmp_heads`: list[list[float]], per-head compressed-block scores (Eq. 8, already softmaxed; all heads equal length = n_blocks)
- `L`: generated token count; `l`: block length; `top_n`: selected block count; `window`: sliding-window size
- `gates`: [g_cmp, g_slc, g_win] (already sigmoided)
- `branch_outputs`: [[o_cmp], [o_slc], [o_win]], equal-length branch output vectors

Invalid input (empty heads, ragged lengths, top_n < 1, window < 1, gates length ≠ 3, ragged or empty output vectors) raises ValueError.

## Output

```python
{
  "shared_scores": [...],       # length n_blocks
  "selected_blocks": [...],     # selected block indices, ascending
  "cmp_kv_len": ..., "slc_kv_len": ..., "win_kv_len": ...,
  "n_remapped": ...,            # sum of the three branch KV lengths
  "output": [...]               # gated aggregation
}
```

## Example

```
Input:  p_cmp_heads = [[0.5, 0.1], [0.2, 0.3], [0.1, 0.2]] (3 heads in group),
        L = 4, l = 2, top_n = 1, window = 2,
        gates = [1.0, 0.0, 0.0], branch_outputs = [[1], [1], [1]]
Output: shared_scores = [0.8, 0.6] (group sum),
        selected_blocks = [0], slc_kv_len = 2,
        cmp_kv_len = 4, win_kv_len = 2, n_remapped = 8,
        output = [1.0]
```
