# Notes · FINCH

## The row-scale bias caused by causal visibility

After softmax, the query row at position \(p\) is normalized over \([0,p]\) only (in the KV-cache setting the denominator scales roughly with the visible token count). In the observation window the oldest query sits at global position \(C\) and the newest at \(C+W-1\): even when both vote for the same context key, the older query's single vote is naturally larger (smaller denominator). SnapKV's plain `mean` over window rows inherits this bias. `FinchPress.score` (kvpress `finch_press.py`, `normalize_scores=True`) fixes it:

```python
non_zero_counts = torch.arange(k_len - window_size, k_len)[None, None, :, None]
attn_weights = attn_weights * non_zero_counts   # row i (from oldest) × (C+i)
scores = attn_weights.mean(dim=-2)              # then mean over window rows
scores = scores.view(bsz, H_kv, groups, C).mean(dim=2)  # GQA group mean
scores = F.pad(scores, (0, window_size), value=scores.max().item() + 1)
```

Multiplying by \(C+i\) restores the "un-normalized accumulated magnitude" before averaging — equivalent to weighting each query by its receptive width. Note the multiplier is `arange(k_len - window, k_len)` = \(C+i\) (**excluding the query itself**); copy the upstream, do not "fix" it to \(C+i+1\).

## Comparison with neighbouring problems

- `kv-observed-attention` (149): **column** normalization — score = column sum ÷ (L−j), averaging the attention each key received by how many queries saw it. FINCH does **row** weighting — scaling each query row by its visible width before the cross-query mean. The two are mirrored in normalization direction.
- `kv-snapkv-pooling` (147): takes the **max** over window rows; FINCH takes the weighted mean.
- `kv-expected-attention` (150): Gaussian-expectation estimation of future queries. FINCH's `F.pad(..., value=max+1)` is the same floor trick as 150's sink guarantee: the max is global (across all kv heads), so window positions can never drop out of any top-k.
- `kv-chunk-independent-topk` (165): FINCH also has a chunk mode, but chunk-wise top-k is already covered by 165; this problem tests only the global top-k.

## Scope note (source verification)

The FINCH **paper** (arXiv:2407.09112) contains a second mechanism: budgets scaled by prompt length and similarity-based key merging within GQA groups. The kvpress `FinchPress` implements "delimiter-driven dynamic window + the normalized scoring above + optional rerotation"; delimiter detection needs a tokenizer and rerotation already has its own problem (167), so neither is included here. This problem follows the kvpress source and tests only the deterministic scoring/selection math — do not read it as a clustering/merging task.

## Pitfalls

- Row semantics: row 0 is the oldest window query (smallest multiplier \(C\)), row \(W-1\) the newest (\(C+W-1\)). Reversing the input row order is the classic mistake.
- The pad value is computed from the pre-pad global max of the `(H,C)` block + 1 — a single scalar shared by all kv heads.
- \(n\_kept\) counts over \(L=C+W\) (window slots consume budget too), not just \(C\).
- \(W=1\): the multiplier is the constant \(C\); row weighting cannot reorder anything — a good sanity probe.
- All-zero attention: context scores are all 0, pad = 1; the top-k tie inside the window breaks toward smaller positions (consecutive from \(C\)).
