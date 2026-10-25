# FINCH: Causally Normalized Window-Attention Scoring

## Problem

SnapKV scores each context key by the **mean** attention it received from an observation window of the most recent queries. FINCH (Zhu et al., 2024, [arXiv:2407.09112](https://arxiv.org/abs/2407.09112); `FinchPress` in `kvpress/presses/finch_press.py` of NVIDIA/kvpress, the `normalize_scores=True` path) spots a bias: **each window query has a different number of causally visible tokens** — the oldest window query votes over few keys while the newest sees nearly everything; with different softmax denominators the attention magnitudes of different query rows are incomparable. FINCH multiplies row \(i\) (counting from the oldest window query) by that row's visible token count \(C+i\) (\(C\) = context length) **before** averaging across queries, turning the row mean into a scale-consistent weighted accumulation; a within-GQA-group mean follows, and finally the observation window itself gets a floor of global-max + 1 (the window is never pruned).

## Rules

Inputs: `attn` of shape \((H_q, W, C)\) — softmaxed attention of the \(W\) window queries over the \(C\) context tokens; **row 0 is the oldest window query (global position \(C\))**, row \(i\) sits at global position \(C+i\); `num_kv_heads` \(=H\) (\(H_q\) divisible by \(H\), group count \(G=H_q/H\)); `compression_ratio` \(r\in[0,1)\).

1. Row scaling: for every query head and window row \(i\in[0,W)\), multiply the whole row by \((C+i)\).
2. Mean over the window dimension (divide by \(W\)) → \((H_q, C)\).
3. GQA group mean: head \(h\) averages query heads \([hG, hG+G)\) → \((H, C)\).
4. Right-append \(W\) pad entries: pad value = **global max over all \(H\) rows of the \((H,C)\) scores + 1**, giving `scores` of shape \((H, L)\), \(L=C+W\). The first \(C\) columns are context positions, the last \(W\) are window positions.
5. \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\); per row take the top-\(n\_kept\) scores (larger first, **ties go to the smaller position**) and output the kept positions ascending.

Return two tensors: `scores` (float32, \((H, L)\)) and `kept` (int64, \((H, n\_kept)\)).

## Example

```
Hq=4, W=2, C=3, num_kv_heads=2, G=2, r=0.4 → L=5, n_kept=3

query head 0: rows×[3,4]  [[1.2,0.6,1.2],[0.4,2.0,1.6]] → row mean [0.8,  1.3,  1.4]
query head 1: rows×[3,4]  [[0.9,0.9,1.2],[2.4,0.8,0.8]] → row mean [1.65, 0.85, 1.0]
kv head 0 = mean(head0, head1) = [1.225, 1.075, 1.2]
(similarly kv head 1 = [1.3125, 0.9125, 1.275])
global max = 1.3125 → pad = 2.3125

scores = [[1.225,  1.075,  1.2,   2.3125, 2.3125],
          [1.3125, 0.9125, 1.275, 2.3125, 2.3125]]
kept   = [[0, 3, 4],
          [0, 3, 4]]
```

## Constraints

- CPU float32; the input `attn` is already softmaxed and contains only the window-queries-over-context-keys block (within-window visibility is handled upstream).
- The row multiplier is \(C+i\) (\(i\) from 0), i.e. the visible key count **excluding the query's own position** — this mirrors the upstream code exactly, it is not \(C+i+1\).
- The pad value uses a global max over all \(H\) rows, not a per-row max.
- With \(r=0\) everything is kept; with \(W=1\) the row scaling degenerates to a constant (it cannot reorder context positions).
- All ties break toward the smaller position; output is sorted ascending.
