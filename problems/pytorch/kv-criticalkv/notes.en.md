# Notes · CriticalKV

## Motivation: high attention ≠ high information

The ScorerPress family (knorm, observed/expected attention, SnapKV) all use only **key-side or attention-side** signals. CriticalKV (arXiv:2502.03805) observes that a token's actual influence on the final logits is `attn_t · (v_t W_o)` — a token with large attention but a value that projects to nearly zero (its per-head outputs cancelling out) can be evicted safely. Hence it multiplies the base score by \(\|v_t W_o\|_1\) (averaged over the query heads of one GQA group). The norm is given directly as an input here to avoid real model weights; its definition is exactly `CriticalKVPress.vwl1norm`: for each query head compute `‖v @ Wo_head‖₁`, then average within the group.

## Why two stages

If the whole ranking were switched to the weighted score, tokens with extremely high base scores could be killed by a small \(\|vW_o\|_1\). The upstream implementation (`criticalkv_press.py`, `CriticalKVPress.score` in NVIDIA/kvpress) compromises with two stages:

```python
selection_budget = int((1 - ratio) * k_len * first_stage_ratio)   # stage 1 eats a fraction of the budget
if first_stage_ratio > 0:
    selection_budget = max(1, selection_budget)                   # short-context guard: protect at least 1
top_k_index = torch.topk(scores, selection_budget).indices
scores = (scores + epsilon) * projected_norm                      # stage 2 re-scores
scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)   # stage-1 hits → infinite floor
```

Note the scatter happens **after** the multiplication: stage-1 tokens bypass value weighting entirely — that is what "protected" means. The outer ScorerPress flow then takes the global top-k of the returned scores (equivalent to step 5 here).

## Difference from ObservedAttention / ExpectedAttention (key point)

- `kv-observed-attention` (order 149): score = historical attention column-sum ÷ (L−j), pure key-side accumulation, **never looks at values**.
- `kv-expected-attention` (order 150): Gaussian expectation over queries, softmax(μ·k/√d + kᵀΣk/2d)×‖v‖₂ — uses only the **L2 norm** of the value as a scalar.
- CriticalKV (this problem): not a new attention estimator but a **wrapper** — take any base score (possibly either of the two above), then multiply by \(\|vW_o\|_1\). The difference is channel relevance: ‖v‖₂ only measures value magnitude, while \(\|vW_o\|_1\) measures the per-channel absolute contribution in the **output basis**; a value with large ‖v‖₂ that aligns with the null space of W_o is over-rated by the former and correctly discounted by the latter.

## Ties and determinism

Upstream `torch.topk` makes no tie guarantees; this problem normalizes to the (−score, position) lexicographic order with ascending output. A stable sort (e.g. `torch.sort(-x, stable=True)`) or explicit key tuples matches the oracle. Also, epsilon is not decoration: `(scores + eps)` keeps zero-base-score tokens at exactly 0 after weighting (no sign flips) and adds numerical headroom.

## Degenerate cases checklist

- `first_stage_ratio=0`: no stage 1; pure \((s+\varepsilon)\cdot n\) top-k.
- `wov_norm ≡ 0`: unprotected tokens all score 0; kept set = stage 1 ∪ (remaining slots filled by the smallest positions).
- `first_stage_ratio=1`: \(b_1=n\_kept\); the kept set is exactly the raw-score top-\(n\_kept\) and the value weighting has zero effect — a good invariant probe.
- `compression_ratio=0`: \(n\_kept=L\), every position is kept.
