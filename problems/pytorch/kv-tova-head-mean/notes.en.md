# Solution Notes · TOVA

## Why only the last query

At generation time, "the history the next step is most likely to reference" is exactly the attention distribution of the last query. TOVA assumes this distribution is relatively stable during decoding and uses it to score all keys in one shot—cheaper than SnapKV (it does not look inside the attention matrix, only at one row).

## Cross-head averaging means "globally shared"

TOVA's original paper stresses that **scores are shared across heads**. As a result, topk picks the same set of positions in every head—keeping tokens that are consistently important for all heads. In kvpress, `scores = attn.mean(dim=head)` followed by `repeat(num_kv_heads, 1)` copies this shared score to each KV head.

This differs from AdaKV's **cross-head flattening to pick a global topk**: AdaKV lets different heads keep different counts, whereas TOVA makes every head keep the same number (and the same positions).

## Tail padding

kvpress uses `F.pad(scores, (0,1), value=max+1)` to keep the last token (`attn[..., -1:, -1:]`, the self-attention slot) with the highest score.

## Relation to ObservedAttention

ObservedAttention uses **the average over all queries for each key**, TOVA uses **the last query**—the two extremes of "past attention vs. future attention".
