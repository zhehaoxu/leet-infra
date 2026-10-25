# Solution Notes · KNorm

## One Line of Code

```python
def solve(keys):
    return -torch.linalg.vector_norm(keys, dim=-1)
```

## Why the Negative Sign

`ScorerPress.compress` uses `scores.topk(n_kept, largest=True)` to pick the kept positions. What KNorm cares about is "**keeping the keys with large norms**" — so the score is taken as `-norm`, letting topk select the positions with the largest norms.

## Precision Pitfall

The official kvpress comment points out: computing the norm directly in bf16 makes **about 74% of the scores tie** (on Qwen3-8B), handing the selection over to topk's tie-break. That is why real implementations use `dtype=torch.float32`. In this problem the keys arrive as float32, so this can be skipped.

## KNorm vs H2O / SnapKV

KNorm never looks at attention and does not use the query either; it depends only on the key itself. This makes it **flash-attention compatible** (no need to retain attn weights), at the cost of slightly lower accuracy. In practice it is often combined with StreamingLLM (sink + KNorm).
