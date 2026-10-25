# Solution Notes · AdaKV

## Key implementation (aligned with kvpress)

```python
n_kept = int(k_len * (1 - cr))          # nominal number kept per head
n_safe = int(n_kept * alpha)
if alpha > 0:
    n_safe = max(1, n_safe) if n_kept > 0 else 0
top_idx = torch.topk(scores, n_safe, dim=-1).indices
scores.scatter_(-1, top_idx, torch.finfo(scores.dtype).max)

n_pruned = num_key_value_heads * (k_len - n_kept)
indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()
```

## Relation to SnapKV

AdaKV is a **wrapper** (`AdaKVPress(press=SnapKVPress(...))`): the underlying SnapKV / KNorm / TOVA can all be wrapped by it. This problem gives the score matrix directly and skips the scorer part.

## Why flatten rather than per-head topk

Per-head means "keep n_kept per head"—the budget per head is fixed; flatten means "keep H·n_kept globally, distributed across heads as needed"—the budget per head is variable. Experiments in the AdaKV paper show flattening saves more, because attention sparsity **varies enormously across heads**.

## What the safeguard does

Pure flattening would **hollow out** some heads whose "average scores are all low"—such a head keeps not a single token and its attention degenerates to attending only to itself. α=0.2 guarantees `n_kept * 0.2` tokens, avoiding degeneration. In the example, α=0.5 forces head 1 to keep at least 1.

## The floating-point ∞ pitfall

kvpress uses `finfo.max` instead of `float('inf')`: because `inf` in reverse topk sorting can produce issues such as `inf - inf = nan`; the largest finite value is more robust.
