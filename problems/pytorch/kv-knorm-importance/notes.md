# 解题说明 · KNorm

## 一行代码

```python
def solve(keys):
    return -torch.linalg.vector_norm(keys, dim=-1)
```

## 为什么加负号

`ScorerPress.compress` 用 `scores.topk(n_kept, largest=True)` 挑保留位。KNorm 关心的是「**保留 norm 大的 key**」——因此分数就取 `-norm`，让 topk 挑到 norm 最大的位置。

## 精度坑

kvpress 官方注释指出：bf16 上直接算 norm 会让 **约 74% 的分数打平**（Qwen3-8B 上），把选择权交给 topk 的 tie-break。所以真实实现要 `dtype=torch.float32`。题目里 keys 传入即 float32，可省。

## KNorm vs H2O / SnapKV

KNorm 完全不看 attention，也不用 query；只依赖 key 本身。这让它**兼容 flash-attention**（不必保留 attn 权重），代价是精度稍逊。实践中常与 StreamingLLM 组合（sink + KNorm）。
