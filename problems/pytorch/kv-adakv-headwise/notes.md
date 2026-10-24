# 解题说明 · AdaKV

## 关键实现（对齐 kvpress）

```python
n_kept = int(k_len * (1 - cr))          # 每 head 名义保留数
n_safe = int(n_kept * alpha)
if alpha > 0:
    n_safe = max(1, n_safe) if n_kept > 0 else 0
top_idx = torch.topk(scores, n_safe, dim=-1).indices
scores.scatter_(-1, top_idx, torch.finfo(scores.dtype).max)

n_pruned = num_key_value_heads * (k_len - n_kept)
indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()
```

## 与 SnapKV 的关系

AdaKV 是**包装器**（`AdaKVPress(press=SnapKVPress(...))`）：底层的 SnapKV / KNorm / TOVA 都可以套上它。题目直接给分数矩阵，跳过 scorer 部分。

## 为什么是 flatten 而不是 per-head topk

per-head 是"每头保 n_kept 个"—— 头间预算固定；flatten 是"全局保 H·n_kept 个、随头分配"—— 头间预算可变。AdaKV 论文实验显示 flatten 更省，因为 attention 稀疏度**跨头差异极大**。

## safeguard 的作用

纯 flatten 会让某些"平均分都低"的头**被掏空**——它一个 token 都不保、attention 变成只对自己。α=0.2 保底 `n_kept * 0.2` 个，避免退化。题目示例里 α=0.5 强制头 1 至少保 1 个。

## 浮点 ∞ 的坑

kvpress 用 `finfo.max` 而不是 `float('inf')`：因为 `inf` 在 topk 反向排序时会出现 `inf - inf = nan` 之类的问题；用最大有限值更稳。
