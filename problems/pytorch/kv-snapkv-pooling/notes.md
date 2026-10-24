# 解题说明 · SnapKV Pooling

## 三步的核心

kvpress 官方 `SnapKVPress.score` 结构：

```python
scores = attn_weights.mean(dim=-2)               # (B,H,P)
scores = F.avg_pool1d(scores.unsqueeze(1),
                      kernel_size=self.kernel_size,
                      padding=self.kernel_size // 2,
                      stride=1).squeeze(1)       # (B,H,P)
# 忽略：per kv-group 平均（题目 num_kv_groups=1）
scores = F.pad(scores, (0, W), value=scores.max().item() + 1)
```

`attn_weights` 已由上游取好 `[..., -W:, :-W]`，即窗口内 query 对更早 key 的部分——所以题目直接给它。

## 为什么 avg_pool1d 需要 unsqueeze

`F.avg_pool1d` 期望 3D 输入 `(N, C, L)`。`scores` 是 `(H, P)`，加个 batch=1 就变成 `(1, H, P)`——每个头当作一个 channel。池化只在最后一维 P 上滑动，池完再 squeeze 回 `(H, P)`。

## Padding 语义

`avg_pool1d` 用 0 填充、`count_include_pad=False` 才是「分母只算合法位」；而 SnapKV 用的是 **默认 `count_include_pad=True`**——分母始终是 `kernel_size`。所以位置 0 的池化结果 = `(0 + base[0] + base[1]) / K`。

## 与 TOVA 的差异

TOVA 只用**最后一个 query**、SnapKV 用**最后 W 个 query 的平均 + 池化**。池化让相邻 key 的分数被"平滑"，减少孤立高分点被优先保留的问题。

## 尾部 window 的 max+1

`window` 是最近 W 个 token——推理时必须保留（否则生成会失忆）。用「比全局最大分还高 1」的常量填在尾部，让 topk(n_kept) 一定优先挑到它们。
