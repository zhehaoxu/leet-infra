# SnapKV：窗口注意力 + 一维平均池化

## 题目描述

**SnapKV**（Li et al., 2024, [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)）：用「**最近 window_size 个 query 对更早 key 的注意力**」当作每个 key 的重要性分。核心创新是加一层 **一维平均池化**（`avg_pool1d` kernel=kernel_size, padding=kernel_size//2, stride=1）把注意力沿 key 位置平滑，缓解相邻位置的重要度突变。

观察窗口末尾的 `window_size` 个 query **不参与剪枝**——直接以「比全局最大分还高 1」的分数被强行保留。

## 规则

给定 `attn` 形状 $(H, W, P)$：H 个头（`num_key_value_heads = num_heads`，本题不做 kv-group 平均）、每个头里 **最后 W 个 query** 对 **前 P 个 key** 的注意力权重（即 `attentions[..., -W:, :-W]` 的结果，`softmax` 已完成），以及 `kernel_size`（奇数）。返回 shape $(H, P + W)$ 的分数：

1. `base = attn.mean(dim=-2)` — 沿 W 求均值 → $(H, P)$
2. `pooled = avg_pool1d(base, kernel_size=K, padding=K//2, stride=1)` — 长度不变（K 为奇数），**边界外的位置视为 0，分母仍是 K**（PyTorch `avg_pool1d` 语义）
3. `out = pad(pooled, right=W, value=pooled.max() + 1)` — 把 window 位置以「保底最大分」拼回

## 示例

```
H=2, W=3, P=5, kernel_size=3
attn[h=0] = [[0.1, 0.2, 0.05, 0.15, 0.1],
             [0.3, 0.1, 0.1, 0.2, 0.05],
             [0.05, 0.4, 0.1, 0.1, 0.2]]
attn[h=1] = [[0.2, 0.1, 0.3, 0.1, 0.05],
             [0.1, 0.2, 0.1, 0.3, 0.2],
             [0.3, 0.1, 0.1, 0.1, 0.25]]

base[h=0] = mean over W → [0.150, 0.2333, 0.0833, 0.150, 0.1167]
pooled[h=0] = avg_pool(k=3, pad=1, stride=1) →
    [0.1278, 0.1556, 0.1556, 0.1167, 0.0889]
scores[h=0] = pooled ++ [max+1]*3 =
    [0.1278, 0.1556, 0.1556, 0.1167, 0.0889, 1.1667, 1.1667, 1.1667]
scores[h=1] = [0.1111, 0.1667, 0.1556, 0.1667, 0.1111, 1.1667, 1.1667, 1.1667]
```

## 约束

- CPU float32；`kernel_size` 保证为奇数（1、3、5、7 …）；`padding = kernel_size // 2`。
- `avg_pool1d` 边界处理：越界补 0，分母固定为 kernel_size（不计越界位）。
