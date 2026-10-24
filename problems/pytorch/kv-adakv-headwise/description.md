# AdaKV：跨头 flatten 全局 top-k + 头保底 safeguard

## 题目描述

普通 `ScorerPress` 让每个 KV 头各挑各的 top-k —— 每头保同样多 token。**AdaKV**（[arXiv:2407.11550](https://arxiv.org/abs/2407.11550)）指出：不同 head 的重要 token 数量差异极大，让 head 之间"抢预算"更省内存。做法是把分数**跨头 flatten**、按全局排序剪 $H \times (L - n_\text{kept})$ 个最低分。

为防止某个头被掏空，加**头保底 safeguard**：每头先强制保留 $\lfloor n_\text{kept} \cdot \alpha \rfloor$ 个最高分（把它们的分数拉到 $+\infty$），再走全局剪枝。

## 规则

给定 `scores` $(H, L)$、`compression_ratio $\in [0,1)$`、`alpha_safeguard $\in [0,1]`：

- $n_\text{kept} = L - \lfloor L \cdot \text{compression\_ratio} \rfloor$ 之**保留数**（等价于 `int(L * (1 - cr))`）
- $n_\text{pruned} = H \cdot (L - n_\text{kept})$ — 全局剪掉这么多
- 若 `alpha_safeguard > 0`：$n_\text{safe} = \max(1, \lfloor n_\text{kept} \cdot \alpha \rfloor)$；对每头把 top-$n_\text{safe}$ 分数置为 $+\infty$（`torch.finfo(float32).max`）
- 从 flatten 后的分数中挑 $n_\text{pruned}$ 个**最小**者剪掉；剩下的保留
- 返回 shape $(H, L)$ 的 bool 掩码 `kept`

## 示例

```
scores = [[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
          [0.6,  0.5, 0.4,  0.3, 0.2,  0.1]]
compression_ratio = 0.5  →  n_kept = 3, n_pruned = 2*(6-3) = 6

alpha_safeguard = 0.5  →  n_safe = max(1, floor(3*0.5)) = 1
  保护 (0,0)、(1,0) 至 +∞
  剩下从 10 个非 ∞ 挑 6 个最小 → (0,5),(0,4),(0,3),(1,5),(1,4),(1,3)
  kept = [[T,T,T,T,T,F],
          [T,F,F,F,F,F]]   ← 头 1 只剩 1 个（被保底救回）

alpha_safeguard = 0.0  →  不保护
  全局最小 6 → 头 1 全被剪：[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 头 0 只剪 0.7]
  kept = [[T,T,T,T,T,T],
          [F,F,F,F,F,F]]
```

## 约束

- CPU float32；$\lfloor\cdot\rfloor$ 用 Python `int()`（向零截断）。
- `topk` 挑最大；`bottom-k` 通过 `topk(-x)` 或 `topk(..., largest=False)`。
- 保底值用 `torch.finfo(torch.float32).max` 而非 `float('inf')`（与 kvpress 一致）。
