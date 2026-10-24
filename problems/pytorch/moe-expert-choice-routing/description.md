# MoE 路由：Expert-Choice 路由

## 题目描述

Shazeer 2022「Mixture-of-Experts with Expert Choice Routing」把路由方向反过来：不再让每个 token 挑 $k$ 个专家，而是**每个专家挑 $k_c$ 个它最想要的 token**。天然实现完美负载均衡（每专家恰好分派 $k_c$ 个 token）。

## 规则

实现 `solve(logits, k_c)`，`logits` 形状 $(T, E)$：

1. `probs = softmax(logits, -1)`（每 token 沿专家维归一化）。
2. 每个专家 $e$：沿 $T$ 维取 `probs[:, e]` 最大的 $k_c$ 个 token（tie 按较小 token 索引优先）。
3. `mask (T, E) bool`：`mask[t, e] = True` 当且仅当 token $t$ 被专家 $e$ 选中。
4. `weights = probs * mask.float()`；沿 $E$ 维 renorm 得 `weights_renormed`。
5. **若某 token 一个专家都没选它**（`mask[t].sum()==0`），其 renorm 行输出全零（分母保护：`clamp_min(1e-30)` 或 `torch.where`）。

返回 `(mask, weights_renormed)`。

## 示例

```
logits = [[2, 1, 0],
          [0, 2, 1],
          [1, 0, 2],
          [3, 3, 0],
          [0, 0, 1],
          [-1, -1, -1]]
k_c = 2

mask =    [[1,0,0],
           [0,1,0],
           [0,0,1],
           [1,1,0],
           [0,0,1],
           [0,0,0]]   # token 5 被 0 个专家选中
weights = [[1,0,0],
           [0,1,0],
           [0,0,1],
           [0.5,0.5,0],
           [0,0,1],
           [0,0,0]]
```

## 约束

- CPU float32；`mask.dtype == torch.bool`；每列 `mask[:, e].sum() == k_c` 恒成立。
