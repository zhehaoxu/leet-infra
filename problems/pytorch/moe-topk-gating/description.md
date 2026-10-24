# MoE 路由：Top-k Gating

## 题目描述

混合专家（MoE）的**路由门控**：给每个 token 在 $E$ 个专家里挑概率最高的 $k$ 个，把它们的门控权重**重新归一化到和为 1**，作为后续专家输出加权合并的系数。这是 Mixtral / DeepSeek-V2 / Qwen-MoE 等模型的 `router` 前向。

## 规则

实现 `solve(logits, k)`：
- `logits` 形状 $(T, E)$：每个 token 对每个专家的原始打分（linear 层输出，未做 softmax）。
- `k` 整数：每个 token 选多少个专家。$1 \le k \le E$。

步骤：
1. `probs = softmax(logits, dim=-1)`（沿专家维）。
2. 每行取 `probs` 最大的 $k$ 个，得到 `top_vals (T,k)` 与 `top_idx (T,k)`。
3. **重新归一化**：`scores = top_vals / top_vals.sum(-1, keepdim=True)`。

返回 `(scores, indices)`：
- `scores` 形状 $(T, k)$，每行和为 1（float）。
- `indices` 形状 $(T, k)$，`long` 类型，专家编号。

> 与 HF `MixtralTopKRouter.forward` 完全一致：先 softmax 再 topk，最后 renorm。tie 情况按较小专家索引优先。

## 示例

```
logits = [[2.0, 1.0, 0.5, -1.0, -2.0, 0.0],
          [0.0, 0.0, 3.0, 2.4, -1.0, -1.0],
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]，k=2

scores = [[0.731059, 0.268941],
          [0.645656, 0.354344],
          [0.731059, 0.268941]]
indices = [[0, 1], [2, 3], [5, 4]]
```

## 约束

- 全部在 CPU 上、`float32`。
- 判题对拍：向量化 `topk` 与逐行 python 排序两条路径。
