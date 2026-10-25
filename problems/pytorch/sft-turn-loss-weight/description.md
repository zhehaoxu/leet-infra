# SFT 数据：逐轮损失权重

## 题目描述

多轮对话里，每条助手回复（turn）长短悬殊：若对每个 assistant token 施加相同权重，一轮 500 token 的长回答会**淹没**一轮 5 token 的短回答，使「每一轮对话」对总损失的贡献与其长度成正比。常见做法是**逐轮归一**：给同一条 assistant 回复内部所有 token 分配 `每轮基数 / 该轮 token 数`，从而**每轮的总贡献恰好等于该轮基数**，再用 `turn_weights` 表达「越靠后的轮越重要」这类偏好。

## 规则

实现 `solve(role_ids, assistant_id, turn_weights=None)`，返回 float32、形状 \((B, L)\) 的逐 token 权重。

- `role_ids`：int64 \((B, L)\)，`role_ids[b, i]` 为该 token 的角色编号。
- 在**每一行内**，把 role 等于 `assistant_id` 的极大连续段称作一「轮」，按从左到右次序在本行内编号 \(t=0,1,2,\ldots\)（每行的轮编号独立从 0 起）。
- 设第 \(t\) 轮含 \(n_t\) 个 token，其**基数** \(c_t\)：`turn_weights` 提供则 \(c_t=\) `turn_weights[t]`（当 \(t\) 超出 `turn_weights` 长度时取 \(1.0\)）；未提供（`None`）则一律 \(c_t=1.0\)。
- 该轮每个 token 的权重 \(= c_t / n_t\)；非 assistant token（含 system/user/pad）权重 \(= 0.0\)。
- 于是每轮内权重之和恰为 \(c_t\)（与长度无关）。

## 示例

\(B=1, L=6\)，`role_ids=[[2,2,0,2,2,2]]`，`assistant_id=2`（第0轮=下标 0、1，共 2 个；第1轮=下标 3、4、5，共 3 个）：

```
turn_weights=None      -> [0.5, 0.5, 0.0, 0.33333, 0.33333, 0.33333]   # 每轮合计 1.0
turn_weights=[2.0,0.5] -> [1.0, 1.0, 0.0, 0.16667, 0.16667, 0.16667]   # 第0轮合计 2.0、第1轮合计 0.5
```

## 约束

- CPU int64 输入、float32 输出，形状严格 \((B, L)\)。
- 轮编号在每行内独立、从 0 起；`turn_weights` 按**行内轮序号**索引。
- 每轮权重之和等于该轮基数 \(c_t\)（等贡献、与 token 数成反比）。
- 非 assistant 位置恒为 `0.0`；某行无 assistant 轮则整行 `0.0`。
