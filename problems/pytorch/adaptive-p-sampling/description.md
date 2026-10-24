# Adaptive-p 采样

## 题目描述

截断类采样（top-k/p）只会「砍尾巴」；**Adaptive-p**（llama.cpp 的 `adaptive-p`）反其道而行——
它**不删任何候选**，而是把整条分布**重新塑形**：用一条以「目标概率 $p^\star$」为峰的钟形函数
给每个 token 赋新 logit，谁的原始概率最接近 $p^\star$，谁就被顶到分布最前。

更妙的是 $p^\star$ 会**自适应**：维护「历史采样命中概率」的指数滑动平均（EMA），若最近总是抽到
高概率 token（说明当下分布太尖），就**抬高** $p^\star$、把分布压得更平以增多样性；反之**压低**。
因此本题带两个状态量 `weighted_sum` / `total_weight` 作显式入/出参。

> 为保证判题**确定性**，均匀随机数 `u` 作显式输入；EMA 状态既入又出。

## 常量

`DISTRIBUTION_WIDTH = 0.3`，`PEAK_LOGIT_VALUE = 5.0`，`SHARPNESS = 10.0`，
`INV_WIDTH = 1 / 0.3`。

## 规则

- 实现 `solve(logits, target, decay, weighted_sum, total_weight, u)`：`logits` 形状 `(..., V)`，
  `target`/`decay`/`weighted_sum`/`total_weight` 为标量，`u` 取值 $[0,1)$。
- 步骤（逐行）：
  1. `probs = softmax(logits)`（这份**原始**概率要留给第 6 步 EMA）。
  2. $\text{adapted\_target} = \text{clamp}\big(\text{total\_weight}==0\,?\ \text{target} : 2\cdot\text{target} - \tfrac{\text{weighted\_sum}}{\text{total\_weight}},\ 0,\ 1\big)$。
  3. $\text{dist}_i = |p_i - \text{adapted\_target}| \cdot \text{INV\_WIDTH}$；
     $\text{new\_logit}_i = \text{PEAK} - \text{SHARPNESS}\cdot\dfrac{\text{dist}_i^2}{1+\text{dist}_i}$。
  4. `filtered = softmax(new_logit)`（和为 1，未删任何候选）。
  5. 用 `u` 在 `filtered`（原顺序）CDF 上逆变换采样得 `chosen`。
  6. EMA 用**原始** `probs[chosen]` 更新：
     `weighted_sum_new = probs[chosen] + decay*weighted_sum`；`total_weight_new = 1 + decay*total_weight`。
- 返回 `(filtered, chosen, weighted_sum_new, total_weight_new)`。

## 示例

```
输入：logits = [2, 1, 0.5, -1, -2, -3]，target = 0.5，decay = 0.9，
      weighted_sum = 0.0，total_weight = 0.0，u = 0.5
  probs(softmax) = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  total_weight==0 → adapted_target = target = 0.5
  dist = |probs-0.5|/0.3 = [0.3343, 0.9305, 1.2202, 1.5670, 1.6300, 1.6532]
  new_logit = 5 - 10*dist²/(1+dist) = [4.1624, 0.5147, -1.7060, -4.5660, -5.1024, -5.3009]
  filtered = softmax(new_logit) = [0.971617, 0.025311, 0.002747, 0.000157, 0.000092, 0.000075]
  CDF 首项已 ≥ u(0.5) → chosen = 0
  EMA: weighted_sum_new = probs[0] + 0.9*0 = 0.600294；total_weight_new = 1 + 0.9*0 = 1.0
输出：filtered = [0.971617, 0.025311, 0.002747, 0.000157, 0.000092, 0.000075], chosen = 0,
      weighted_sum_new = 0.600294, total_weight_new = 1.0
```

## 约束

- float32 输入；`target ∈ [0,1]`、`decay ∈ [0,1]`；`total_weight ≥ 0`；`u ∈ [0,1)`
- `filtered` 每行和为 1（**不置 0**，全部候选都在）；`chosen` 为 `filtered` 逆变换命中位置
- EMA 更新务必用 `softmax(logits)`（原始 probs）的 `chosen` 项，而**非** `filtered[chosen]`
- `total_weight==0` 时 `adapted_target` 退化为 `target`
