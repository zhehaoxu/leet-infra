# 详解：PPO 截断价值损失

## 出处
Schulman et al., *Proximal Policy Optimization*（arXiv:1707.06347）给出 clip 目标；对 value function 采用同款截断 MSE 是标准工程实现（OpenAI Baselines / CleanRL PPO 的 `vp clipped`）。把 \(V_\theta\) 限制在旧值 ±ε 内再与目标 \(R\) 求平方误差，两误差取 max 得保守损失。

## 核心与易错点
1. **max 而非 min**：与策略 clip 用 `min`（悲观下界）不同，价值 clip 取 `max(unclipped, clipped)`——宁可高估损失、不让越界更新被奖励。写反会变成乐观截断。
2. **clip 围绕 V_old**：区间是 \([V_\text{old}-\epsilon, V_\text{old}+\epsilon]\)，不是围绕 returns。
3. **退化为 MSE**：新值都在区间内时 clip 后＝原值，两项相等 → 普通 MSE；这是「没触发截断」的自检。
4. 目标是 GAE 的 `returns`（R = A + V），不是 advantage。

## 对拍口径
参考解向量化 `maximum((vn-r)^2,(clip(vn,vo±eps)-r)^2).mean()`。判题 `_oracle` 逐元素用 Python `min(max(...))` 求 clip、`max(...)` 取较大平方误差再平均（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「大 ε 不触发截断==普通 MSE」与「越界样本被拉回」的专项断言。

## 复杂度
逐元素 \(O(N)\)。
