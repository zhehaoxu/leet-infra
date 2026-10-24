# 解题说明 · Adaptive-p 采样

## 变形而非截断

adaptive-p 把「采样偏好」编码进一个**峰函数**：以目标概率 $p^\star$ 为中心，token 的原始概率
离 $p^\star$ 越近，新 logit 越高。

$$\text{new\_logit}_i = \text{PEAK} - \text{SHARPNESS}\cdot\frac{\text{dist}_i^2}{1+\text{dist}_i},\quad
\text{dist}_i = |p_i - p^\star|\cdot \text{INV\_WIDTH}$$

- 近峰处 $\text{dist}^2$ 项 → **二次**精细区分（把最接近 $p^\star$ 的几个拉开）。
- 远处 $/(1+\text{dist})$ 让增长退化为**线性**，避免 softmax 下远端爆掉、又不彻底清零。
- 全程不删候选，`filtered` 覆盖整个词表——这是它和 top-p/min-p 的本质区别。

## adapted_target 的反馈

EMA 记的是「过去每一步实际抽中 token 的**原始**概率」的加权平均
$\bar p = \text{weighted\_sum}/\text{total\_weight}$。目标写成

$$p^\star = \text{clamp}(2\cdot\text{target} - \bar p,\ 0, 1)$$

于是 $\bar p$ 偏高（历史分布太尖）→ $p^\star$ 被压低 → 峰移到小概率区 → 下一步分布更平，
反过来亦然。围绕 `target` 形成负反馈。`total_weight==0`（首步）时 $p^\star=\text{target}$。

## EMA 更新（最易错）

$$\text{weighted\_sum}' = p_{\text{orig}}[\text{chosen}] + \text{decay}\cdot\text{weighted\_sum},\quad
\text{total\_weight}' = 1 + \text{decay}\cdot\text{total\_weight}$$

务必用 **`softmax(logits)` 的 `probs[chosen]`（原始概率）**，不是变形后的 `filtered[chosen]`
（源码 `accept()` 里用的是缓存的 `original_probs[pending_token_idx]`）。`total_weight` 每步加 1
并几何衰减，故 $\bar p$ 是近期命中概率的滑动平均。

## 契约

与合集一致：`filtered` 原顺序返回、`chosen` 由原顺序 CDF 逆变换得到；额外把两个 EMA 状态
作为返回的第 3、4 项，供下一步作入参。
