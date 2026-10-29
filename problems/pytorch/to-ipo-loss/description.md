# IPO：Identity Preference Optimization 平方损失

## 题目描述

**IPO**（Azar et al. 2024, *A General Theoretical Paradigm to Understand Learning from Human Preferences*, arXiv:2310.12036）指出 DPO 的 logistic 偏好损失有一个理论缺陷：它对「chosen 已经明显优于 rejected」的样本仍持续提供梯度，把隐式奖励差越推越大，最后退化成对偏好数据的过拟合。IPO 的修法极其直接——把隐式奖励差 $\Delta$ 拉向一个**确定的目标边际** $1/(2\beta)$，用平方损失替代 sigmoid：

$$\mathcal{L}_\text{IPO}=\mathbb{E}\Big[\big(\Delta-\tfrac{1}{2\beta}\big)^2\Big],\qquad \Delta=(\log\pi_\theta(y_w)-\log\pi_\text{ref}(y_w))-(\log\pi_\theta(y_l)-\log\pi_\text{ref}(y_l)).$$

这样 $\Delta$ 到达目标后梯度恰好为零，模型既不会继续拉开差距，也不会在目标附近来回震荡。

本题给定四条**序列级**对数概率（chosen/rejected × 当前/参考），只需组装 $\Delta$ 与平方项。

## 规则

给定长度 $N\ge 1$ 的批次，四条序列级对数概率形状均为 $(N,)$、dtype 一致：

1. 逐样本计算隐式奖励差
   $$\Delta=(\text{logp}_w-\text{ref}_w)-(\text{logp}_l-\text{ref}_l).$$
   入参已是整条回答的 logprob 之和，**不再做长度归一**。
2. 损失 $=\frac{1}{N}\sum(\Delta-\frac{1}{2\beta})^2$，$\beta>0$。
3. **不乘 $\beta$**：与 DPO 的 $-\log\sigma(\beta\Delta)$ 不同，$\beta$ 只通过目标边际 $1/(2\beta)$ 进入，整体缩放会改变损失尺度但不改变最优解。
4. 返回 0 维 float64 标量。

## 示例

```
logp_w = [-1.0, -2.0]   ref_w = [-1.2, -1.8]
logp_l = [-3.0, -2.6]   ref_l = [-2.8, -2.4]
β = 0.5  ->  目标边际 1/(2β) = 1.0

样本 0：Δ = (-1.0 + 1.2) - (-3.0 + 2.8) = 0.2 - (-0.2) = 0.4
样本 1：Δ = (-2.0 + 1.8) - (-2.6 + 2.4) = -0.2 - (-0.2) = 0.0
损失 = mean([(0.4-1)², (0.0-1)²]) = mean([0.36, 1.0]) = 0.68
```

若误把 $\beta$ 乘在整体损失上，$\beta=0.5$ 时结果会变成 0.34；若目标写成 $1/\beta$ 或漏掉平方，数值都对不上。

## 约束

- 返回 0 维 float64；float32 输入内部升到 float64。
- $\Delta$ 恰等于 $1/(2\beta)$ 时损失为 0（$\beta=0.5$ 对应 1.0，$\beta=0.25$ 对应 2.0）。
- 参考值只通过 $\Delta$ 进入：两侧 ref 同时平移 $c$，$\Delta$ 不变、损失不变；交换偏好/被拒两侧并同时交换 ref，$\Delta$ 不变，损失也不变。
- 非法输入抛 `ValueError`：非张量 / 非一维 / 非浮点 / 形状或 dtype 不一致 / 空样本 / 含 NaN·Inf / $\beta\le 0$。
- CPU、确定性、不调用随机数。
