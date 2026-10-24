# PPO / GRPO：截断代理目标（Clipped Surrogate Objective）

## 题目描述

策略优化里，直接用"新策略相对旧策略的概率比" $\rho=\frac{\pi_\theta}{\pi_{\theta_\text{old}}}$ 乘优势 $A$ 做梯度上升，一旦某样本更新过头、$\rho$ 偏离 1 太远，就会破坏稳定性。**PPO** 的截断代理目标给 $\rho$ 加了个 $[1-\epsilon,\,1+\epsilon]$ 的信任域：

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\Big[\min\big(\rho_t A_t,\;\ \text{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big]$$

取 $\min$ 的效果：当优势为正（该动作好）时，$\rho$ 涨过 $1+\epsilon$ 就不再奖励（惩罚项封顶，防激进）；当优势为负时，$\rho$ 跌破 $1-\epsilon$ 也不再进一步打压——即"只在改善方向上限制步长"。GRPO 复用同一目标，只是优势 $A$ 改由"组内 reward 标准化"得到（本题直接给定 $A$）。**注意 $\rho=\exp(\log\pi_\theta - \log\pi_{\theta_\text{old}})$**，实现里一律给的是对数概率。作为 loss 优化时取负号。

## 规则

给定 batch（$N$）：`logp_new`、`logp_old`（均 $(N,)$，对数概率）、`advantage`（$(N,)$）、标量 `epsilon` $\epsilon$：

1. 概率比 $\rho_i = \exp($`logp_new[i]` $-$ `logp_old[i]`$)$
2. 截断比 $\hat\rho_i = \mathrm{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)$
3. 逐样本目标 $= \min(\rho_i A_i,\ \hat\rho_i A_i)$
4. loss $= -($ ③ 的均值 $)$

返回 0 维标量 `torch.Tensor`＝ batch 平均 PPO-clip loss（已取负，供最小化）。

## 示例

```
epsilon = 0.2
logp_new = [-1.0, -1.5, -0.7, -2.0]
logp_old = [-1.2, -1.3, -0.9, -1.8]
advantage = [1.0, -1.0, 0.5, -2.0]

ρ = exp(diff) = exp([0.2, -0.2, 0.2, -0.2]) = [1.2214, 0.8187, 1.2214, 0.8187]
clip(ρ,0.8,1.2) = [1.2, 0.8187, 1.2, 0.8187]
  i0 A>0: min(1.2214*1, 1.2*1)      = 1.2
  i1 A<0: min(0.8187*-1, 0.8187*-1) = -0.8187
  i2 A>0: min(1.2214*0.5, 1.2*0.5)  = 0.6
  i3 A<0: min(0.8187*-2, 0.8187*-2) = -1.6375
mean = -0.16405 → loss = 0.164048
```

## 约束

- CPU float32；$\epsilon \in (0,1)$。
- $\rho$ 由对数概率差取 `exp` 得到，不直接给概率比。
- 截断只作用于**上界当 $A>0$、下界当 $A<0$** 的方向（由 $\min$ 自动实现），返回取负后的标量。
