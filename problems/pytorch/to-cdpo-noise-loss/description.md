# cDPO：带标签噪声的偏好损失

## 题目描述

真实偏好数据并不干净：标注者会点错、`chosen`/`rejected` 可能被写反。**cDPO**（conservative DPO，Eric Mitchell 2023 年博客 *A note on DPO with noisy preferences & relationship to IPO* 提出；噪声偏好的理论分析见 Ray Chowdhury et al. 2024, *Provably Robust DPO: Aligning Language Models with Noisy Feedback*, arXiv:2403.00409）的处理方式就是最朴素的**标签平滑**：假设每条偏好标签以概率 $\varepsilon$ 被翻转，于是硬标签 $\{1,0\}$ 变成软标签 $\{1-\varepsilon,\varepsilon\}$，DPO 的负对数似然随之变成

$$\mathcal{L}_\text{cDPO}=-\mathbb{E}\Big[(1-\varepsilon)\log\sigma(\beta\Delta)+\varepsilon\log\sigma(-\beta\Delta)\Big],\qquad \Delta=(\text{logp}_w-\text{ref}_w)-(\text{logp}_l-\text{ref}_l).$$

它不改变 DPO 的结构，只在「分对了」的样本上保留一份把 logit 往反方向拉的力，从而抑制模型对疑似错误标签的过拟合。$\varepsilon=0$ 严格退化为 DPO。

本题给定四条**序列级**对数概率，组装上式即可。

## 规则

给定长度 $N\ge 1$ 的批次，四条序列级对数概率形状均为 $(N,)$、dtype 一致：

1. 逐样本隐式奖励差 $\Delta=(\text{logp}_w-\text{ref}_w)-(\text{logp}_l-\text{ref}_l)$，logit $x=\beta\Delta$。
2. 损失 $=\frac{1}{N}\sum\big[-(1-\varepsilon)\log\sigma(x)-\varepsilon\log\sigma(-x)\big]$。
3. 两个 log-sigmoid 都必须走**稳定式**（等价 `F.logsigmoid`），不允许先算 $\sigma$ 再取 log。
4. $\varepsilon\in[0,0.5]$；$\beta>0$。返回 0 维 float64 标量。

## 示例

```
logp_w = [-1.0]   ref_w = [-1.2]
logp_l = [-3.0]   ref_l = [-2.8]
β = 0.5, ε = 0.1

Δ = (-1.0 + 1.2) - (-3.0 + 2.8) = 0.4
x = β·Δ = 0.2
logσ(0.2)  = -0.598139
logσ(-0.2) = -0.798139
loss = -(0.9 × -0.598139 + 0.1 × -0.798139) = 0.618139
```

作为对照，同一组输入取 $\varepsilon=0$ 时损失为 $0.598139$——噪声项把它抬高了，这正是「对已分对样本保留反向拉力」的体现。若 $\Delta=0$（$x=0$），无论 $\varepsilon$ 取多少损失都等于 $\ln 2$。

## 约束

- 返回 0 维 float64；float32 输入内部升到 float64。
- $\varepsilon=0$ 必须与 DPO 数值一致；$\varepsilon=0.5$ 时软标签对称，损失为 $-\frac12(\log\sigma(x)+\log\sigma(-x))$。
- 已分对（$x>0$）时，$\varepsilon$ 增大必须使损失严格增大。
- 非法输入抛 `ValueError`：非张量 / 非一维 / 非浮点 / 形状或 dtype 不一致 / 空样本 / 含 NaN·Inf / $\beta\le 0$ / $\varepsilon\notin[0,0.5]$。
- CPU、确定性、不调用随机数。
