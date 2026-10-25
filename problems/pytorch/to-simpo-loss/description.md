# 对齐训练：SimPO 长度归一化偏好损失

## 题目描述

**SimPO**（Simple Preference Optimization, Meng et al., arXiv:2405.14734）用**长度归一化**的平均对数概率当「奖励」，去掉 DPO 的参考模型，并加一个**目标边际** \(\gamma\) 拉开优劣样本：

\[ r_\theta(x,y) = \frac{1}{|y|}\log \pi_\theta(y|x) \]

损失（对偏好对 \(y_w \succ y_l\) 的 sigmoid 型目标）：

\[ \mathcal{L} = -\log\sigma\bigl(\beta\,(r_\theta(x,y_w) - r_\theta(x,y_l)) - \gamma\bigr) \]

其中给定的 `logp_w`、`logp_l` 是序列级对数概率 \(\log\pi_\theta(y|x)\)，除以响应长度即得 \(r\)。

## 规则

输入同形状 \((N,)\) 的 `logp_w`、`logp_l`（序列对数概率）、`len_w`、`len_l`（响应 token 数），标量 `beta`（温度）、`gamma`（目标边际）。逐样本
\[ r_w=\frac{\text{logp}_w}{\text{len}_w},\quad r_l=\frac{\text{logp}_l}{\text{len}_l},\quad \mathcal{L}=-\frac1N\sum_i\log\sigma\bigl(\beta(r_{w,i}-r_{l,i})-\gamma\bigr) \]
返回 0 维 float32 标量。禁止现成对齐库。

## 示例

`logp_w=[-2.0,-1.5]`、`len_w=[4,3]`、`logp_l=[-3.0,-2.5]`、`len_l=[5,4]`、`beta=2.0`、`gamma=0.5`：

```
r_w = [-0.5, -0.5] ;  r_l = [-0.6, -0.625]
margin = 2*(r_w-r_l) - 0.5 = [-0.3, -0.25]
loss = -mean(logsigmoid(-0.3), logsigmoid(-0.25)) = 0.840147
```

## 约束

- CPU float32；`len_w`、`len_l` 为正；\(\beta>0\)。
- 长度归一化是 SimPO 相对 DPO 的关键（消除「长回答累加更多对数概率」的偏置）。
- \(\gamma=0\) 且奖励改用未归一化、含参考模型时接近 DPO；本题只按上式实现。
