# Diffusion：DDPM 反向后验一步 p(x_{t-1}|x_t,x_0)

## 题目描述

采样时要把 \(x_t\) 去噪回 \(x_{t-1}\)。若模型已给出对干净样本的预测 \(\hat x_0\)，则真实的逆后验 \(q(x_{t-1}\mid x_t, x_0)\) 是高斯的，均值、方差有闭式（Ho et al. DDPM 式 (7)–(11)）：

\[ \mu_\theta = \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat x_0, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t \]

一步采样 \(x_{t-1} = \mu_\theta + \sqrt{\tilde\beta_t}\,z\)，\(z\sim\mathcal N(0,I)\)（作显式入参保确定性）。其中 \(\alpha_t=1-\beta_t\)。

## 规则

给定 `x_t`、`x0_pred`（模型预测的 \(\hat x_0\)，同形状）、标量 `beta_t`、`alpha_bar_t`（\(\bar\alpha_t\)）、`alpha_bar_prev`（\(\bar\alpha_{t-1}\)）、`z`（同形状）。返回 `x_prev`（\(x_{t-1}\)，同形状）。float32。禁止现成扩散库。

## 示例

\[ x_t=\begin{bmatrix}0.4&-1.1\\ 0.9&2.2\end{bmatrix},\ \hat x_0=\begin{bmatrix}0.7&-1.5\\ 1.1&2.6\end{bmatrix},\ \beta_t=0.02,\ \bar\alpha_t=0.65,\ \bar\alpha_{t-1}=0.72,\ z=\begin{bmatrix}0.2&-0.3\\ 0.5&0.1\end{bmatrix} \]

```
x_prev = [[0.37602317333221436, -0.9818338751792908],
          [0.8293452262878418 ,  1.8810272216796875 ]]
```

## 约束

- 标量 \(0<\bar\alpha_t\le\bar\alpha_{t-1}\le1\)（越靠前信号越多），\(\beta_t\in(0,1)\)；\(\alpha_t=1-\beta_t\)。
- \(z=0\) 时输出即后验均值 \(\mu_\theta\)。
- 两系数满足：\(x_t\) 权重 \(\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\)、\(\hat x_0\) 权重 \(\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\)，方差 \(\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\)。
