# Diffusion：DDIM 确定性去噪一步

## 题目描述

DDIM（Song et al., arXiv:2010.02502）把 DDPM 的随机采样改成**确定性**的非马尔可夫逆过程：取 \(\sigma_t=0\)，同一 \(\hat x_0\) 预测下不再注入噪声，因而可用更少步数、并可从同一噪声「反演」出确定结果。

给定模型预测的噪声 \(\epsilon_\theta\)，先反解出干净样本估计：

\[ \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}} \]

再直接跳向 \(t-1\)（\(\sigma_t=0\) 的 DDIM 更新，Song 式 (12)）：

\[ x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta \]

## 规则

给定 `x_t`、`eps_pred`（\(\epsilon_\theta\)，同形状）、标量 `alpha_bar_t`（\(\bar\alpha_t\)）、`alpha_bar_prev`（\(\bar\alpha_{t-1}\)）。返回 `x_prev`（同形状）。float32，不含随机项。禁止现成扩散库。

## 示例

\[ x_t=\begin{bmatrix}0.4&-1.1\\ 0.9&2.2\end{bmatrix},\ \epsilon_\theta=\begin{bmatrix}0.3&-0.7\\ 1.2&-0.4\end{bmatrix},\ \bar\alpha_t=0.65,\ \bar\alpha_{t-1}=0.72 \]

```
x_prev = [[0.39293810725212097, -1.0922671556472778],
          [0.8350236415863037 ,  2.352832794189453 ]]
```

## 约束

- 标量 \(0<\bar\alpha_t\le\bar\alpha_{t-1}\le1\)。
- **退化性质**：\(\bar\alpha_{t-1}=\bar\alpha_t\) 时更新是恒等（\(x_{t-1}=x_t\)）；\(\bar\alpha_{t-1}=1\) 时 \(x_{t-1}=\hat x_0\)（一步完全去噪到干净估计）。
- 若 `x_t` 恰由某 \(x_0\) 按前向公式与同一 `eps_pred` 生成，则 \(\hat x_0\) 精确还原该 \(x_0\)。
