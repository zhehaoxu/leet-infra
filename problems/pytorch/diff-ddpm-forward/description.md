# Diffusion：DDPM 前向加噪 q(x_t|x_0)

## 题目描述

DDPM 的前向过程「一步到位」地把干净样本 \(x_0\) 加噪到第 \(t\) 步：不必真的迭代 \(t\) 次，因为高斯可合并，结果仍是均值、方差已知的正态分布——

\[ x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I) \]

其中 \(\bar\alpha_t\in(0,1]\) 是信号保留比例（由噪声调度给出）。噪声 \(\epsilon\) 作为显式入参传入以保证判题确定性。

## 规则

给定 `x0`、同形状的 `eps`、标量 `alpha_bar`（\(\bar\alpha_t\)）。返回

\[ x_t = \sqrt{\bar\alpha_t}\cdot x_0 + \sqrt{1-\bar\alpha_t}\cdot \epsilon \]

形状与 `x0` 相同，float32。禁止现成扩散库。

## 示例

`alpha_bar=0.6`、\(x_0=\begin{bmatrix}1.0&-2.0\\ 0.5&3.0\end{bmatrix}\)、\(\epsilon=\begin{bmatrix}0.3&-0.7\\ 1.2&-0.4\end{bmatrix}\)：

```
x_t = [[0.9643333554267883, -1.9919122457504272],
       [1.146245002746582 ,  2.070807933807373 ]]
```

## 约束

- `alpha_bar` 为标量 float，取值 \((0,1]\)；`x0` 与 `eps` 同形状。
- \(\bar\alpha_t=1\) 时 \(x_t=x_0\)（无噪声）；\(\bar\alpha_t\to0\) 时趋近纯噪声方向 \(\epsilon\)。
- 全程 float32，逐元素运算，支持任意形状（含高维）。
