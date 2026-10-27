# Diffusion：DPM-Solver++(2M) 二阶更新（含 Euler 一阶）

## 题目描述

DPM-Solver++（Lu et al.）在**数据预测**（x₀-prediction）形式下把去噪 ODE 写成指数积分器：令 \(\alpha_t=\sqrt{\bar\alpha_t}\)、\(\sigma_t=\sqrt{1-\bar\alpha_t}\)、\(\lambda_t=\log(\alpha_t/\sigma_t)\)（即半个 log-SNR），从 \(\bar\alpha_{s_0}\) 走到 \(\bar\alpha_t\) 的一步为

\[ x_t = \frac{\sigma_t}{\sigma_{s_0}}\,x_{s_0} - \alpha_t\left(e^{-h}-1\right)\left(D_0 + \tfrac{1}{2}D_1\right),\qquad h=\lambda_t-\lambda_{s_0} \]

其中 \(D_0\) 是当前步的 x₀ 预测。2M（multistep 二阶）用上一级预测做一阶差分外推：

\[ D_1 = \frac{D_0 - D_0^{\text{prev}}}{r_0},\qquad r_0=\lambda_{s_0}-\lambda_{s_1} \]

`order=1` 时丢掉 \(D_1\)。一阶在数学上等价于 DDIM 的确定性闭式——这正是这题最值得记住的一点：看起来完全不同的两种写法是同一条指数积分器。

## 规则

给定 `x`（当前样本 \(x_{s_0}\)）、`x0_pred`（\(D_0\)）、标量 `alpha_bar_t`、`alpha_bar_s0`，以及 `order`；`order=2` 时还要给 `x0_pred_prev`（\(D_0^{\text{prev}}\)）与 `alpha_bar_s1`。返回与 `x` 同形状的 float32 张量（内部 float64）。禁止直接用现成采样器。

## 示例

\[ x=1.0,\ D_0=0.5,\ D_0^{\text{prev}}=0.4,\ \bar\alpha_t=0.81,\ \bar\alpha_{s_0}=0.64,\ \bar\alpha_{s_1}=0.49 \]

```
order=1 -> 0.885890
order=2 -> 0.937698
```

## 约束

- \(\bar\alpha\) 必须严格落在 \((0,1)\) 内；`order` 只能是 1 或 2。
- `order=2` 要求 \(r_0>0\)，即 \(s_1\) 必须是比 \(s_0\) 更噪的上一级（\(\bar\alpha_{s_1}<\bar\alpha_{s_0}\)）；否则抛 `ValueError`。
- **退化性质**：\(\bar\alpha_t=\bar\alpha_{s_0}\) 时输出原样等于输入；\(\bar\alpha_t\to1\) 时收敛到 \(D_0\)；两次预测相同时二阶自动退化成一阶。
- 一步不得越过 x₀ 预测（系数与 \(\sigma\) 比都为正，结果落在 \(x\) 与 \(D_0\) 之间）。
