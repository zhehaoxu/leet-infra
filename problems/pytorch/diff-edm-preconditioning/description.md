# VE/EDM 预条件：四系数与带 churn 的 Euler 步

## 题目描述

EDM（Karras et al., 2206.00364）把扩散网络写成「对任意噪声级 \(\sigma\) 都能工作」的预条件形式，而不是让网络直接吃 \(\sigma\) 尺度各异的样本。`EDMEulerScheduler` 里有四个系数：输入侧的 `c_in` 与 `c_noise`、输出侧的 `c_skip` 与 `c_out`。网络在预条件空间给出裸输出 \(F\)，再按

\[ D(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big) \]

拼回去噪样本。采样器则是这条 \(x_0\) 预测上的 Euler 离散化，额外带一个可选的随机「churn」：先把噪声级抬到 \(\hat\sigma\) 并加一点噪声，再走一步 ODE。

本题实现「预条件 + 单步」的完整计算：给出模型输入、条件标量、去噪样本、下一步样本，以及实际使用的 \(\hat\sigma\)。

## 规则

记 \(\sigma = \text{sigmas}[i]\)、\(\sigma_d = \text{sigma\_data}\)、\(K = \text{len(sigmas)} - 1\)。

**1. churn**：只有当 \(\text{s\_tmin} \le \sigma \le \text{s\_tmax}\) 时

\[ \gamma = \min\!\left(\frac{\text{s\_churn}}{K},\ \sqrt{2}-1\right),\qquad \hat\sigma = \sigma(1+\gamma) \]

窗口外 \(\gamma = 0\)、\(\hat\sigma = \sigma\)。\(\gamma>0\) 时先加噪：

\[ x \leftarrow x + \text{noise}\cdot \text{s\_noise}\cdot\sqrt{\hat\sigma^2-\sigma^2} \]

其中 `noise` 是**显式入参**（与样本同形状的浮点张量），\(\gamma=0\) 时完全不参与计算。

**2. 输入侧系数**（用**未加噪**的 \(\sigma\)）：

\[ c_{\text{in}} = \frac{1}{\sqrt{\sigma^2+\sigma_d^2}},\qquad c_{\text{noise}} = \tfrac14\ln\sigma,\qquad \text{model\_input} = x_{\text{原始}}\cdot c_{\text{in}} \]

**3. 输出侧系数**（用**加噪后**的 \(\hat\sigma\)）：

\[ c_{\text{skip}} = \frac{\sigma_d^2}{\hat\sigma^2+\sigma_d^2},\qquad c_{\text{out}} = \pm\frac{\hat\sigma\,\sigma_d}{\sqrt{\hat\sigma^2+\sigma_d^2}} \]

`prediction_type="epsilon"` 取正号，`"v_prediction"` 取负号。

\[ \text{denoised} = c_{\text{skip}}\cdot x + c_{\text{out}}\cdot \text{model\_output} \]

**4. Euler 步**：

\[ \text{derivative} = \frac{x-\text{denoised}}{\hat\sigma},\qquad dt = \text{sigmas}[i+1] - \hat\sigma,\qquad \text{prev\_sample} = x + \text{derivative}\cdot dt \]

返回 `(model_input, c_noise, prev_sample, denoised, sigma_hat)`：前三个是与 `sample` 同形状的 float32 张量，`c_noise` 与 `sigma_hat` 是 python float。

## 示例

`sample=[[1.0, -2.0]]`、`model_output=[[0.5, 0.25]]`、`sigmas=[2.0, 1.0, 0.5, 0.0]`、`sigma_data=0.5`：

```
i=0            -> model_input=[0.485071, -0.970142]  c_noise=0.173287  sigma_hat=2.0
                 denoised=[0.301359, 0.003621]       prev=[0.65068, -0.99819]
i=1            -> c_noise=0.0                        denoised=[0.423607, -0.288197]
i=2            -> c_noise=-0.173287                  denoised=[0.676777, -0.911612]  prev 与 denoised 相同
s_churn=10     -> sigma_hat=2.828427                 model_input 仍为 [0.485071, -0.970142]
                 denoised=[0.282547, 0.044304]       prev=[0.606915, -0.890599]
```

第 0 步里 \(c_{\text{in}} = 1/\sqrt{4.25} = 0.485071\)，恰好与 \(c_{\text{out}}\) 在 \(\sigma_d=0.5\) 时相等——这不是巧合，而是 \(\epsilon\) 支下 \(c_{\text{out}} = c_{\text{in}}\cdot\sigma\sigma_d\) 且 \(\sigma\sigma_d=1\)。第 2 步是最后一步，\(\text{sigmas}[3]=0\) 使 \(dt=-\hat\sigma\)，于是 `prev_sample` 精确等于 `denoised`。`s_churn=10` 时 \(\gamma\) 被 \(\sqrt2-1\) 截断，\(\hat\sigma = 2\sqrt2\)；注意 `model_input` 仍用原始样本与 \(\sigma=2\) 计算，只有 `denoised`/`prev` 用加噪后的样本与 \(\hat\sigma\)。

## 约束

- `sample` 与 `model_output` 必须是同形状的非空浮点张量；`sigmas` 必须是一维浮点张量、长度至少 2、有限非负且单调不增。
- `step_index` 是整数（`bool` 不算），落在 \([0, K-1]\)；当前步的 \(\sigma\) 必须为正。
- `sigma_data` 必须是有限正数；`s_churn` / `s_noise` 必须为有限非负数；`s_tmin \le s_tmax`；`s_tmax` 默认 \(+\infty\)。
- `prediction_type` 只接受 `"epsilon"` / `"v_prediction"`，其余抛 `ValueError`。
- \(\gamma>0\) 时必须显式给出与 `sample` 同形状的浮点 `noise`，否则抛 `ValueError`；\(\gamma=0\) 时忽略 `noise`。
- 内部用 float64 计算，三个张量输出为 float32。
