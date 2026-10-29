# LCM 边界条件系数与少步去噪单步

## 题目描述

Latent Consistency Model（LCM）把扩散模型蒸馏成「4 步甚至 1 步」的采样器。它沿用 EDM 的**边界条件预条件**：网络预测的 \(x_0\) 不直接当答案，而是与当前样本按两个系数 \(c_{\text{skip}}\)、\(c_{\text{out}}\) 混合，使去噪结果在噪声级两端自动满足一致性条件。系数只由**离散 timestep** 决定，并按 `timestep_scaling`（默认 10）拉伸。

本题实现 `LCMScheduler.step` 的完整单步：由 timestep 算出边界条件系数、按预测类型换算 \(x_0\)、裁剪、混合，再按下一步的 \(\bar\alpha\) 重加噪。

## 规则

记 \(\bar\alpha_t = \text{alphas\_cumprod}[t]\)，\(t = \text{timesteps}[i]\)（\(i\) = `step_index`）。

**1. 边界条件系数**（\(\sigma_d\) 上游写死 0.5）：

\[ s = t\cdot\text{timestep\_scaling},\qquad c_{\text{skip}} = \frac{\sigma_d^2}{s^2+\sigma_d^2},\qquad c_{\text{out}} = \frac{s}{\sqrt{s^2+\sigma_d^2}} \]

注意 \(c_{\text{out}}\) **没有**乘 \(\sigma_d\)，这是 LCM 与 EDM 的差别。

**2. \(x_0\) 换算**（用 \(\bar\alpha_t\) 的平方根）：

- `epsilon`：\(x_0 = \dfrac{x-\sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}\)
- `sample`：\(x_0 = \text{model\_output}\)
- `v_prediction`：\(x_0 = \sqrt{\bar\alpha_t}\,x-\sqrt{1-\bar\alpha_t}\,\text{model\_output}\)

`clip_sample=True` 时再把 \(x_0\) 钳到 \([-\text{clip\_sample\_range}, \text{clip\_sample\_range}]\)。

**3. 混合**：\(\text{denoised} = c_{\text{out}}x_0 + c_{\text{skip}}x\)。

**4. 重加噪**：记 \(\bar\alpha_{\text{prev}} = \text{alphas\_cumprod}[\text{timesteps}[i+1]]\)（下标不存在时沿用当前 \(t\)，该值在最后一步不参与计算；下标为负时取 1.0）。

- 非最后一步：\(\text{prev} = \sqrt{\bar\alpha_{\text{prev}}}\,\text{denoised} + \sqrt{1-\bar\alpha_{\text{prev}}}\,\text{noise}\)
- 最后一步（\(i = \text{len(timesteps)}-1\)）：\(\text{prev} = \text{denoised}\)，不消耗 `noise`

`noise` 是显式入参（与样本同形状的浮点张量）。

返回 `(denoised, prev_sample, c_skip, c_out)`：前两个是与 `sample` 同形状的 float32 张量，后两个是 python float。

## 示例

`alphas_cumprod=[0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.06, 0.02, 0.005, 0.001]`、`timesteps=[9, 5, 2]`、`sample=[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]`、`model_output=[[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]]`：

```
i=0  t=9  c_skip=3.086324e-05 c_out=0.999985
     denoised=[[1.000015, -1.000046, 1.0], [0.999992, 1.000077, -1.000031]]
     prev=[[0.848282, -0.848293, 0.617787], [0.156807, 1.078794, -0.272066]]
i=1  t=5  c_skip=9.999000e-05 c_out=0.999950
i=2  t=2  c_skip=6.246096e-04 c_out=0.999688
     denoised=[[0.868255, -1.000937, 1.0], [0.233425, 1.001562, -1.000625]]
     prev 与 denoised 相同（最后一步不重加噪）
```

`t=9` 时 \(s=90\)，\(c_{\text{skip}}=0.25/8100.25=3.086\times10^{-5}\)；同一条公式在 EDM 里还要再乘 \(\sigma_d=0.5\)，所以这里的高噪声端 \(c_{\text{out}}\) 几乎是 1。最后一步的 `prev` 就是 `denoised`，与传入的 `noise` 无关。

## 约束

- `sample` / `model_output` 必须是同形状的非空浮点张量。
- `alphas_cumprod` 必须是一维浮点张量、非空、全部有限且元素落在 \((0,1)\)。
- `timesteps` 必须是一维**整数**非空张量，每个取值都在 \([0, T-1]\)（\(T\) = `alphas_cumprod` 长度）；允许重复、允许非单调。
- `step_index` 是整数（`bool` 不算），落在 \([0, \text{len(timesteps)}-1]\)。
- `prediction_type` 只接受 `epsilon` / `sample` / `v_prediction`；`clip_sample` 必须是 `bool`；`clip_sample_range` / `timestep_scaling` 必须是有限正数。
- 非最后一步必须显式给出与 `sample` 同形状的浮点 `noise`，否则抛 `ValueError`。
- 内部用 float64 计算，两个张量输出为 float32。
