# EulerAncestral：σ_up/σ_down 拆分与祖先噪声注入

## 题目描述

Euler Ancestral 采样器把 Euler ODE 步改成「每步重新加一点噪声」的 SDE 步：从 \(\sigma_{\text{from}}\) 退到 \(\sigma_{\text{to}}\) 时，不把噪声级一次降到底，而是把这段下降拆成两部分——一部分走确定性的 Euler 步到 \(\sigma_{\text{down}}\)，另一部分以新噪声的形式注入，幅度为 \(\sigma_{\text{up}}\)。这就是「祖先采样」：每一步都在往分布里补随机性，所以不同的随机种子会给出不同的图，而 \(x_0\) 预测仍然沿着模型学到的方向前进。

本题实现 `EulerAncestralDiscreteScheduler.step` 的完整单步：模型输入缩放、\(x_0\) 预测、\(\sigma_{\text{up}}/\sigma_{\text{down}}\) 拆分、Euler 更新与噪声注入。

## 规则

记 \(\sigma = \text{sigmas}[i]\)、\(\sigma' = \text{sigmas}[i+1]\)，样本为 \(x\)，网络输出为 `model_output`。

**1. 模型输入端缩放**（上游 `scale_model_input`）：

\[ \text{scaled\_input} = \frac{x}{\sqrt{\sigma^2+1}} \]

**2. \(x_0\) 预测**：

- `epsilon`：\(x_0 = x - \sigma\,\text{model\_output}\)
- `v_prediction`：\(x_0 = \text{model\_output}\cdot\dfrac{-\sigma}{\sqrt{\sigma^2+1}} + \dfrac{x}{\sigma^2+1}\)

（第二支**照抄上游实现**，注意样本项是 \(x/(\sigma^2+1)\) 而不是 \(x/\sqrt{\sigma^2+1}\)；详见 notes。）

**3. 噪声级拆分**：

\[ \sigma_{\text{up}} = \sqrt{\frac{\sigma'^2\,(\sigma^2-\sigma'^2)}{\sigma^2}},\qquad \sigma_{\text{down}} = \sqrt{\sigma'^2-\sigma_{\text{up}}^2} \]

**4. Euler 步 + 注入噪声**：

\[ \text{derivative} = \frac{x-x_0}{\sigma},\qquad \text{prev\_sample} = x + \text{derivative}\cdot(\sigma_{\text{down}}-\sigma) + \text{noise}\cdot\sigma_{\text{up}} \]

`noise` 是**显式入参**（与样本同形状的浮点张量），\(\sigma_{\text{up}}=0\) 时完全不参与计算，此时可以不传。

返回 `(scaled_input, prev_sample, pred_original_sample, sigma_up, sigma_down)`：前三个是与 `sample` 同形状的 float32 张量，后两个是 python float。

## 示例

`sample=[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]`、`model_output=[[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]]`、`sigmas=[2.0, 1.0, 0.5, 0.0]`：

```
i=0  sigma_up=0.866025 sigma_down=0.5   scaled=[[0.447214, -0.894427, 0.223607], ...]
     x0=[[0.0, -2.5, 2.0], [0.05, 3.4, -2.1]]
     prev=[[0.683013, -2.808013, 1.841506], [-0.116506, 3.949519, -1.841747]]
i=1  sigma_up=0.433013 sigma_down=0.25
     prev=[[0.841506, -2.404006, 1.170753], [0.066747, 3.47476, -1.670873]]
i=2  sigma_up=0.0      sigma_down=0.0     prev 与 x0 完全相同（不需要 noise）
```

第 0 步里 \(\sigma_{\text{up}} = \sqrt{1^2(4-1)/4} = \sqrt{0.75} = 0.866025\)，\(\sigma_{\text{down}} = \sqrt{1-0.75} = 0.5\)，两者平方和恰好等于 \(\sigma'^2=1\)。最后一步 \(\sigma'=0\) 使两项都归零，\(dt=-\sigma\)，于是 `prev_sample` 精确等于 \(x_0\)。

## 约束

- `sample` / `model_output` 必须是同形状的非空浮点张量；`sigmas` 必须是一维浮点张量、长度至少 2、有限非负且单调不增。
- `step_index` 是整数（`bool` 不算），落在 \([0, K-1]\)；当前步的 \(\sigma\) 必须为正。
- `prediction_type` 只接受 `"epsilon"` / `"v_prediction"`，其余抛 `ValueError`（上游对 `"sample"` 抛 `NotImplementedError`，本题统一用 `ValueError`）。
- \(\sigma_{\text{up}}>0\) 时必须显式给出与 `sample` 同形状的浮点 `noise`，否则抛 `ValueError`；\(\sigma_{\text{up}}=0\) 时忽略 `noise`。
- 内部用 float64 计算，三个张量输出为 float32。
