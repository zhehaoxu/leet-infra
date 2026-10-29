# DPM-Solver 的 x₀/ε 口径转换与动态阈值

## 题目描述

DPM-Solver 与 DPM-Solver++ 对**模型输出口径**的要求正好相反：前者离散化的是噪声预测 \(\epsilon_\theta\) 的积分，后者离散化的是数据预测 \(x_0\) 的积分。同一份权重可以配任意一种口径，所以调度器必须在进入数值积分前统一把网络裸输出换算好——这段逻辑就是 `convert_model_output`。

同一函数还承担一件独立的事：**动态阈值**（Imagen, 2205.11487）。高 guidance scale 下预测的 \(x_0\) 会出现饱和像素，动态阈值按每个样本的分位数把预测压回 \([-1,1]\)，再换算回去。

本题实现这两个步骤：按 `algorithm_type` / `prediction_type` 把裸输出换成 \(x_0\) 或 \(\epsilon\)，并按开关决定是否插入阈值化。

## 规则

记当前噪声级为 \(\sigma\)，样本为 \(x\)，网络裸输出为 `model_output`。

**1. \(\alpha/\sigma_t\) 换算**：

\[ \text{use\_flow\_sigmas} \Rightarrow \alpha = 1-\sigma,\ \sigma_t = \sigma; \qquad \text{否则} \Rightarrow \alpha = \frac{1}{\sqrt{\sigma^2+1}},\ \sigma_t = \sigma\alpha \]

**2. 数据口径**（`algorithm_type` 属于 `{"dpmsolver++", "sde-dpmsolver++"}`），输出为 \(x_0\)：

- `epsilon`：\(x_0 = (x - \sigma_t\,\epsilon)/\alpha\)
- `sample`：\(x_0 = \text{model\_output}\)
- `v_prediction`：\(x_0 = \alpha x - \sigma_t\,\text{model\_output}\)
- `flow_prediction`：\(x_0 = x - \sigma\,\text{model\_output}\)（用**原始 \(\sigma\)**，不经过 \(\alpha/\sigma_t\)；此时 `model_output` 是速度 \(v=\epsilon-x_0\)）

**3. 噪声口径**（`algorithm_type` 属于 `{"dpmsolver", "sde-dpmsolver"}`），输出为 \(\epsilon\)：

- `epsilon`：\(\epsilon = \text{model\_output}\)
- `sample`：\(\epsilon = (x - \alpha\,\text{model\_output})/\sigma_t\)
- `v_prediction`：\(\epsilon = \alpha\,\text{model\_output} + \sigma_t x\)
- `flow_prediction`：不支持，抛 `ValueError`

**4. 动态阈值**（`thresholding=True`），对每个样本独立：

1. 把 \(x_0\) 展平成 \((B, -1)\)，取 \(|x_0|\) 的 `dynamic_thresholding_ratio` 分位数 \(s\)（线性插值）；
2. \(s \leftarrow \min(\max(s, 1), \text{sample\_max\_value})\)；
3. \(x_0 \leftarrow \operatorname{clip}(x_0, -s, s)/s\)。

数据口径直接对 \(x_0\) 做阈值；噪声口径则先由 \(\epsilon\) 还原 \(x_0\)，阈值化后再换回 \(\epsilon = (x-\alpha x_0^{\text{thr}})/\sigma_t\)。

**5. 方差通道**：`variance_type` 为 `"learned"` / `"learned_range"` 时只对 `epsilon` 生效，`model_output` 是 \((B, 6, \dots)\) 而 `sample` 是 \((B, 3, \dots)\)，只取前 3 个通道当作 \(\epsilon\)。

返回与 `sample` 同形状的 float32 张量。

## 示例

取 `sample=[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]`、`model_output=[[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]]`、`sigma=0.5`（\(\alpha=0.894427\)、\(\sigma_t=0.447214\)）：

```
dpmsolver++ / epsilon        -> [[0.868034, -2.361068, 0.934017], [0.229508, 3.454102, -1.827051]]
dpmsolver++ / v_prediction   -> [[0.67082, -1.900658, 0.782624], [0.178885, 2.772724, -1.475805]]
dpmsolver   / v_prediction   -> [[0.894427, -0.67082, -0.447214], [0.201246, 1.162755, -0.402492]]
dpmsolver   / epsilon        -> 原样返回 model_output
thresholding ratio=0.9 max=1.2
                             -> [[0.723362, -1.0, 0.778347], [0.191257, 1.0, -1.0]]
variance_type="learned"（前 3 通道）-> 与第一行完全相同
```

第一行里 \(x_0 = (1 - 0.447214\times0.5)/0.894427 = 0.868034\)。阈值化那一行：第 0 个样本的 \(|x_0|\) 是 \([0.868, 2.361, 0.934]\)，0.9 分位数 \(= 0.934+0.8\times(2.361-0.934)=2.0756\)，被 `sample_max_value=1.2` 截断成 \(s=1.2\)，于是 \(0.868/1.2=0.723362\)，超出部分钳到 \(\pm1\)。

## 约束

- `model_output` / `sample` 必须是浮点张量、至少 2 维、非空；非 `learned` 时两者形状必须完全相同。
- `sigma` 必须是有限正实数（`bool` 不算）；`use_flow_sigmas=True` 时还要求 \(0<\sigma<1\)。
- `prediction_type` 只接受 `epsilon` / `sample` / `v_prediction` / `flow_prediction`；`algorithm_type` 只接受四种取值之一；`variance_type` 只接受 `fixed_small` / `fixed_large` / `learned` / `learned_range`。
- `thresholding` / `use_flow_sigmas` 必须是 `bool`；`dynamic_thresholding_ratio` 落在 \((0,1]\)；`sample_max_value` 必须 \(\ge 1\)。
- `learned` / `learned_range` 只支持 `epsilon` 预测，且要求 `model_output` 为 \((B,6,\dots)\)、`sample` 为 \((B,3,\dots)\)。
- 内部用 float64 计算，返回 float32。
