# 扩散采样：LTX-2 三路增量引导合成

## 题目描述

LTX-2 是音视频联合生成的 DiT，采样时除了常规 **CFG**，还叠两路「残疾版模型」的引导：

- **STG**（spatio-temporal guidance）：一次前向里扰动若干个 transformer block，得到结构更差但仍对齐文本的条件预测，用「正常预测减扰动预测」把结果推离劣质分布；
- **模态隔离引导**：一次前向里关掉 A2V / V2A 跨模态注意力，得到只剩本模态信息的条件预测，用同样的差值抑制跨模态串扰。

三路引导在 LTX-2 里写成**同一个增量形式**：全部以条件预测 \(\epsilon_c\) 为基准、逐项相加，而不是像 CFG 那样在 \(\epsilon_u\) 上做平移。本题要求实现单模态的单步合成，窗口判定与各分量的开关也一并写对。

## 规则

给定同形状的四段预测（第 0 维为 batch）：`pred_cond` \(\epsilon_c\)、`pred_uncond` \(\epsilon_u\)、`pred_stg`、`pred_modality`，以及尺度 `cfg_scale = s_c`、`stg_scale = s_t`、`modality_scale = s_m`、当前步 `step`、总步数 `num_inference_steps = N`、窗口 `start / stop`。

三路共用一个**左闭右开**窗口：

\[ \text{within} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \]

各分量的开关：

\[ \text{cfg\_on} = \text{within} \wedge \lnot\text{isclose}(s_c, 1.0) \]

\[ \text{stg\_on} = \text{within} \wedge \lnot\text{isclose}(s_t, 0.0) \]

\[ \text{mod\_on} = \text{within} \wedge \lnot\text{isclose}(s_m, 1.0) \]

合成（每项只在开时出现）：

\[ \epsilon = \epsilon_c + (s_c-1)\,(\epsilon_c-\epsilon_u) + s_t\,(\epsilon_c-\epsilon_{stg}) + (s_m-1)\,(\epsilon_c-\epsilon_{mod}) \]

注意尺度语义**不统一**：CFG 与模态隔离是「以 1 为恒等」的尺度，所以增量系数要减 1；STG 是「以 0 为恒等」，增量系数就是 \(s_t\) 本身。写成 \(s_c(\epsilon_c-\epsilon_u)\) 会让 \(s_c=1\) 时凭空多出一次引导，写成 \((s_t-1)\) 会让 \(s_t=0\) 时反而减掉一份差值。

返回 float32 张量，形状与 `pred_cond` 相同。某分量关闭时其输入可传 `None`；若分量生效却传了 `None`，抛 `ValueError`。

## 示例

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ \epsilon_{stg}=\begin{bmatrix}0.2&-0.8&0.5\\0.1&0.3&-0.2\end{bmatrix},\ \epsilon_{mod}=\begin{bmatrix}0.3&-1.0&0.55\\0.05&0.2&-0.3\end{bmatrix} \]

`step=1`、`N=10`（窗口 \([0,10)\) 全开）：

```
默认尺度 (1.0, 0.0, 1.0) -> [[ 0.400000, -1.200000,  0.600000], [ 0.000000,  0.500000, -0.500000]]
(3.0, 1.0, 1.5)        -> [[ 1.250000, -3.300000,  1.525000], [-0.525000,  2.050000, -1.900000]]
仅 CFG (3.0)           -> [[ 1.000000, -2.800000,  1.400000], [-0.400000,  1.700000, -1.500000]]
```

第一行是默认尺度：三项都落在恒等点，输出与 \(\epsilon_c\) 逐位相同。

## 约束

- 四段输入都至少一维（第 0 维为 batch），形状必须一致；非张量或形状不符抛 `ValueError`。
- `cfg_scale`、`stg_scale`、`modality_scale` 为有限实数（`stg_scale` 可为负，对应反向引导）；传入非实数或非有限值抛 `ValueError`。
- `step` 是整数（`bool` 不算）且 \(0\le \text{step} < N\)，\(N>0\)。
- `start`、`stop` 落在 \([0,1]\) 且 `start <= stop`。
- **退化性质**：三个尺度都在恒等点时输出必须与 `pred_cond` 逐位相同；窗口为空时同样只返回 `pred_cond`，且此时不需要任何分支预测。
- **可加性**：三项增量都以 `pred_cond` 为基准，因此「全开结果减 `pred_cond`」等于三项增量之和。
- 计算内部可用 float64，返回 float32。
