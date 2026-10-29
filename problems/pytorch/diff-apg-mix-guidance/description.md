# 扩散采样：APG-Mix 的 CFG 预热与正交投影引导

## 题目描述

**Adaptive Projected Guidance**（APG）把 CFG 的引导方向 \(\Delta=\epsilon_c-\epsilon_u\) 拆成两部分：沿条件预测 \(\epsilon_c\) 方向的分量，和与它正交的分量。它只保留正交分量（或按 `eta` 混入一部分平行分量）再放大，避免高尺度下把样本推向 \(\epsilon_c\) 的模长方向、造成过曝和饱和。

APG-Mix 是 HunyuanImage-2.1 里的组合用法：**去噪前几步先跑普通 CFG，同时用 CFG 的差值预热一个动量缓冲区；从某一步起切到 APG**，此时的引导方向不再是当前步的差值，而是动量平滑过的差值。本题要求实现这个「先 CFG 预热、后投影引导」的单步算子，并把动量状态作为跨步输出显式带出。

## 规则

给定同形状的 `pred_cond` \(\epsilon_c\) 与 `pred_uncond` \(\epsilon_u\)（第 0 维为 batch），当前步 `step`、总步数 `num_inference_steps = N`、跨步动量状态 `buffer`（同形张量，首次调用传 `None`），以及参数 `cfg_scale = s_c`、`apg_scale = s_a`、`momentum = \mu`、`norm_threshold = \tau`、`eta`、`apg_start_step`、窗口 `start / stop`、`use_original_formulation`。

先判两个开关：

\[ \text{cfg\_on} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \ \wedge\ \lnot\text{isclose}(s_c,\, \text{off}) \]

\[ \text{apg\_on} = \text{cfg\_on} \ \wedge\ \text{step} > \text{apg\_start\_step} \ \wedge\ \lnot\text{isclose}(s_a,\, \text{off}) \]

其中 \(\text{off}=1.0\)（默认）或 \(0.0\)（`use_original_formulation=True`），`isclose` 为 `math.isclose` 默认容差。注意 APG 的起点判定是**严格大于**：`step` 恰好等于 `apg_start_step` 时仍走 CFG。APG 依赖 CFG，CFG 关闭时 APG 必然关闭。

差值 \(d=\epsilon_c-\epsilon_u\)。动量更新为

\[ \text{running} = d + \mu\cdot\text{running\_prev},\qquad \text{running\_prev}=0\ \text{(buffer 为 None 时)} \]

`momentum` 为 `None` 时不维护动量（此时 `buffer` 必须为 `None`），一律用当前步的 \(d\)。

三个分支：

| 情况 | 输出 | 动量 |
| --- | --- | --- |
| `cfg_on` 为假 | \(\epsilon_c\) | 不更新，`buffer` 原样带回 |
| 仅 CFG | \(\text{base} + s_c\, d\) | 更新为 `running` |
| APG | \(\text{base} + s_a\,\big(\text{orth} + \eta\,\text{parallel}\big)\) | 更新为 `running`（截断前） |

其中 \(\text{base}=\epsilon_u\)（默认）或 \(\epsilon_c\)（`use_original_formulation=True`）。APG 分支的 `orth` / `parallel` 由**截断后的差值** \(\tilde{d}\) 分解得到：

\[ \tilde{d} = \begin{cases} d_{\text{used}}\cdot\min\!\left(1,\ \tau/\lVert d_{\text{used}}\rVert_2\right), & \tau>0\\ d_{\text{used}}, & \tau=0\end{cases} \]

\[ u = \frac{\epsilon_c}{\lVert \epsilon_c\rVert_2},\qquad \text{parallel}=\langle \tilde{d}, u\rangle\, u,\qquad \text{orth}=\tilde{d}-\text{parallel} \]

这里的 \(d_{\text{used}}\) 就是上表的 `running`（有动量时是平滑值，否则是当前步差值）。所有范数与内积都在**除第 0 维（batch）之外的所有维度**上按样本独立计算，\(\lVert\cdot\rVert_2\) 的分母下限为 \(10^{-12}\)。

注意两个不对称之处：① CFG 分支用**未平滑**的 \(d\) 做平移，动量只是被顺带写入缓冲区；② 缓冲区保存的是**截断前**的 `running`，截断只影响当步输出。

返回 `(pred, new_buffer)`，均为 float32；有动量时 `new_buffer` 与 `pred_cond` 同形，无动量（`momentum=None`）时为 `None`。输入 `buffer` 不得被就地修改。

## 示例

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\quad \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix} \]

取默认参数（\(s_c=3.5\)、\(s_a=10\)、\(\mu=-0.5\)、\(\tau=10\)、\(\eta=0\)、`apg_start_step=5`、窗口 \([0,10)\)）：

```
step=3  pred = [[ 1.150000, -3.200000,  1.600000], [-0.500000,  2.000000, -1.750000]]
step=3  buf  = [[ 0.300000, -0.800000,  0.400000], [-0.200000,  0.600000, -0.500000]]
step=6  pred = [[ 0.253061, -0.359184,  0.179592], [-0.800000,  0.150000,  0.250000]]
step=6  buf  = [[ 0.150000, -0.400000,  0.200000], [-0.100000,  0.300000, -0.250000]]
step=6* pred = [[ 0.406122, -0.318367,  0.159184], [-1.800000,  0.400000,  0.500000]]
```

带 `*` 的一组是 `buffer=None`（从零起）的同一步结果：它的动量等于当前差值，投影方向因此不同。`step=6` 用 `step=3` 返回的 `buf` 作为输入，动量是 \(d_6-0.5\,d_3=0.5\,d_6\)。

## 约束

- `pred_cond` 必须提供，`pred_uncond` 在 `cfg_on` 生效时必须提供；二者至少一维且形状一致，否则抛 `ValueError`。
- `step` 是整数（`bool` 不算）且 \(0\le \text{step} < N\)，\(N>0\)；`apg_start_step` 为非负整数。
- `cfg_scale`、`apg_scale`、`norm_threshold`、`eta` 为有限非负数；`momentum` 为 `None` 或有限实数（可为负）。
- `buffer` 为 `None` 或与 `pred_cond` 同形的张量；`momentum=None` 却传了 `buffer` 抛 `ValueError`。
- `start`、`stop` 落在 \([0,1]\) 且 `start <= stop`。
- **退化性质**：`momentum=None` 时 `new_buffer` 恒为 `None`；\(\eta=1\) 且 \(\tau=0\) 时 APG 退化为以 `base` 为基准的普通 CFG；\(\eta=0\) 时输出增量与 \(\epsilon_c\) 逐样本正交。
- 计算内部可用 float64，返回 float32。
