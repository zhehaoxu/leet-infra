# 扩散采样：PAG 的双尺度引导与两套窗口

## 题目描述

**Perturbed Attention Guidance**（PAG）在采样时多跑一支「被故意弄坏的模型」：把某些自注意力层换成近似恒等映射，得到一份质量更差、但仍保留结构的条件预测 \(\epsilon_{\text{skip}}\)。它和普通 CFG 的做法一样，用「好预测减坏预测」的差值作为引导方向，只不过这次参考的是 perturbation 分支而不是无条件分支：

\[ \Delta_{\text{cfg}}=\epsilon_c-\epsilon_u,\qquad \Delta_{\text{pag}}=\epsilon_c-\epsilon_{\text{skip}} \]

PAG 的两个尺度各自还有独立的生效区间：CFG 通常全程打开，PAG 只在去噪早期（默认步数占比的 \(1\%\sim 20\%\)）打开——那时结构还没定下来，扰动分支的差值最有信息量。两个区间在 diffusers 里的**端点语义并不一致**，本题的核心就是把这套分支与窗口判定写对。

## 规则

给定同形状的三段预测（第 0 维是 batch）：

- `pred_cond` \(\epsilon_c\)：条件预测；
- `pred_uncond` \(\epsilon_u\)：无条件预测；
- `pred_cond_skip` \(\epsilon_{\text{skip}}\)：扰动注意力分支的条件预测。

以及尺度 `guidance_scale = s`、`perturbed_scale = s_{\text{skip}}`、当前步 `step`、总步数 `num_inference_steps = N`，和两组区间参数 `start / stop`、`perturbed_start / perturbed_stop`。

先判定两个分支是否开启：

\[ \text{cfg\_on} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \ \wedge\ \lnot\text{isclose}(s, 1) \]

\[ \text{pag\_on} = \big[\lfloor \text{perturbed\_start}\cdot N\rfloor < \text{step} < \lfloor \text{perturbed\_stop}\cdot N\rfloor\big] \ \wedge\ \lnot\text{isclose}(s_{\text{skip}}, 0) \]

注意 CFG 区间是**左闭右开**，PAG 区间是**两端都开**。`isclose` 用 `math.isclose` 的默认容差（`1e-9` 相对、`0` 绝对），即 `s` 与 `1.0` 相近时 CFG 视为关闭、`s_skip` 与 `0.0` 相近时 PAG 视为关闭。

然后按开启情况选分支：

| cfg_on | pag_on | 输出 |
| --- | --- | --- |
| 否 | 否 | \(\epsilon_c\) |
| 否 | 是 | \(\epsilon_{\text{skip}} + s_{\text{skip}}\,\Delta_{\text{pag}}\) |
| 是 | 否 | \(\epsilon_u + s\,\Delta_{\text{cfg}}\) |
| 是 | 是 | \(\epsilon_u + s\,\Delta_{\text{cfg}} + s_{\text{skip}}\,\Delta_{\text{pag}}\) |

返回 float32 张量，形状与 `pred_cond` 相同。某个分支关闭时，它对应的输入可以传 `None`；但若分支生效却传了 `None`，抛 `ValueError`。

## 示例

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ \epsilon_{\text{skip}}=\begin{bmatrix}0.2&-0.8&0.5\\0.1&0.3&-0.2\end{bmatrix} \]

取 \(s=7.5\)、\(s_{\text{skip}}=2.8\)、\(N=10\)、区间参数为默认值（CFG 区间 \([0,10)\)，PAG 区间 \((0,2)\)）：

```
step=1 -> [[ 2.910000, -7.520000,  3.480000], [-1.580000,  4.960000, -4.590000]]
step=0 -> [[ 2.350000, -6.400000,  3.200000], [-1.300000,  4.400000, -3.750000]]
step=5 -> [[ 2.350000, -6.400000,  3.200000], [-1.300000,  4.400000, -3.750000]]
```

`step=1` 是两个分支同时开启的结果；`step=0` 落在 PAG 开区间的左端点外，只剩 CFG，因此与 `step=5` 逐位相同。

## 约束

- 三个输入都至少一维（第 0 维为 batch），形状必须一致；传入非张量或形状不符抛 `ValueError`。
- `guidance_scale`、`perturbed_scale` 为有限非负数；`step`、`num_inference_steps` 为整数（`bool` 不算），且 \(N>0\)、\(0\le \text{step} < N\)。
- `start`、`stop`、`perturbed_start`、`perturbed_stop` 都落在 \([0,1]\)，且各自 `start <= stop`，否则抛 `ValueError`。
- 窗口用 `int(...)` 截断取整（不是四舍五入），所以步数很少时 PAG 区间可能整段为空。
- **退化性质**：\(s=1\) 时 CFG 分支关闭；\(s_{\text{skip}}=0\) 时 PAG 分支关闭；两者同时关闭时输出必须与 `pred_cond` 逐位相同。
- 计算内部可用 float64，返回 float32。
