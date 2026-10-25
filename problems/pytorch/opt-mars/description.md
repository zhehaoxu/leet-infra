# 优化器：MARS 方差缩减动量与信任比

## 题目描述

随机梯度估计的方差是大模型训练的隐形税：动量平均抹掉一部分，但对"梯度突然变向"的步仍然反应过度。MARS（*Make vAriance Reduction Shine*，Yuan et al., arXiv:**2411.10438**）把随机优化里的**方差缩减（STB）修正**搬进 Adam 型更新：用相邻两步梯度之差 \((g_t-g_{t-1})\) 构造一个**方差缩减的梯度估计量** \(c_t\)，再对这个估计量做**逐张量的信任比裁剪** \(\rho_t=\min(1,1/\|c_t\|_F)\)，裁剪后的 \(c̃_t\) **同时**驱动一阶矩与二阶矩。以 MARS-AdamW 实例化：

\[
c_t=g_t+\gamma'\,(g_t-g_{t-1}),\qquad \gamma'=\frac{\gamma\beta_1}{1-\beta_1},\qquad
c̃_t=\frac{c_t}{\max(1,\|c_t\|_F)},\qquad
\theta_t\leftarrow\theta_{t-1}-\frac{\text{lr}}{1-\beta_1^t}\cdot\frac{m_t}{\sqrt{v_t/(1-\beta_2^t)}+\varepsilon}
\]

其中 \(m_t=\beta_1m_{t-1}+(1-\beta_1)c̃_t\)、\(v_t=\beta_2v_{t-1}+(1-\beta_2)c̃_t^{\odot2}\)（二阶矩的驱动信号也是裁剪后的 \(c̃\)，不是原始 \(g\)）。实现它：**继承 `torch.optim.Optimizer`**，按固定梯度序列反复 `step()` 就地更新。CPU float32、全程无随机性。

## 规则

- 工厂函数接受 `params` 与 `lr`、`betas`、`gamma`、`eps`、`weight_decay`、`ams_bound`，返回 `torch.optim.Optimizer` 子类实例。完整签名、参数名与默认值只写在 starter 的注释里，那是唯一口径。
- 状态初始化全零：\(m\)、\(v\)、`last_g`（若 `ams_bound=True` 再加 \(v_{\max}\)）。`last_g` 初始为 0，即**首步约定 \(g_0:=0\)**，因此第一步的差分项 \(g_1-g_0=g_1\) 不为零。
- 对第 \(t\) 步、梯度 \(g\)（步计数逐参数独立、从 1 起），按**固定顺序**：
  1. **解耦权重衰减**（仅当 `weight_decay≠0`，先于一切矩估计）：\(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\)；
  2. **方差缩减估计**：\(c=g+\gamma'(g-\text{last\_g})\)，\(\gamma'=\gamma\beta_1/(1-\beta_1)\)（首步 \(\text{last\_g}=0\Rightarrow c=(1+\gamma')g\)）；
  3. **信任比裁剪**（逐张量）：\(\rho=\min(1,\ 1/\|c\|_F)\)，\(c̃=\rho\cdot c\)（等价于 \(c/\max(1,\|c\|_F)\)；\(\|\cdot\|_F\) 是整张量 2-范数）；
  4. \(m\leftarrow\beta_1m+(1-\beta_1)c̃\)；\(v\leftarrow\beta_2v+(1-\beta_2)c̃\odot c̃\)；
  5. **二阶矩自适应裁剪**（仅当 `ams_bound=True`）：\(v_{\max}\leftarrow\max(v_{\max},v)\)，分母基取 \(v_{\max}\)；否则取 \(v\)；
  6. **偏差修正与更新**（与 `opt-adamw` 同口径）：\(\theta\leftarrow\theta-\dfrac{\text{lr}}{1-\beta_1^t}\cdot\dfrac{m}{\sqrt{\text{denom\_base}/(1-\beta_2^t)}+\varepsilon}\)；
  7. `last_g ← g`。
- `grad is None` 的参数整体跳过（不衰减、不推进步计数、不更新 `last_g`）。
- 1-D 与多维参数走**完全相同的路径**（信任比按整张量算，没有形状分支）。
- 全程 float32、CPU、确定性；不许调用现成的 MARS 实现。

## 示例

两个参数 \(W_0=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\)、\(b_0=[1.0,-0.5,0.2]\)，`lr=0.1`、`betas=(0.9,0.99)`、`gamma=0.025`、`eps=1e-8`、`weight_decay=0.01`、`ams_bound=False`，喂入与示例相同的三步梯度 \(g_1,g_2,g_3\)（同 opt-copt 示例）。\(\gamma'=0.025\cdot0.9/0.1=0.225\)。

第 1 步展开：衰减 \(W\cdot0.999\)；\(c=(1+0.225)g_1\)，两个张量的 \(\|c\|_F\approx0.538,0.281\) 都小于 1 ⇒ 信任比不触发；首步偏差修正后更新量 \(\approx0.1\cdot\mathrm{sign}(g_1)\)。三步轨迹（\(W\) 展平）：

```
t=1  W=[0.399500, -0.099800, -0.000100, 0.199700, 0.299600, 0.000100]   b=[0.899000, -0.399500, 0.099800]
t=2  W=[0.400933, -0.139417, -0.097481, 0.113004, 0.273941, -0.029651]   b=[0.884797, -0.385796, 0.010913]
t=3  W=[0.374778, -0.173529, -0.072064, 0.151830, 0.227537, -0.059808]   b=[0.823631, -0.348057, -0.009175]
```

第 2 步起差分项 \(g_t-g_{t-1}\) 进入 \(c\)，动量开始携带"梯度变化率"的信息——轨迹在 \([-0.1,0.1]\) 的截幅带内折返。

## 约束

- \(\gamma\ge0\)，\(0\le\beta_{1,2}<1\)；`params` 可混合 1-D 与多维张量
- 每步后的参数须与独立重算逐元素一致（`atol=rtol=1e-5`）
- 三个判别口径评测直接断言：`gamma=0` ⇒ \(c\equiv g\)，梯度范数全程小于 1 ⇒ 信任比永不触发 ⇒ 与**原生 AdamW** 逐位一致（`atol=rtol=1e-7`）；\(\|c\|_F>1\) 的首步 ⇒ \(\|c̃\|_F=1\) 且 \(m_1\) 的范数恰为 \((1-\beta_1)\)（与梯度尺度无关）；`ams_bound=True` 的分母不小于历史峰值（二阶矩的自适应裁剪）
