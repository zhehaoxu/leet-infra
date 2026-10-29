# Diffusion：APG 的自适应投影引导

## 题目描述

classifier-free guidance 的更新向量是两支预测之差：

\[ \delta=\epsilon_c-\epsilon_u \]

把它相对条件预测 \(\epsilon_c\) 分解成平行与正交两部分，直觉上平行分量主要改变幅度（\(w\) 一大就过饱和），正交分量才真正改变方向。**Adaptive Projected Guidance**（APG）就按这个分解重写引导：正交部分原样保留，平行部分乘一个衰减系数 \(\eta\)，于是既保留条件信息，又不至于把幅度推爆。

本题还带上 APG 在采样器里的两个工程部件：可选的**动量缓冲**（把相邻步的差分做指数滑动平均）与**范数截断**（把更新向量的 L2 范数逐样本压在阈值 \(\tau\) 以内）。

## 规则

给定同形状的 `eps_cond` / `eps_uncond`、`guidance_scale=s`、平行保留系数 `eta`、范数阈值 `norm_threshold=τ`、动量系数 `momentum` 与其上一步缓冲 `prev_momentum`，以及做归约的维度 `dims`，依次执行：

1. \(\delta=\epsilon_c-\epsilon_u\)；若启用动量，则缓冲 \(b\leftarrow\delta+m\cdot b_{\text{prev}}\)（\(b_{\text{prev}}\) 缺省为全零），并令 \(\delta=b\)。
2. 若 \(\tau>0\)，令 \(\delta\leftarrow\delta\cdot\min(1,\tau/\lVert\delta\rVert_2)\)，逐样本一个标量因子。
3. 取 \(\hat{e}=\epsilon_c/\max(\lVert\epsilon_c\rVert_2,10^{-12})\)，分解
   \[ \delta_\parallel=(\delta\cdot\hat{e})\,\hat{e},\qquad \delta_\perp=\delta-\delta_\parallel \]
4. 组合 \(\text{update}=\delta_\perp+\eta\,\delta_\parallel\)，再以**无条件**预测为基准外推：
   \[ \text{out}=\epsilon_u+s\cdot\text{update} \]

返回 `(pred, new_momentum)`：`pred` 为 float32 张量，`new_momentum` 为更新后的缓冲（float32、同形状），未启用动量时为 `None`。缓冲保存的是**截断之前**的 \(\delta\)。

## 示例

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ s=3,\ \tau=0 \]

```
eta=1.0 -> [[ 1.000000, -2.800000,  1.400000], [-0.400000,  1.700000, -1.500000]]
eta=0.5 -> [[ 0.595918, -1.587755,  0.793878], [-0.400000,  0.875000, -0.675000]]
eta=0.0 -> [[ 0.191837, -0.375510,  0.187755], [-0.400000,  0.050000,  0.150000]]
```

`eta=1` 一栏与裸 CFG 逐位相同；`eta=0` 一栏里更新向量与 \(\epsilon_c\) 正交。

## 约束

- 输入至少二维（第 0 维是 batch），两支形状必须一致；`s` 非负有限，\(\eta\in[0,1]\)，\(\tau\ge 0\)，\(m\in[0,1)\)。
- `dims` 默认除第 0 维外的全部维度，不能为空、不能含第 0 维、不能越界或重复。
- `prev_momentum` 只能在启用动量时给出，且形状必须与输入一致；未启用动量时返回的缓冲是 `None`。
- **退化性质**：\(\eta=1\) 时输出恰为 \(\epsilon_u+s(\epsilon_c-\epsilon_u)\)；\(\eta=0\) 时更新向量与 \(\epsilon_c\) 正交；\(\eta\) 在两端之间线性插值。\(\lVert\epsilon_c\rVert=0\) 时方向取零向量，整段差分都算正交分量。
- 差分全零时截断的因子取 1（不能出现 \(0/0\) 或 NaN）。
- 截断只作用于本步更新，不写入动量缓冲。
