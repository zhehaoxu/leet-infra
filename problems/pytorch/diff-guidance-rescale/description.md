# Diffusion：CFG 的 rescale 与 guidance interval

## 题目描述

classifier-free guidance 用无条件与条件两支预测做外推：

\[ \hat\epsilon = \epsilon_u + w\,(\epsilon_c-\epsilon_u) \]

\(w\) 一大，外推结果就跑到训练分布之外，画面出现过饱和/过曝（高对比、色块）。**guidance rescale**（Lin et al. 的 std 匹配）把它按每个样本的整体标准差拉回来：

\[ \hat\epsilon \leftarrow \hat\epsilon\cdot\frac{\operatorname{std}(\epsilon_c)}{\operatorname{std}(\hat\epsilon)} \]

注意这只是「缩放到同一幅度」，**不保均值**。另一侧的问题是引导并非每步都有益：**guidance interval**（Kynkäänniemi et al.）只在 \(t\in[t_{\text{low}},t_{\text{high}}]\) 内施加引导，区间外直接返回条件预测。本题把两者合成一个算子，并留一个混合强度 \(\varphi\)：\(\varphi=0\) 为裸外推，\(\varphi=1\) 为完全 rescale。

## 规则

给定同形状的 `eps_cond` / `eps_uncond`、`guidance_scale=w`、当前步 `t`、区间 `t_low` / `t_high`（闭区间）、强度 `rescale=φ∈[0,1]` 与求标准差的维度 `dims`（默认除第 0 维外全部，口径为总体标准差），返回 float32 结果。`t` 不在区间内时原样返回 `eps_cond`。

## 示例

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ w=3,\ t=0.5 \]

```
rescale=1 -> [[ 0.425607, -1.191700,  0.595850], [-0.123014,  0.522809, -0.461302]]
rescale=0 -> [[ 1.000000, -2.800000,  1.400000], [-0.400000,  1.700000, -1.500000]]
```

## 约束

- `w` 必须非负有限；\(\varphi\in[0,1]\)；`t`、`t_low`、`t_high` 必须落在 \([0,1]\) 且 `t_low<=t_high`。
- `dims` 不能为空、不能含第 0 维（要逐样本算标准差）、不能越界或重复；输入至少二维。
- **退化性质**：\(w=1\) 时 \(\hat\epsilon=\epsilon_c\)（rescale 比值恰为 1，输出逐位等于 `eps_cond`）；区间外输出逐位等于 `eps_cond`。
- \(\varphi=1\) 时每个样本的总体标准差等于 `eps_cond` 的；\(\varphi=0.5\) 时是裸外推与完全 rescale 的线性插值。
- 引导后标准差为 0（例如两支都是常数）无法 rescale，抛 `ValueError`。
