# Diffusion：无分类器引导 (CFG)

## 题目描述

**Classifier-Free Guidance**（Ho & Salimans, arXiv:2207.12598）让同一个网络同时给出「有条件」和「无条件」两种噪声预测 \(\epsilon_\text{cond}\)、\(\epsilon_\text{uncond}\)，再把二者线性外推，放大条件信号（如文本提示）的影响，以「多样性 ↔ 贴合条件」的权衡换取更高保真度。

工程上（diffusers / Stable Diffusion）最常用的组合式，以引导权重 \(w\ge 1\)：

\[ \tilde\epsilon = \epsilon_\text{uncond} + w\,\bigl(\epsilon_\text{cond} - \epsilon_\text{uncond}\bigr) \]

等价于 \((1-w)\,\epsilon_\text{uncond} + w\,\epsilon_\text{cond}\)；与论文的 \((1+w')\epsilon_\text{cond}-w'\epsilon_\text{uncond}\) 相差参数化 \(w'=w-1\)。

## 规则

给定同形状的 `eps_cond`、`eps_uncond` 与标量 `w`（guidance scale）。返回 \(\tilde\epsilon\)（同形状）。float32。禁止现成扩散库。

## 示例

\[ \epsilon_\text{cond}=\begin{bmatrix}1.0&-0.5&0.2\\ 0.3&0.7&-1.1\end{bmatrix},\ \epsilon_\text{uncond}=\begin{bmatrix}0.4&0.1&-0.3\\ 0.6&-0.2&0.5\end{bmatrix},\ w=7.5 \]

```
eps_guided = [[ 4.9,  -4.4,   3.45],
              [-1.65,  6.55, -11.5 ]]
```

## 约束

- `eps_cond` 与 `eps_uncond` 同形状；`w` 为标量 float，通常 \(w\ge1\)。
- **退化性质**：\(w=0\) 时 \(\tilde\epsilon=\epsilon_\text{uncond}\)；\(w=1\) 时 \(\tilde\epsilon=\epsilon_\text{cond}\)（不加引导）；\(w>1\) 沿 cond 方向外推放大条件。
- 逐元素线性组合，结果可能超出输入数值范围（引导放大所致，属预期）。
