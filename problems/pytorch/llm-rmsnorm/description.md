# 大模型组件：RMSNorm

## 题目描述

**RMSNorm（Root Mean Square Layer Normalization）** 是 LLaMA / Qwen / Mistral / DeepSeek 等现代大模型的标配归一化层。相比 LayerNorm，它**不减均值、不估偏置**，只用均方根（RMS）做缩放，更省算力、更稳。

$$ \text{out}_i = w_i \cdot \frac{x_i}{\sqrt{\dfrac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} $$

> 本合集把大模型按 `transformers` 的**组件**逐个实现，最后拼成整模型。判题环境装有 `transformers`，
> 提交时你的实现会与 **`transformers` 的 `Qwen2RMSNorm`（同权重）** 逐元素对拍（`allclose`）——
> 所以要复刻的是它的**真实语义**，而不仅是公式。

## 规则

输入是待归一化的张量 $x$（形状 $(\dots, d)$，最后一维为特征维）、逐元素可学习权重 $w$（形状 $(d,)$，即 HF 里的 `weight`）与浮点 $\epsilon$，返回与 $x$ **同形状**的归一化结果。语义要点：

- 方差取**最后一维的均方** $\frac{1}{d}\sum_j x_j^2$，**不减均值**（这是它与 LayerNorm 的根本区别）。
- $\epsilon$ **加在根号内**：$\sqrt{\text{ms} + \epsilon}$。
- HF 在 **float32** 下计算均方与归一化，再转回原 dtype，最后才与 `weight` 相乘——请照此保证数值一致。
- 不修改输入张量。

## 示例

```
x = [1, 2, 3, 4]，weight = [1, 1, 1, 1]，eps = 1e-6
均方 = (1+4+9+16)/4 = 7.5，RMS = sqrt(7.5 + 1e-6) ≈ 2.7386
输出 ≈ [0.365148, 0.730297, 1.095445, 1.460593]
```

## 约束

- $1 \le d \le 4096$；$x$ 至少 1 维、最后一维为 $d$；`weight` 形状 $(d,)$
- dtype 以 float32 为主，保持与输入一致
- 用 `torch.allclose`（相对容差 $10^{-5}$）对拍 `transformers` 参考实现
- 不要直接调用 `transformers` 的模块来「走捷径」——那是判题用的 oracle
