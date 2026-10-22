# RMSNorm

## 题目描述

RMSNorm 是 LLM（LLaMA、Mistral 等）的标配归一化：省掉 LayerNorm 的均值中心化与 bias，只按**均方根**缩放：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon},\qquad
y_i = \frac{x_i}{\text{RMS}(x)}\,\gamma_i
$$

论文论证均值中心化并非必要，去掉后每层省一次归约，效果几乎不变。

## 规则

- 实现 `solve(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor`。
- 沿**最后一个维**计算 $\text{mean}(x^2)$（`keepdim=True`），再 `x * rsqrt(ms + eps) * weight`。
- **不做**均值减法、**不加** bias 参数——这是它与 LayerNorm 的本质区别，做错了测试能看出来（全正输入归一化后仍应全正）。
- $\epsilon$ 在开方**内**加，保护全零行不除零。

## 示例

```
输入：x = [[1.0, 2.0, 3.0]], γ = [1,1,1], eps = 1e-6
ms = (1+4+9)/3 ≈ 4.666667，rms ≈ 2.1602469
输出：[[0.46291000, 0.92581999, 1.38873005]]
```

## 约束

- float32/float64，形状 `(..., d)`，`weight` 形状 `(d,)`
- 对输入的整体尺度缩放具有不变性：`solve(c*x, w) ≈ solve(x, w)`（c>0）
