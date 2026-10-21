# GELU（精确与 tanh 近似）

## 题目描述

实现 GELU（Gaussian Error Linear Unit）激活函数——Transformer 家族（BERT、GPT）的标准激活。它把输入按标准正态分布的 CDF 加权：

$$
\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\left(1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

工程上还有一个便宜的双曲正切近似（BERT/LLaMA 论文常用）：

$$
\text{GELU}_{\tanh}(x) = \frac{x}{2}\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)
$$

## 规则

- 实现 `solve(x: torch.Tensor, approximate: str = "none") -> torch.Tensor`。
- `approximate == "none"` 用精确 erf 公式；`approximate == "tanh"` 用上面的 tanh 近似。
- `approximate` 为其他值时抛 `ValueError`（与 `torch.nn.functional.gelu` 行为一致）。
- `torch.erf` / `torch.tanh` 可作原语；向量化实现。

## 示例

```
输入：[-1.0, 0.0, 1.0]  (approximate="none")
输出：[-0.15865, 0.0, 0.84134]
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
- 两种模式的结果都应与 `torch.nn.functional.gelu` 对应模式在 `1e-5` 内一致
