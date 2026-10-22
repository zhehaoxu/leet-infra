# LayerNorm 前向

## 题目描述

Layer Normalization 对**每个样本自己的尾部特征维**做归一化，与 batch 无关，训练/推理行为一致：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta,\qquad
\mu,\sigma^2 = \text{mean/var over the last } \dim(\gamma) \text{ dims}
$$

Transformer 的每个 block 都靠它稳定每层输入分布。

## 规则

- 实现 `solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor`。
- 归一化维是 x 的**尾部** `weight.dim()` 个维（`weight.shape` 即 normalized_shape，必须与 x 的尾维一致）。
- 方差用**有偏**口径（除 $n$，与 `F.layer_norm` 一致）——LayerNorm 没有 BatchNorm 的 running stats，也不做 Bessel 校正。
- $\epsilon$ 加在 **sqrt 内部**（分母是 $\sqrt{\sigma^2+\epsilon}$，不是 $\sqrt{\sigma^2}+\epsilon$）。
- `keepdim` 统计后广播，向量化实现。

## 示例

```
输入：x = [[1.0, 2.0, 3.0, 4.0]], γ = [1,1,1,1], β = [0,0,0,0], eps = 1e-5
μ = 2.5, σ²(有偏) = 1.25
输出：[[-1.34163547, -0.44721183, 0.44721183, 1.34163547]]
```

## 约束

- float32/float64，任意形状（维数 ≥ weight.dim()）
- 常数向量（σ²=0）不得产生 NaN；结果与 `torch.nn.functional.layer_norm` 一致
