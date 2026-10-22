# WeightNorm 重参数化详解

## 核心思想

把权张量 $w$ 拆成**方向**与**长度**两部分独立参数化：

$$ w = g\cdot\frac{v}{\|v\|_{\text{dim}}} $$

$g$（weight_g）是标量/每输出通道的尺度，$v$（weight_v）决定方向。原始参数 $w$ 不再直接优化，而优化 $(g,v)$。

## 为什么

- 解耦权重「大小」与「方向」，条件数更好，SGD 收敛更快、允许更大学习率——OpenAI GPT-1、以及现在 diffusion 模型里常包的 `torch.nn.utils.weight_norm`。
- 训练时每步由 $(g,v)$ 现算有效权重 $w$，前向与普通线性/卷积一致。

## 实现与坑

- 范数是 **L2**，沿指定 `dim`（卷积常用 `dim=1` 或按输出通道，本关默认 `dim=0`），`keepdim=True` 广播回原形状。
- **除零保护**：$v$ 某切片全 0 时 $\|v\|=0$，需 `clamp_min(1e-12)`，否则 `0/0=NaN`。这是本关的隐藏考点。
- $g$、$v$、输出同形状；$g$ 可能是标量或沿 dim 的向量，靠广播对齐。

## 与 SpectralNorm / WeightNorm 去耦

`torch.nn.utils.weight_norm` 是这个重参数化的官方封装（注意其 `dim` 语义与旧版差异）；本关只需按公式手算有效权重，不涉及反向。
