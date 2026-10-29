# Squared-ReLU 注意力：relu² 替代 softmax 的行归一前向

## 题目描述

把 attention 里的 softmax 换成一个逐元素激活，是省掉整行归约、让注意力可以直接切块的常见做法。Shen et al.（arXiv 2309.08586）系统比较了点式注意力 \( \phi=L^{-\alpha}h \)、\(h\in\{\mathrm{relu},\mathrm{relu}^2,\dots\}\) 这一族，其中 **squared ReLU** 在视觉 Transformer 上最接近 softmax 的效果。

本题要求实现完整的 relu² 注意力前向：激活后的分数按行归一（分母带 `eps` 下界），可选地在点积之前对 q、k 在 head 维做 RMSNorm（QK-Norm）。它和 softmax 注意力的差别不只是"换个激活"——下面的规则会逐条说明。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- \( \text{logits}=q k^{\mathsf T}/\sqrt{h_d} \)，然后 \( a=\mathrm{relu}(\text{logits})^2 \)。**先 relu 再平方**；反过来会负负得正，把负 logit 也算成贡献。
- `mask` 中 True = 保留、False = 把激活后的分数置 0；被屏蔽位置权重恒为 0。
- \( \text{denom}=\sum_j a_j \)，\( \text{weights}=a/\max(\text{denom},\texttt{eps}) \)，\( \text{output}=\text{weights}\cdot v \)。
- `qk_norm=True` 时先把 q、k 各自按 head 维做**无权** RMSNorm（\(x/\sqrt{\mathrm{mean}(x^2)+\texttt{eps}}\)），再做点积；`eps` 同时用作分母下界。
- 返回 `(output, weights)`：`output` 形状 `(..., L, v_d)`、`weights` 形状 `(..., L, S)`，均为 float32；内部用 float64 计算。
- 分母为 0（整行没有正 logit，或整行被 mask）时该行权重全 0、输出为 0，不得出现 NaN；\(0<\text{denom}<\texttt{eps}\) 时该行被等比缩小，行和小于 1。
- 前置维按广播规则对齐，例如 q `(2, 3, L, h_d)` 配 k、v `(2, 1, S, ·)`。
- **注意**：relu² 是二次齐次，加上行归一后 \(\sqrt{h_d}\) 这个温度会被精确抵消，写不写结果都一样；同理，把 q 或 k 整体乘一个正常数不改变权重。这不是 bug，而是该注意力族的性质。

## 示例

```
query = [[1, 0], [1, 1]]
key   = [[1, 0], [0, 1], [-1, 0]]
value = [[1, 0], [0, 1], [5, 5]]

weights = [[1.0, 0.0, 0.0],
           [0.5, 0.5, 0.0]]
output  = [[1.0, 0.0],
           [0.5, 0.5]]
```

第一行 logits 为 \([\frac{1}{\sqrt2},0,-\frac{1}{\sqrt2}]\)，relu² 后只有第一项非零，于是权重 \([1,0,0]\)，输出直接等于第一个 value。第二行 logits 为 \([\frac{1}{\sqrt2},\frac{1}{\sqrt2},-\frac{1}{\sqrt2}]\)，前两项各得 0.5 的权重——注意第二项的 logit 明明是 0，在 softmax 下会分到 \(1/3\)，relu² 却给它严格的 0。

## 约束

- `query`、`key`、`value` 必须是 float32/float64 张量，`ndim >= 2`，head 维非空，且全部有限；出现 NaN 或 ±inf 抛 ValueError。
- q 与 k 的 head 维必须相同；k 与 v 的前置维必须完全一致；三者前置维不可广播时抛 ValueError。
- `mask` 必须是 bool 张量且能广播到分数矩阵；`qk_norm` 必须是 bool；`eps` 必须是有限正实数（bool、0、负数、NaN 都抛 ValueError）。
- 时间 \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S\cdot h_d)\)，权重矩阵占用 \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S)\)。
