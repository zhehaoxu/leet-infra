# 全连接线性层详解

## 权重方向：`(out, in)` 的历史包袱

`nn.Linear` 把权重存成 `(out_features, in_features)`，于是前向是 $y = xW^\top + b$
（对 `x` 的最后一维做点积）。很多人第一次会写成 `x @ weight` 而形状对不上——正确是
`x @ weight.T`。这一约定源自早期 MATLAB/caffe 的按列组织，改不动是因为海量
`state_dict` 都按此顺序保存。

## 为什么它对 batch 维透明

线性变换只作用于**最后一维**，前面所有维度都被当作独立的「样本」并行处理：
`(N, L, in) → (N, L, out)`。实现上等价于把前置维 flatten 成行做一次矩阵乘再 reshape 回去，
PyTorch 的广播让这步无需手动 reshape。

## 与其它算子的关系

- **卷积**可看作「局部窗口上的线性层」：`Conv2d` 对每个感受野套用一组 `(out, in·k·k)`
  的线性权重（见 im2col 一题把它显式变成矩阵乘）。
- **注意力的 QKV 投影、FFN、LM head** 全是线性层，只是 `in/out` 尺寸不同。
- **bias 融合**：推理引擎常把 bias 合进 matmul 的 epilogue，省一次 elementwise。

## 梯度视角

对 `x`：$\partial y/\partial x = W^\top$（把梯度映回输入空间）；对 `weight`：
$\partial y/\partial W = \delta^\top x$（外积）。理解前向的点积结构，反向就是它俩的转置重排。
