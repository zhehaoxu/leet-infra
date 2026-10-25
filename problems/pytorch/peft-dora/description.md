# PEFT：DoRA 幅值-方向分解低秩适配

## 题目描述

**DoRA**（Weight-Decomposed Low-Rank Adaptation, Liu et al., arXiv:2402.09353）把权重向量按「幅值 × 方向」拆开，再只对**方向**部分做 LoRA 式低秩更新。相比纯 LoRA，它让低秩增量更聚焦于改变权重方向，从而在同等参数量下往往更接近全量微调。

给定冻结基座 \(W_0\)（\((d_\text{out}, d_\text{in})\)）与低秩因子 \(A:(r, d_\text{in})\)、\(B:(d_\text{out}, r)\)，缩放 \(s=\alpha/r\)，以及可训练的幅值向量 \(m:(d_\text{out},)\)：

1. 合成未归一化权重 \(V = W_0 + s\,B A\)（\((d_\text{out}, d_\text{in})\)）。
2. 逐**输出行**取 L2 范数 \(n = \lVert V\rVert_2 \;(\text{dim}=1,\ \text{keepdim})\)，即方向 \(\hat V = V / n\)。
3. 幅值-方向重组 \(W_\text{eff} = m \odot \hat V\)（\(m\) 沿输入维广播到每一行）。
4. 前向 \(y = x\,W_\text{eff}^\top\)。

## 规则

输入 `x` 形状 \((\dots, d_\text{in})\)。返回 `y`（形状 \((\dots, d_\text{out})\)）。全程 float32，禁止调用现成 PEFT 库。

## 示例

采用与 LoRA 相同的固定输入 \(W_0(3\times5), A(2\times5), B(3\times2), x(2\times5)\)，\(\alpha=8.0, r=2\)（\(s=4.0\)），幅值 \(m=[0.7, 0.8, 0.6]\)：

```
y = [[ 0.3858,  0.9513, -0.3756],
     [ 0.3263, -0.1246, -0.0308]]
```

## 约束

- CPU float32；`x` 支持 2D 与 3D（最后维为 \(d_\text{in}\)）。
- 行范数 \(n\) 沿 dim=1（每个输出行一个标量），\(m\) 长度等于 \(d_\text{out}\)。
- **退化性质**：当 \(m = \lVert V\rVert_\text{row}\) 时 \(W_\text{eff}=V\)，DoRA 前向等于合并 LoRA 后的前向；当 \(\alpha=0\) 且 \(m=\lVert W_0\rVert_\text{row}\) 时回到基座前向 \(x W_0^\top\)。
