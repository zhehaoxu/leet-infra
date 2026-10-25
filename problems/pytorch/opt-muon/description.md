# 优化器：Muon Newton-Schulz 正交化动量

## 题目描述

Muon（MomentUm Orthogonalized by Newton-schulz，Keller Jordan, 2024；Modded-NanoGPT 纪录用）专用于神经网络里的**二维权重矩阵**。它先跑普通 SGD-动量，再把动量矩阵**正交化**成"最接近的半正交矩阵"作为更新方向——即用 \(USV^\top\) 里的 \(UV^\top\) 替换 \(U S V^\top\)。为高效求正交化，用 5 步 **Newton-Schulz 迭代**近似零次幂，避免显式 SVD。

本题要求你**继承 `torch.optim.Optimizer`** 实现 Muon，`step()` 就地更新参数；评测喂固定梯度序列、比对最终参数与独立参照实现是否逐元素一致。为在 CPU 上稳定判题，Newton-Schulz 全程 **float32**（原实现用 bfloat16）。

## 规则

实现工厂函数 `solve(params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0)`，返回 `torch.optim.Optimizer` 子类实例；`params` 为二维权重矩阵列表，`step()` 就地更新它们。对每个有梯度 \(g\) 的参数 \(p\)（第 \(t\) 步、\(t\) 从 1 计）：

1. 动量缓冲：\(buf \leftarrow \mu\, buf + (1-\mu)\, g\)
2. Nesterov 组合：\(G \leftarrow (1-\mu)\, g + \mu\, buf\)
3. **Newton-Schulz 正交化** \(\operatorname{NS}(G)\)（见下），得半正交方向 \(O\)
4. 谱缩放：\(scale = \max\!\big(1,\ \tfrac{\text{rows}}{\text{cols}}\big)^{0.5}\)，\(O \leftarrow O\cdot scale\)
5. 解耦权重衰减（`weight_decay`≠0）：\(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
6. 参数更新：\(p \leftarrow p - lr\cdot O\)

**Newton-Schulz 迭代 \(\operatorname{NS}(G)\)**，quintic 系数 \((a,b,c)=(3.4445,\,-4.7750,\,2.0315)\)：
- 若 \(\text{rows}>\text{cols}\)：先转置（记 \(X\leftarrow G^\top\)），否则 \(X\leftarrow G\)；
- 归一化：\(X \leftarrow X/(\lVert X\rVert_F + 10^{-7})\)（\(\lVert\cdot\rVert_F\) 为整体 Frobenius 范数，保证谱范数 \(\le 1\)）；
- 迭代 \(ns\_steps\) 次：\(A = X X^\top,\; B = bA + cA^2,\; X = aX + BA\)；
- 若之前转置过，则再转置回来。

初始 \(buf=0\)；`grad is None` 跳过；**禁止**调用现成 Muon/做显式 SVD，需按上式手写迭代。

## 示例

单个权重 \(p=\begin{bmatrix}0.5 & -0.2 & 0.1\\ 0.3 & 0.4 & -0.1\end{bmatrix}\)，`lr=0.1`、`momentum=0.95`、`ns_steps=5`、`weight_decay=0`，三步梯度：

```
g₁ = [[ 0.2,-0.1, 0.05],[ 0.1, 0.3,-0.2]]
g₂ = [[-0.15, 0.2, 0.1],[ 0.05,-0.1, 0.3]]
g₃ = [[ 0.1, 0.05,-0.2],[-0.3, 0.15, 0.1]]
三步后 p ≈ [[0.432205, -0.322988, 0.124788], [0.256377, 0.285218, -0.137443]]
```

## 约束

- 输入 float32、二维矩阵（如 \(2\times3\)、\(5\times2\) 等非方阵要能正确处理转置分支）
- Newton-Schulz 用 float32，不做 bfloat16；Frobenius 归一化加 \(10^{-7}\) 防零
- 与独立参照逐步完全一致（口径见 notes.md）
