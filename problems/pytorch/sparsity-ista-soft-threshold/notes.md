# 详解：soft-thresholding 与 ISTA 一步

## 出处
soft-thresholding 作为 L1 的近端算子源自 Donoho & Johnstone 的收缩估计（1994/1995），ISTA 作为线性逆问题的迭代阈值法由 Daubechies, Defrise & De Mol（*An iterative thresholding algorithm for linear inverse problems with a sparsity constraint*, 2004）系统化；FISTA（Beck & Teboulle, 2009, arXiv 0904.0426）在其上加速。Wanda 之后，基于阈值/近端的掩码学习也常被用来解释「为什么剪枝后的权重需要微调」。

## 核心与易错点
1. **梯度方向**：数据项 \(\tfrac12\|Wx-y\|_2^2\) 对 \(x\) 的梯度是 \(W^\top(Wx-y)\)，不是 \(W(Wx-y)\)；\(W\) 是 \((m,n)\)，只有左乘 \(W^\top\) 才回到 \((n,)\)。
2. **先走梯度步再收缩**：顺序不可交换。若先对 \(x\) 收缩再走梯度步，得到的点既不是近端步也不稀疏（下一步又会被梯度填满）。
3. **阈值边界**：\(|z|=\lambda\) 时输出 0。实现上 `clamp(|z|-lam, min=0)` 天然给出 0，但若用 `(z - sign(z)*lam)` 直接写会在 \(|z|<\lambda\) 处产生 \(-(\lambda-|z|)\) 的**符号翻转**，这是最经典的实现错误。
4. **`support` 的定义**：是输出非零的索引，不是「被收缩过的索引」，也不含被压成 0 的位置。
5. **数值类型**：`support` 必须是整型索引；`-0.0` 视为 0（`== 0` 成立），判题按值比较。

## 对拍口径
参考解先 `W @ x - y`、`W.T @ r`，再 `sign(z)*clamp(|z|-lam, min=0)`，最后 `nonzero(...).flatten()` 取支撑。判题 `_oracle` 用纯 Python 双重循环按定义逐标量求和，并另存中间量 \(z\) 用于断言。独立断言：`lam=0` 与手算梯度步逐位一致；`lam` 大于所有 \(|z|\) 时全零且 `support` 为空；每个保留分量的幅度恰减少 `lam` 且符号不变；`support` 与「\(|z|>\lambda\)」的集合完全一致；\(|z|=\lambda\) 被置 0；`eta=0` 时纯收缩。

## 复杂度
\(O(mn)\)（一次矩阵–向量乘与一次转置乘）；显存 \(O(n+m)\)。
