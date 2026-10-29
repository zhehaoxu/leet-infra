# 稀疏化：soft-thresholding 与 ISTA 一步

## 题目描述

L1 正则的最小化 \(\min_x \tfrac12\|Wx-y\|_2^2+\lambda\|x\|_1\) 在稀疏编码、剪枝掩码学习里到处出现。它的近端梯度法（ISTA，iterative shrinkage-thresholding）每步只做两件事：

1. 沿数据项梯度走一步；
2. 对结果施加 **soft-thresholding**（L1 的 proximal 算子）。

soft-thresholding 的定义是

$$
\mathcal{S}_\lambda(z)=\operatorname{sign}(z)\cdot\max(|z|-\lambda,\,0)
$$

它把每个分量朝 0 **收缩一个固定量 \(\lambda\)**，幅度不足 \(\lambda\) 的直接压成 0——这正是「L1 产生稀疏」的几何来源：阈值附近整段被抹平。对照最常见 top-k 硬剪枝（`sparsity-magnitude-prune`）：硬剪是保留固定个数、不管数值差多少；soft-thresholding 是保留所有「值得留」的，个数由 \(\lambda\) 和权重分布共同决定，而且数值本身也被改动，因此天然可微（除 0 点外）。

## 规则

给定 `x`（float32，\((n,)\)）、`W`（float32，\((m,n)\)）、`y`（float32，\((m,)\)）、标量 `lam`（\(\lambda\ge0\)）与步长 `eta`（\(\eta\ge0\)）：

1. 残差 \(r=Wx-y\)；
2. 梯度 \(g=W^\top r\)（数据项 \(\tfrac12\|Wx-y\|^2\) 对 \(x\) 的梯度）；
3. 梯度步 \(z=x-\eta g\)；
4. 逐元素 `x_new = sign(z) * max(|z| - lam, 0)`；
5. `support` 为 \(|x\_new|>0\) 的索引，**升序**排列。

返回 `(x_new, support)`：前者 float32 \((n,)\)；后者 int64 一维。禁止调用 `torch.nn.functional.softshrink` 等现成近端算子。

## 示例

```
x = [1, -2, 0.5]
W = [[1, 0, 0],
     [0, 1, 1]]
y = [1, 1]
lam = 0.2, eta = 0.5

r = Wx - y  = [1 - 1, (-2 + 0.5) - 1] = [0, -2.5]
g = Wᵀ r    = [0, -2.5, -2.5]
z = x - 0.5*g = [1, -0.75, 1.75]

x_new = [1-0.2, -(0.75-0.2), 1.75-0.2] = [0.8, -0.55, 1.55]
support = [0, 1, 2]
```

## 约束

- CPU float32；`lam`、`eta` 为非负标量。
- \(|z|\) **恰好等于** `lam` 的分量必须置 0（保留条件严格为大于）；`support` 相应不含该索引。
- `lam=0` 时必须精确退化为梯度步 \(x-\eta g\)；`eta=0` 时只做收缩。
- 收缩保持符号、只削减幅度：保留分量的 \(|x\_new|=|z|-\lambda\)。
