# 量化：BitNet b1.58 三值量化与加减计数 GEMM

## 题目描述

**BitNet b1.58**（Ma et al., arXiv:2402.17764）把权重约束到 \(\{-1,0,1\}\)，并用**绝对均值（absmean）**作为块缩放因子，把矩阵乘法退化为查表式的加减计数，从而在大模型推理时大幅降低乘法开销。

给定权重 \(W\in\mathbb R^{d_\text{out}\times d_\text{in}}\)：

1. 计算缩放因子
   \[s=\frac{1}{d_\text{out}\cdot d_\text{in}}\sum_{i,j}|W_{ij}|.\]
2. 三值化
   \[T_{ij}=\operatorname{clamp}\left(\operatorname{round}\left(\frac{W_{ij}}{s}\right), -1, 1\right),\quad T_{ij}\in\{-1,0,1\}.\]
3. 反量化权重
   \[\hat W_{ij}=s\cdot T_{ij}.\]
4. 对输入 \(X\in\mathbb R^{b\times d_\text{in}}\) 做三值 GEMM：
   \[Y=X\,\hat W^{\mathsf T}=s\left(\sum_{j:T_{ij}=1}X_{:,j}-\sum_{j:T_{ij}=-1}X_{:,j}\right)\in\mathbb R^{b\times d_\text{out}}.\]

## 任务

实现 `solve(W, X)`：

- 输入：`W` (float32 \((d_\text{out}, d_\text{in})\))、`X` (float32 \((b, d_\text{in})\))。
- 返回 `(s, T, W_hat, Y)`：
  - `s` 为 float32 标量；
  - `T` 为 int64 形状 \((d_\text{out}, d_\text{in})\)；
  - `W_hat` 为 float32 形状与 `W` 相同；
  - `Y` 为 float32 形状 \((b, d_\text{out})\)。

## 提示

- `round` 后 clamp 到 \(\{-1,0,1\}\) 即可得到三值张量；注意 0 在四舍五入后保持 0。
- GEMM 可利用正、负掩码分别对 `X` 求和再相减，避免逐元素构造大中间张量。
- 容差 \(10^{-5}\)。
