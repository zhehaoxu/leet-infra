# Weight-Only INT8 反量化矩阵乘

## 题目描述

LLM 推理最常用的一档量化是 **weight-only**（W8A16 / W4A16）：只把权重存成整型省显存、带宽，激活仍保持浮点算（精度几乎不掉）。反量化是「按输出通道」的一个 scale 向量。本题模拟 `Linear` 层用 int8 权重跑一次矩阵乘。

## 规则

实现 `solve(x, Wq, w_scale)`：

- `x`：浮点激活 $[M, K]$（不量化）。
- `Wq`：int8 权重 $[N, K]$，$N$ 为输出通道（每个 `Wq[n]` 一行）。
- `w_scale`：每输出通道的对称 scale $[N]$（float，等价于该通道 $\text{absmax}/127$）。

真实浮点权重是 $W[n,k] = \text{w\_scale}[n]\cdot Wq[n,k]$，要求计算 $x$ 对它做全连接：

$$\text{out}[m,n] = \sum_{k} x[m,k]\,W[n,k] = \text{w\_scale}[n]\sum_{k} x[m,k]\,Wq[n,k]$$

返回 `out`（float，形状 `[M, N]`，dtype 与 `x` 一致）。

> 关键结构：per-channel scale 与求和下标 $k$ 无关，能**提到矩阵乘外面**——先做 `x @ Wqᵀ`，再按列（输出通道）乘 `w_scale`。这正是硬件上「整型/半精度累加后统一 rescale」的依据，也是避免物化整份反量化权重（省显存）的写法。两种写法浮点结果应在容差内一致。

## 示例

```
x       = [[1.0, 2.0, 0.0],
           [0.5, -1.0, 3.0]]
Wq      = [[10, -100, 50],
           [127,   0, -64]]
w_scale = [0.01, 0.02]

x @ Wqᵀ = [[-190, 127],      out = 按列乘 w_scale
            [ 255, -128.5]]
        = [[-1.9, 2.54],
           [ 2.55, -2.57]]
```

## 约束

- `Wq` 为 int8（可能含 ±127/−128），累加用 float（`x` 是浮点，直接 `x @ Wq.float().T` 即可）
- 输出 dtype、shape 如上；结果应与 `F.linear(x, Wq.float() * w_scale[:, None])` 一致（内置 oracle）
- 不修改输入
