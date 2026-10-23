# RoPE 旋转位置编码（Rotary Position Embedding）

## 题目描述

RoPE 通过按位置把向量分成的**相邻二维子空间各旋转一个角度**来注入位置信息，使得两点注意力分数只依赖它们的**相对位置**。

对最后一维 $d$，第 $i$ 对分量 $(x_{2i}, x_{2i+1})$ 在位置 $p$ 下按角度 $p\cdot\theta_i$ 做 2D 旋转，其中旋转频率

$$ \theta_i = \text{base}^{-2i/d} $$

对应的旋转为

$$ \begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} $$

## 规则

- 实现 `solve(x, positions, base=10000.0)`，`x` 形状 `(..., L, d)`，`positions` 形状 `(L,)`，返回同形状。
- **相邻配对**（rotate-half 的 interleaved 约定）：$(0,1),(2,3),\dots$，即偶数下标与其后继配对，各用角度 $p\cdot\text{base}^{-2i/d}$（$i$ 为对编号，从 0 起）。
- 最后一维 $d$ 为偶数；第 $i$ 对的频率指数是 $-2i/d$（等价 `base ** (-(0::2)/d*2)` 归一）。
- 位置 $p=0$ 时恒等（$\cos 0=1,\sin 0=0$）。

## 示例

```
输入：x=[[[1,0]]], positions=[1], base=10000, d=2
   i=0 → θ=10000^0=1 → 角度=1*1=1rad → 旋转 (1,0) → (cos1, sin1)
输出：[[[0.5403023, 0.8414710]]]

输入：x=[[[1,0,0,1]]], positions=[2], d=4
   pair0 θ=10000^0=1, 角度=2 → (1,0)→(cos2, sin2)=(-0.4161, 0.9093)
   pair1 θ=10000^(-0.5)=0.01, 角度=0.02 → (0,1)→(-sin0.02, cos0.02)=(-0.0200, 0.9998)
输出：[[[-0.4161468, 0.9092974, -0.0199987, 0.9998000]]]
```

## 约束

- float32/float64 输入；$d$ 为偶数
- 与复数实现 `view_as_complex(x) * exp(1j * p * θ)` 一致
