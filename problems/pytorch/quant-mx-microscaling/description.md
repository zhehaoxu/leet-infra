# 量化：MX Microscaling 块缩放与 E8M0 共享指数编码

## 题目描述

**Microscaling（MX）**是 OCP MX v1.0 与 `torchao` 中 `mx_fp4` 等格式采用的块缩放方案：每个 **32 元素块**共享一个**整型编码的 2 的幂指数**，而不是存一个浮点 scale。这种 E8M0 共享指数把 scale 量化成

\[\text{scale}=\begin{cases}0 & \text{code}=0\\ 2^{\text{code}-127} & 1\le \text{code}\le 255\end{cases}\]

其中 127 是 FP8 风格的指数偏置。code 为 8-bit 无符号整数。

给定权重 \(W\in\mathbb R^{M\times N}\)（\(N\) 是 32 的倍数），按行优先把最后维切成每块 32 个元素。对每块：

1. 计算块内最大绝对值
   \[m_b=\max_{j\in\text{block}_b}|W_{ij}|.\]
2. 若 \(m_b=0\)，则 \(\text{code}_b=0\)，\(\text{scale}_b=0\)。
3. 否则取不小于 \(m_b\) 的最小 2 的幂：
   \[e_b=\left\lceil\log_2 m_b\right\rceil,\quad \text{code}_b=\operatorname{clip}(e_b+127, 1, 255),\quad \text{scale}_b=2^{\text{code}_b-127}.\]
4. 块内归一化值
   \[\hat W_{ij}=W_{ij}/\text{scale}_{b(i,j)}.\]

## 任务

实现 `solve(W)`：

- 输入：`W`（float32 \((M, N)\)，\(N\bmod 32=0\)）。
- 返回 `(scale_codes, normalized)`：
  - `scale_codes` 为 uint8，形状 \((M, N/32)\)；
  - `normalized` 为 float32，形状与 `W` 相同。

## 提示

- `torch.log2` 与 `torch.ceil` 可求块内上取整指数；注意 0 块单独处理。
- `scale_codes` 用 `torch.clamp(..., 0, 255).to(torch.uint8)` 得到。
- 对合法的 \(m_b>0\)，验证 \(\hat W_{ij}\cdot\text{scale}_b\) 能精确还原 \(W_{ij}\)（\(10^{-5}\) 内）。
