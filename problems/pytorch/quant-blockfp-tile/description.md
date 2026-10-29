# 量化：BlockFP 1×128 Tile 与 UE8M0 共享指数

## 题目描述

**BlockFP**（DeepSeek-V3 技术报告 §3.1 采用的 tile 量化格式）把矩阵沿最后维切成 **1×128 的 tile**，每个 tile 共享一个 **UE8M0** 指数。UE8M0 与 E8M0 同构：8-bit 无符号指数、偏置 127，code 为 0 时 scale 为 0，否则

\[\text{scale}=2^{\text{code}-127}.\]

给定 \(W\in\mathbb R^{M\times N}\)（\(N\bmod 128=0\)），按行优先每 128 列一个 tile：

1. 计算 tile 内最大绝对值 \(m_t\)。
2. 若 \(m_t=0\)，则 \(\text{code}_t=0\)，\(\text{scale}_t=0\)。
3. 否则
   \[e_t=\left\lceil\log_2 m_t\right\rceil,\quad \text{code}_t=\operatorname{clip}(e_t+127, 1, 255),\quad \text{scale}_t=2^{\text{code}_t-127}.\]
4. tile 内归一化值 \(\hat W=W/\text{scale}_t\)。

## 任务

实现 `solve(W)`：

- 输入：`W`（float32 \((M, N)\)，\(N\bmod 128=0\)）。
- 返回 `(scale_codes, normalized)`：
  - `scale_codes` 为 uint8，形状 \((M, N/128)\)；
  - `normalized` 为 float32，形状与 `W` 相同。

## 提示

- 与 MX Microscaling 的差异主要在于 tile 大小与格式来源（BlockFP / DeepSeek-V3），编码口径一致。
- 处理 tile 全 0 时避免除 0 产生 NaN。
