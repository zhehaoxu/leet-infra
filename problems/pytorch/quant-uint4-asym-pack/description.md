# 量化：per-group 非对称 uint4 打包与解包

## 题目描述

AutoGPTQ 等 per-group 非对称 4-bit 量化把权重按组量化为 uint4 整数 \(q\in[0,15]\)，再用共享的 scale \(s\) 与 zero-point \(z\) 还原：

\[W_{ij}=s_g\cdot(q_{ij}-z_g).\]

为了节省显存，每两个 uint4 码被打包进一个 uint8：**低 4 位放偶数索引、高 4 位放奇数索引**：

\[\text{buf}_k=(q_{2k}\ \&\ 0x\text{F})\ |\ ((q_{2k+1}\ \&\ 0x\text{F})\ll 4).\]

本题给定打包后的 uint8 缓冲 `packed`、每组 scale `scales` 与 zero-point `zeros`，要求解包并还原出原始浮点权重 `W`。

## 任务

实现 `solve(packed, scales, zeros, group_size, N)`：

- `packed`：uint8，形状 \((M, \lceil N/2\rceil)\)。
- `scales`：float32，形状 \((M, \lceil N/\text{group_size}\rceil)\)。
- `zeros`：int64，形状与 `scales` 相同，取值 \(0\dots15\)。
- `group_size`：正整数，整除 \(N\)。
- `N`：原始列数。
- 返回 `W`：float32，形状 \((M, N)\)。

## 提示

- 先解包：`(packed & 0xF)` 得到偶数位，`((packed >> 4) & 0xF)` 得到奇数位，再交错拼接成 \((M, N)\) 的 uint4 整数。
- 按列索引 `j // group_size` 取对应的 scale 与 zero-point。
- 若 \(N\) 为奇数，最后一个 uint8 的高 4 位属于填充，应忽略。
