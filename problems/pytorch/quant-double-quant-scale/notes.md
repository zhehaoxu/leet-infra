# 量化：double quantization 的 scale 二级量化——算法详解

## 1. 为什么 scale 还要再量化

QLoRA 的权重存储是「4-bit 码 + 每 64 权重一个 FP32 absmax」。4-bit 部分是 0.5 byte/权重，而 absmax 部分摊到每个权重是 \(32/64=0.5\) bit——**缩放开销几乎等于码本身的一半**。65B 模型上这部分单独就有 GB 级。

**double quantization**（Dettmers et al., arXiv:2305.14314 §3）把一级 scale 数组再当数据量化一遍：每 256 个 FP32 scale 共享一个 FP32 二级 scale，一级 scale 降为 8 bit。开销变成

\[
\frac{8}{64}+\frac{32}{64\cdot256}=0.125+0.001953\approx0.127\ \text{bit/param}
\]

比 0.5 省 0.373 bit/param。本题实现其中的**数值内核**：给定一级 scale 数组，做二级量化的编码与反量化。

## 2. 对称定点与分组（判题考点）

- **对称**：一级 scale 恒为正，用有符号定点 \([-qmax, qmax]\)、`qmax = 2^(k-1)-1`，零点为 0——不需要 zero-point，反量化就是一次乘法；
- **分组**：`block_size`（QLoRA 取 256，与 `compress_statistics` 的 `quantize_blockwise(..., blocksize=256)` 一致）个 scale **连续**成一组，尾部不满也单独成组；
- **组内 max 精确落在 \(\pm qmax\)**：\(s^{(2)}=\max|s|/qmax\)，因此组内绝对值最大的 scale 反量化无损——`check_error_bound` 对每组断言这一点；
- **误差上界**：\(|\hat s-s|\le s^{(2)}/2\)（round 的固有误差），`check_error_bound` 用 \(s^{(2)}/2+10^{-7}\) 的界验证；
- **全零组**：\(s^{(2)}=1\)、码 0、重建 0，避免除零。

## 3. 与字节账的分工

- `peft-qlora-nf4-forward`：只算**字节数** \(n/2+n/64+\lceil (n/64)/256\rceil\cdot4\)，不碰二级量化的数值；
- 本题：只做**数值往返**（编码、二级 scale、重建），不数字节；两题各自题面自洽。
- 注意口径差异：bitsandbytes 实际存储一级 scale 用 **FP8**（e4m3），本题与题面统一用**对称定点**以便逐元素精确对拍——两者都是「8 bit/scale + 每 256 组一个 FP32」的账，数值路径不同。

## 4. 分组边界为什么重要

`check_group_boundaries` 用「前 256 个 0.01、后 44 个 1.0」的极端数据断言：

- 组数为 2（尾部 44 个单独成组）；
- 组 0 的 \(s^{(2)}=0.01/127\)、组 1 的 \(s^{(2)}=1/127\)。

若把分组写成「按值排序后分桶」或「尾部并入上一组」，两组的 \(s^{(2)}\) 会互相污染（后 44 个 1.0 会把组 0 的 \(s^{(2)}\) 撑大 100 倍），断言立刻失败。

## 5. 实现要点

- 逐组循环即可（判题规模 ≤ 300），无需向量化的 ragged 分组；
- 内部 float64：`s/s2` 在 float32 下的舍入会让「组内 max 精确还原」的断言在 \(10^{-6}\) 容差边缘抖动；
- `bits=8` 时 `qmax=127`；`bits=2` 时 `qmax=1`（example 用它做最小锚点）。

## 6. 验证方法

- `check_example`：4 个 0.25、2 bit、组宽 2 → 码全 1、精确还原。
- `check_group_boundaries`：B=300、组宽 256 的两组独立。
- `check_error_bound`：误差 ≤ s2/2、组内 max 无损。
- `check_matches_oracle`：B=37、4 bit、组宽 8（尾部 5 个）逐组重算。
- `check_zero_group_and_validation`：全零组 + 四类非法入参。
