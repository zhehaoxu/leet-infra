# 量化：double quantization 的 scale 二级量化

## 题目描述

块量化里每个 block 都要存一个 FP32 缩放（QLoRA 的 block=64，即 0.5 bit/param 的额外开销；65B 模型上这本身就是 GB 级）。**double quantization**（Dettmers et al., NeurIPS 2023, arXiv:2305.14314 §3）把**缩放本身再量化一次**：一组 `block_size` 个一级 scale 共享一个 FP32 二级 scale，一级 scale 存成 8-bit 定点。

本题实现这个「scale 的 scale」的数值内核（存储字节账在 `peft-qlora-nf4-forward`）。设一级 scale 数组为 \(s\in\mathbb R^{B}\)，按 `block_size`（默认 256）连续分组（尾部不足一组也单独成组）：

\[
s^{(2)}_g=\frac{\max_{b\in g}|s_b|}{2^{k-1}-1}\quad(\text{若 }\max=0,\ s^{(2)}_g=1),\qquad
c_b=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{s_b}{s^{(2)}_g}\right),-(2^{k-1}-1),\,2^{k-1}-1\right),\qquad
\hat s_b=c_b\,s^{(2)}_g
\]

即**对称定点**量化：`k` 位有符号、`qmax = 2^(k-1)-1`，组内绝对值最大的 scale 精确落在 \(\pm qmax\) 上。

## 任务

实现 `solve(scales, bits, block_size)`：

- 输入：`scales` float32 `(B,)`（全部为正的一级缩放，但不强制校验）、`bits` 为 2～8 的整数、`block_size` 为正整数。
- 按 `block_size` 连续分组（最后一组可以不满）；每组独立求 `s2`、量化并反量化。
- 输出 `(codes, group_scale, deq)`：`codes` int64 `(B,)`；`group_scale` float32 `(n_groups,)`；`deq` float32 `(B,)`。

**约定**：内部 float64，输出 float32；容差 \(10^{-5}\)。`bits` 越界或 `block_size <= 0` 抛 `ValueError`。

## 示例

```
scales = [0.25, 0.25, 0.25, 0.25], bits = 2, block_size = 2
qmax = 1；组 0 s2 = 0.25/1 = 0.25 → codes [1,1]、deq [0.25,0.25]
组 1 同理
输出：codes=[1,1,1,1], group_scale=[0.25,0.25], deq=scales（精确还原）
```

## 约束

- 分组边界是「连续 block_size 个」而不是按值分桶；最后一组不满时**单独成组**，不跨组借 scale。
- 每组绝对值最大的元素必须精确还原（误差 ≤ 1e-6）。
- 全零组的 `s2 = 1.0`、码全 0、重建全 0（避免除零）。
