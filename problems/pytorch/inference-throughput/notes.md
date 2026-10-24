# 解题思路

## 面试官意图

「为什么 TTFT 随 prompt 变长而变长，但每多生成一个 token 的成本基本恒定？」「为什么 decode 阶段
加 batch 几乎免费而算力加不动？」——这些推理服务的核心现象，都能从 prefill/decode 两阶段的
FLOPs 结构讲清楚。本题用 $2N$ 口径把它变成一道可算的账。

## $2N$ 每 token

一次前向，每个参数参与一次「乘 + 加」，所以每 token 约 $2N$ 次浮点运算。这是 `transformer-mfu` 里
$6ND$（前向 $2ND$ + 反向 $4ND$）的训练口径在**纯推理前向**下的对应物：这里只取 $2N$。

## 两阶段的本质差异

- **prefill**：`prompt_len` 个 token **一次性并行**算，矩阵很大 → 算术强度高 → **compute-bound**。
  FLOPs $=\text{batch}\cdot\text{prompt\_len}\cdot 2N$，决定 TTFT。
- **decode**：每步只有 **1 个新 token**（batch 维并行但序列维=1），矩阵是瘦长条 → 每读一遍权重
  只做 $2N\cdot\text{batch}$ 的活 → 常常 **memory-bound**（要重读权重 + KV cache）。

本题把两阶段都用同一个 `effective_flops` 折算时间，是一个简化教学口径；真实系统里 decode 往往受
带宽而非算力限制（见 `roofline-arithmetic` 与 `kv-cache-memory`）。

## 从账里能读出的优化方向

- 想降 TTFT：prefill 是算力活 → 提高并行度 / 用 chunked prefill 与 decode 交错。
- 想提高吞吐：**continuous batching** 把多个请求的 decode 拼成大 batch，摊薄每 token 的权重读取，
  把 memory-bound 的 decode 往 compute-bound 推。
- `gen_len` 只影响 decode 项，不影响 TTFT——这正是 `check_zero_generation` 固化的结构。

## 常见错误

- 把 prefill 也算成随 `gen_len` 增长（prefill 只吃 prompt 一次）。
- $2N$ 写成 $N$ 或 $4N$（那是训练前向+反向的口径）。
- 时间用整数除法截断（应先转 float）。

## 判题口径

`test_cases.py` 用逐阶段独立重算核对 FLOPs 与时间，固化：`gen_len=0 → 无 decode`、batch 对
FLOPs/时间的线性、`effective_flops` 翻倍只减半时间而不改 FLOPs、`total ≥ ttft` 的不变量。
