# 算子融合的显存带宽收益（Roofline 视角）

## 题目描述

逐元素算子链（如 `Linear → Bias → ReLU → Dropout`）若各自启动一个 kernel，每个 kernel 都要把
中间张量**从 HBM 读进来、再写回 HBM**，海量时间耗在访存而非计算上。**算子融合（fusion）** 把
相邻算子并进一个 kernel，中间结果只留在片上（寄存器/SRAM）直接流转，从而砍掉反复的 HBM 往返。
本题用 Roofline 的**访存**视角量化这一收益。

## 规则

- 实现 `solve(n_elems, dtype_bytes, num_ops)`，返回 `(non_fused_bytes, fused_bytes, saved_bytes, speedup)`。
- 建模：一条由 `num_ops` 个逐元素算子串成的链，每个算子恰好**读 1 个、写 1 个** $N$ 元素、
  每元素 `dtype_bytes` 字节的张量。
- **非融合**（每算子各一个 kernel）：第 $i$ 个算子从 HBM 读其输入、写其输出。$m$ 个算子共
  $m$ 次读 + $m$ 次写，HBM 流量：
  $$ \text{non\_fused\_bytes} = 2\,m\,N\,\text{dtype\_bytes} $$
- **融合**（整条并成一个 kernel）：只需读最初输入一次、写最终输出一次：
  $$ \text{fused\_bytes} = 2\,N\,\text{dtype\_bytes} $$
- 节省与加速比（假设 kernel 时间**受 HBM 带宽 bound**，时间与流量成正比）：
  $$ \text{saved\_bytes} = \text{non\_fused} - \text{fused} = 2(m-1)N\,\text{dtype\_bytes} $$
  $$ \text{speedup} = \frac{\text{non\_fused}}{\text{fused}} = m $$
- 前三个返回整数（字节），`speedup` 返回浮点。

## 示例

```
输入：n_elems = 1<<20 = 1,048,576,  dtype_bytes = 2 (fp16),  num_ops = 5
  non_fused_bytes = 2*5*1048576*2 = 20,971,520
  fused_bytes     = 2*1048576*2   =  4,194,304
  saved_bytes     = 16,777,216
  speedup         = 5.0
输出：(20971520, 4194304, 16777216, 5.0)
```

## 约束

- `n_elems, num_ops` 为正整数；`dtype_bytes` 常取 1/2/4
- 该模型是**纯访存 roofline**：假设每个算子一读一写、时间由带宽决定，故加速比恰等于算子数 $m$；
  未计 kernel 启动开销、片上容量限制与计算-bound 情形（见 notes 讨论）
