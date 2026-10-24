# 环形 All-Reduce 通信量与耗时

## 题目描述

数据并行训练每步都要用 **All-Reduce** 把所有卡上的梯度求和同步。Ring（环形）All-Reduce 是无
中心节点的常用实现：它让每卡传输量**与卡数几乎无关**，这正是它能扩展到大集群的原因。本题算出
每卡的通信量和耗时。

## 规则

- 实现 `solve(message_bytes, world, bandwidth_bps, latency_per_hop=0.0)`，返回二元组
  `(bytes_per_rank, time_seconds)`（均为 float）。
- Ring All-Reduce 分两阶段：**Reduce-Scatter** + **All-Gather**，各 $W-1$ 步，每步每卡发送
  $\frac{S}{W}$ 字节（$S=$ `message_bytes`，$W=$ `world`）。故每卡总发送量：
  $$ \text{bytes\_per\_rank} = 2 \cdot \frac{W-1}{W} \cdot S $$
  关键：当 $W$ 增大时系数 $\frac{W-1}{W}\to 1$，每卡传输量趋于 $2S$ 而**不再随 $W$ 线性增长**
  （对比 naive 参数服务器方案的 $2S$ 全压在 server 上）。
- **耗时** = 传输时间 + 逐跳延迟：
  $$ \text{time} = \frac{\text{bytes\_per\_rank}}{\text{bandwidth\_bps}} + 2(W-1)\cdot \text{latency\_per\_hop} $$
  两阶段合计 $2(W-1)$ 个通信跳，每跳一个固定延迟（start-up / 同步开销）。
- 边界：$W \le 1$ 时**无需通信**，返回 `(0.0, 0.0)`。

## 示例

```
输入：message_bytes=1e9, world=8, bandwidth_bps=1e11（100GB/s）, latency_per_hop=0.0
  bytes_per_rank = 2 * (8-1)/8 * 1e9 = 1.75e9   （1,750,000,000）
  time = 1.75e9 / 1e11 + 2*7*0 = 0.0175 秒
输出：(1750000000.0, 0.0175)
```

## 约束

- `message_bytes ≥ 0`，`world ≥ 1`（整数），`bandwidth_bps > 0`，`latency_per_hop ≥ 0`
- 返回二元组 `(bytes_per_rank, time_seconds)`，均为 float
- 判题对 float 用相对容差 `1e-6`
