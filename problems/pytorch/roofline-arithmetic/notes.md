# 解题思路

## 面试官意图

Roofline 是性能分析的第一性工具：给你一个算子的计算量和访存量，你能不能立刻判断它到底卡在算力
还是带宽、优化空间有多大？这是做 kernel / 推理优化的基本功，也直接解释了你为什么要在
`operator-fusion` 里做算子融合——本质就是**提高算术强度**。

## 三个量

1. **算术强度 $AI=\text{flops}/\text{bytes}$**：算子本身的属性，衡量「每搬一字节换回多少次运算」。
   逐元素算子 $AI$ 很低（读一堆只做一次加乘），大矩阵乘 $AI$ 高。
2. **屋脊点 $\text{ridge}=\text{peak}/\text{bandwidth}$**：硬件属性，衡量「这台机器每字节需要配多少
   flop 才能喂饱算力」。例如 $200$ TFLOP + $1$ TB/s → 需要 $AI\ge200$ 才不浪费算力。
3. **可达性能 $=\min(\text{peak}, AI\cdot\text{bandwidth})$**：两条屋顶线（水平的算力顶 + 斜的带宽斜坡）
   的包络，取小者。

## 判定规则

- $AI < \text{ridge}$：还在斜坡上 → **memory-bound**，性能 $=AI\cdot\text{bandwidth}$，此时加算力没用，
  要**减少访存**（融合、提高数据复用）。
- $AI \ge \text{ridge}$：已到平顶 → **compute-bound**，性能 $=\text{peak}$，此时该优化算法/用更好的核函数。
- 边界 $AI=\text{ridge}$ 两条线相交，取哪边都一样（本题约定归 compute-bound）。

## 直觉例子

A100/H100 的 ridge 常在数百 flop/byte。所以像 softmax、layernorm、逐元素加这类低 $AI$ 算子几乎永远
memory-bound——这解释了「为什么把它们融合进 matmul 的 epilogue 能白赚性能」。

## 常见错误

- 把 $AI$ 写成 $\text{bytes}/\text{flops}$（分子分母颠倒）。
- memory-bound 时误答性能 $=\text{peak}$（其实还没到顶）。
- ridge 与 $AI$ 的比较方向搞反，导致瓶颈判定与可达性能不一致。

## 判题口径

`test_cases.py` 用 `min(peak, AI*bw)` 包络式独立重算可达性能（不复用 solve 的 if/else），并单测
memory/compute 两侧、$AI=\text{ridge}$ 边界归类、以及「可达性能恒 $\le$ peak」的不变量。
