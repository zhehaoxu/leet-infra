# 集合通信延迟账：Ring vs Recursive-Doubling

## 题目描述

同一个 All-Reduce，可以用完全不同的通信拓扑去实现：**Ring（环形）** 步数多但每步只搬小数据、带宽利用最优；**Recursive-Doubling（递归倍增 / 蝶形）** 步数只有 $\log_2 W$、延迟低，但每步搬的数据更多。谁快取决于「消息大小 vs 单次启动开销」——这就是经典的 **α-β 代价模型**要回答的问题（NCCL 自动选算法就是在算这笔账）。

## α-β 模型（务必按此公式，勿用其它变体）

记：$W$ = rank 数（$W$ 为 2 的幂），$S$ = 每 rank 的消息字节数，$\alpha$ = 每一步的启动/延迟开销（秒），$\beta$ = 每字节传输时间（秒/字节）。

- **Ring All-Reduce**：$2(W-1)$ 步，每步搬 $\frac{S}{W}$ 字节：
  $$ T_{ring} = 2(W-1)\left(\alpha + \frac{S}{W}\,\beta\right) $$
- **Recursive-Doubling All-Reduce**：$\log_2 W$ 步，每步搬整份 $S$ 字节：
  $$ T_{rd} = \log_2 W\,(\alpha + S\,\beta) $$

直觉：$S$ 大 → ring 的带宽项 $\frac{2(W-1)}{W}S\beta\ (\approx 2S\beta)$ 远小于 rd 的 $\log_2 W\cdot S\beta$，ring 赢；$\alpha$ 大、$S$ 小 → 步数少（$\log_2 W \ll 2(W-1)$）的 rd 赢。

## 规则

实现 `solve(W, S, alpha, beta)`，返回字典：

- `"ring_time"`：$T_{ring}$（float）
- `"rd_time"`：$T_{rd}$（float）
- `"faster"`：`"ring"` 若 $T_{ring}\le T_{rd}$，否则 `"recursive_doubling"`（**相等时判 ring**）
- `"speedup"`：较慢者 / 较快者 $=\max/\min$（float，$\ge 1$）

## 示例

```
W=8, S=1e6, alpha=1e-5, beta=1e-9:
  T_ring = 14*(1e-5 + 125000*1e-9) = 0.00189
  T_rd   = 3*(1e-5 + 1e-3)         = 0.00303
  → {"ring_time":0.00189, "rd_time":0.00303, "faster":"ring", "speedup":1.6031746}
```

## 约束

- $W\in\{2,4,\dots,64\}$ 且为 2 的幂；$S,\alpha,\beta > 0$（float）
- 严格用上面两条公式；浮点用相对误差 $10^{-6}$ 判等，`faster` 用字符串精确匹配
- 不涉及具体带宽/拓扑模拟，纯闭式代数计算
