# Kernel 发射时间线：busy/idle 账与 CUDA Graph 收益

## 题目描述

GPU 上一个 kernel 的开销有两套彼此独立的时间线：**CPU 侧**每发射（launch）一次要付固定启动开销，
**GPU 侧**按 kernel 自身耗时在流上串行执行。kernel 一短，CPU 就来不及把后续 kernel 塞进队列，GPU
只能干等——这就是 **launch-bound（发射受限）**，是「GPU 利用率莫名很低」的头号原因。**CUDA Graph**
把一串 kernel 预先捕获成一张图、replay 时只付一次启动开销，正是 PyTorch 官方给的对药。

本题给一条单流 kernel 序列，算这笔 busy/idle 账并量化 graph 收益。口径分工：`operator-fusion` 算的是
融合省下的 **HBM 字节**，`roofline-arithmetic` 给的是**单个算子**的性能上限，本题算的是**一串 kernel 的
发射时间线**——启动开销如何变成 GPU 空转。

## 规则

记 $d_i$ 为 `durations_us[i]`（第 $i$ 个 kernel 的 GPU 时长，$\mu s$），$L$ 为 `launch_overhead_us`
（每次发射的 CPU 开销，$\mu s$），$n$ 为 kernel 个数，$\textstyle W=\sum_i d_i$ 为有效计算时长。

- **eager（每 kernel 各发一次）**：CPU 从 $t=0$ 起连续发射，每个花 $L$，于是第 $i$ 个 kernel 的
  **发射就绪时刻** $r_i=(i+1)L$（首 kernel 也要等自己那次发射，故从 $L$ 起）。单流串行执行：
  $$ e_{-1}=0,\qquad s_i=\max(r_i,\;e_{i-1}),\qquad e_i=s_i+d_i $$
  `eager_wall_us` $=e_{n-1}$。
- **graph（整串捕获，replay 只发射一次）**：一次发射放行全部 kernel，它们背靠背执行：
  $$ \text{graph\_wall}=L+W $$
- `wall_us` = `graph_wall_us`（若 `use_graph`）否则 `eager_wall_us`。
- `busy_ratio` $=W/\text{wall\_us}$（所选时间线的有效计算占比，恒 $\le 1$；分母为 0 时记 `1.0`）。
- `launch_bound_count`：**eager** 时间线里「GPU 因等发射而空转」的 kernel 个数，即满足 $r_i>e_{i-1}$
  的 $i$ 的个数。它是 eager 时间线的属性，与 `use_graph` 无关。
- `speedup` $=\text{eager\_wall}/\text{graph\_wall}$（分母为 0 时记 `1.0`，恒 $\ge 1$）。

## 示例

```
输入：durations_us=[30,5,5,20], launch_overhead_us=20, use_graph=False
  r = [20, 40, 60, 80]
  k0: s=max(20,0)=20, e=50   r0>0      -> 空转
  k1: s=max(40,50)=50, e=55  r1<e0     -> 不空转
  k2: s=max(60,55)=60, e=65  r2>e1     -> 空转
  k3: s=max(80,65)=80, e=100 r3>e2     -> 空转
  eager_wall=100, W=60, graph_wall=20+60=80, busy=60/100=0.6, 空转数=3, speedup=100/80=1.25
输出：(100.0, 80.0, 100.0, 0.6, 3, 1.25)

输入：同上但 use_graph=True
  wall 取 graph_wall=80，busy=60/80=0.75（launch_bound_count 仍按 eager 口径 = 3）
输出：(100.0, 80.0, 80.0, 0.75, 3, 1.25)
```

## 约束

- $1\le$ `len(durations_us)` $\le 4096$；`durations_us[i]` $\ge 0$；`launch_overhead_us` $\ge 0$（均 float，$\mu s$）
- 返回六元组 `(float, float, float, float, int, float)`，浮点用 `atol=1e-5, rtol=1e-5` 判等
- 时间线只在**一条串行流**上定义：不模拟多流并发、CPU 端异步队列深度与 graph 捕获本身的耗时
- 退化口径：$n=1$ 或 $L=0$ 时 eager 与 graph 相同（`speedup=1.0`）；$L=0$ 时 `launch_bound_count=0`；
  全部 $d_i=0$ 时 `eager_wall = n*L`、`graph_wall = L`
