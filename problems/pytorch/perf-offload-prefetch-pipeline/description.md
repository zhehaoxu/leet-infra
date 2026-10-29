# Offload 预取流水线：单引擎多缓冲下的计算-传输重叠时间线

## 题目描述

显存不够时，ZeRO-Offload / ZeRO-Infinity 这类方案把参数或优化器状态放在 DRAM（甚至 NVMe），按层搬进 HBM 再算。搬运和计算如果能**重叠**，墙钟时间就等于「计算总时长 + 没被盖住的传输时长」；盖不住的部分就是 **stall（空等）**。

能否重叠取决于两件事：**搬运引擎是串行的**（同一时刻只能搬一层），以及 **HBM 上有几个缓冲槽**——槽数 `buffers=1` 时，下一层必须等当前层算完、槽释放后才能开始搬，完全无法重叠；槽数 ≥ 2 才能做到「一边算当前层、一边搬下一层」。

本题用整数时间线把这条流水线算清楚：给定逐层计算耗时、逐层传输字节、引擎带宽与槽数，输出 makespan、总空等时间，以及每层的计算开始、预取开始、预取结束时刻。

## 规则

`compute[j]` 为第 `j` 层计算耗时，`transfer_bytes[j]` 为第 `j` 层需从 DRAM 搬入的字节，`bw` 为引擎带宽（字节/单位时间），`buffers` 为 HBM 缓冲槽数。单引擎、严格按层序预取、计算与传输可同时进行：

1. 传输耗时 `transfer_time[j] = ceil(transfer_bytes[j] / bw)`（0 字节记 0，不占引擎）；
2. 第 `j` 层占用槽 `j % buffers`，从**预取开始**一直占到**计算结束**；
3. 预取开始 `s_j = max(engine_free, finish[j-buffers])`（`j < buffers` 时后者记 0）：`engine_free` 是上一层预取的结束时刻（引擎串行），`finish[j-buffers]` 是占用同一槽的旧层算完的时刻；
4. 预取结束 `e_j = s_j + transfer_time[j]`；
5. 计算开始 `c_j = max(finish[j-1], e_j)`（首层的 `finish[-1]` 记 0）——不能早于上一层的计算结束，也不能早于自己的预取完成；
6. `finish[j] = c_j + compute[j]`；**空等**定义为 `Σ(c_j - finish[j-1])`；`makespan = finish[L-1] = Σ compute + 空等`。

返回 `(makespan, stall_total, compute_start, transfer_start, transfer_end)`，后三项是按层序排列的列表。

## 示例

`bw=50`，四层，`compute=[5,3,4,2]`、`transfer_bytes=[100,200,50,150]` → 传输耗时 `[2,4,1,3]`，`buffers=2`：

```
j0: 槽0 空 -> s=0  e=2  c=2   finish=7       槽0 占 [0,7)
j1: 槽1 空 -> s=2  e=6  c=7   finish=10      槽1 占 [2,10)
j2: 槽0 在 finish[0]=7 释放 -> s=7  e=8  c=10 finish=14
j3: 槽1 在 finish[1]=10 释放 -> s=10 e=13 c=14 finish=16
```

```
makespan      = 16        # = Σcompute 14 + 空等 2
stall_total   = 2         # 只有首层干等了第一次预取
compute_start = [2, 7, 10, 14]
transfer_start= [0, 2, 7, 10]
transfer_end  = [2, 6, 8, 13]
```

对照 `buffers=1` 的同一组数据：每层都要等上一层算完才能开搬，空等变成 `2+4+1+3=10`，makespan 变成 `24`。

## 约束

- `compute[j] >= 1`、`transfer_bytes[j] >= 0`、`bw >= 1`、`buffers >= 1`；允许空输入（返回 `0, 0` 与三个空列表）。
- 传输耗时是**向上取整**：40 字节 / 30 字节每秒 = 2 个单位，不是 1.33。
- 槽位只在下标对 `buffers` 取模的层之间复用；`j < buffers` 的层一开始就能预取（槽初始为空）。
- 0 字节层的预取不占用引擎时间，但仍受槽位约束（`buffers=1` 时要等上一层算完才能「发起」）。
- 返回值类型固定：前两项 `int`，后三项 `list[int]`。
