# 解题思路

## 面试官意图

只会调 `dist.all_reduce` 的人，和能说清「环上 $2(W-1)$ 步、每步搬一块、为什么通信量与 $W$ 几乎无关」
的人，差距就在这里。本题把 Ring All-Reduce 拆开让你手动跑一遍——这是理解 NCCL 带宽最优性的唯一硬核方式。

## 为什么是这两段

All-Reduce = **Reduce-Scatter** $+$ **All-Gather**。

- Reduce-Scatter：$W-1$ 步把「每块跨 rank 求和」沿环流动，结束后 rank `q` 只完整拥有第 `q` 块（对角线）。
- All-Gather：再用 $W-1$ 步把这些「完整单块」沿环扩散，结束后人人持有全部块的完整和。

每步每个 rank 只发**一块**（大小 $S/W$），共 $2(W-1)$ 步，于是**每 GPU 通信量** $=2(W-1)\cdot\frac SW$
$=2\frac{W-1}{W}S$——与 `parallel-allreduce`、`distributed-zero-communication` 里的环系数 $2cS$ 完全对上。
关键洞察：这个量几乎不随 $W$ 增长（$c\to1$），带宽最优，这正是 ring 打败「都发给 rank0 再广播」的原因。

## 调度的两个易错点

1. **同步轮语义**：一步内所有发送必须基于**更新前**的缓冲。先算好本步所有 `sends`、再统一落到 `bufs`，
   否则同一步里被改过的值会被下游 rank 二次使用，中间态就错了。
2. **下标公式**：Reduce-Scatter 发 $(q-1-t)\bmod W$、All-Gather 发 $(q-t)\bmod W$，二者差一个偏移——
   差一位就会让「半成品」被当成完整和扩散。题面给了 $W=3$ 的完整轨迹，写不对就逐步手算比对 `step0/step1`。

## 实现要点

`bufs` 用输入各行 `.clone()` 起头（别原地改 `init`）；每步 `torch.stack` 存一个 $(W,W)$ 快照。$W=1$ 时
两个 `range(0)` 都不进，自然返回 `[]`。判题用同调度的独立重放 `_ref` 对拍，另校验末快照==All-Reduce 结果、
reduce-scatter 后对角线==`full`、长度 $=2(W-1)$、不修改输入。

## 相关题

`distributed-all-reduce`（终值）、`distributed-reduce-scatter` / `distributed-all-gather`（单段语义）、
`parallel-allreduce`（把这里的步数折算成通信字节与耗时）。
