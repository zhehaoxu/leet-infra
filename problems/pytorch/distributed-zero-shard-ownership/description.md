# ZeRO 一维分片归属：跨 rank 边界的参数切分

## 题目描述

ZeRO-1/2/3 在把训练态沿数据并行切分时，第一步是把所有参数**展平**成一条一维大 buffer，再把它切成尽量等长的连续分片，每个 rank 独占一个分片去做优化器更新。麻烦在于：一个**参数**（或它对应的梯度/优化器状态）在展平 buffer 里占据一段连续区间，这段区间可能正好**跨过两个分片的边界**——于是它被拆给两个 rank 各更新一半。DeepSpeed 的实现里这就是那个反复踩坑的「ownership 偏移」：判定归属时边界是开还是闭、参数恰好压在边界上算不算跨界，写错一个 off-by-one 就会漏掉或错算一个参数。

本题给你一个纯整数模型：算出每个 rank 分到多少元素、以及哪些参数跨越了分片边界。

## 分片模型（务必按此定义）

设参数按展平顺序排成列表，第 $i$ 个参数有 $\texttt{numels}[i]$ 个元素，总元素数 $\texttt{total}=\sum_i \texttt{numels}[i]$。沿数据并行度 $W$ 切成 $W$ 份**尽量等长**的连续分片：

- 分片边界为 $b_r=\left\lfloor \dfrac{\texttt{total}\cdot r}{W}\right\rfloor,\; r=0,1,\dots,W$（注意 $b_0=0,\ b_W=\texttt{total}$）。
- 第 $r$ 个 rank（$0\le r<W$）**独占**全局下标半开区间 $[\,b_r,\ b_{r+1}\,)$，即它负责的元素数为 $b_{r+1}-b_r$。

一个参数 $i$ 占据半开区间 $[s_i,\ e_i)$，其中 $s_i=\sum_{j<i}\texttt{numels}[j]$，$e_i=s_i+\texttt{numels}[i]$。若存在某个**内部**边界 $b_r$（$1\le r\le W-1$）满足

$$ s_i < b_r < e_i $$

（严格在参数区间内部，端点重合不算），则该参数**跨越**了分片边界、被两个 rank 分管。

## 规则

实现 `solve(param_numels, world)`：

- `param_numels`：一列非负整数，按展平顺序给出每个参数的元素数；
- `world`：数据并行度 $W\ge 1$。
- 返回 `(per_rank, straddle)`：
  - `per_rank`：长度为 $W$ 的 `int64` 张量，`per_rank[r]` 是 rank $r$ 拥有的元素数（各元素数之和等于 `total`）；
  - `straddle`：升序 `int64` 张量，元素为**跨越分片边界**的参数的 0-based 下标；没有则返回空张量。
- $W=1$ 时只有一个分片、无内部边界，`straddle` 必为空。

## 示例

```
param_numels = [3, 4, 2, 5, 6], world = 4
  total = 20, 边界 b = [0, 5, 10, 15, 20]
  每个分片：[0,5) [5,10) [10,15) [15,20)  → per_rank = [5, 5, 5, 5]
  各参数区间：p0=[0,3) p1=[3,7) p2=[7,9) p3=[9,14) p4=[14,20)
    p1 含边界 5（3<5<7）→ 跨界
    p3 含边界 10（9<10<14）→ 跨界
    p4 含边界 15（14<15<20）→ 跨界
  straddle = [1, 3, 4]
→ (tensor([5,5,5,5]), tensor([1,3,4]))
```

## 约束

- `0 ≤ len(param_numels) ≤ 1024`，每个 `numels[i] ≥ 0`；`1 ≤ world ≤ 1024`
- 返回两个 `int64` 张量 `(per_rank, straddle)`；用 `torch.equal` 判等
- 这是纯整数归属模型，不涉及 padding / 通信 / 真实显存
