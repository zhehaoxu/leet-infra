# 解题思路

## 面试官意图

All-to-All 是 MoE（专家并行）绕不开的原语：每个 rank 按「目标专家在哪个 rank」把 token 分桶，再用一次
All-to-All 把每桶送到对应 rank、同时收齐发给自己的那些。能说清它和 Scatter/All-Gather 的包含关系，
就说明真正理解了「数据重分布」这一类通信，而不是只背了名字。

## 一句话本质

把各 rank 的数据看成 $W\times W$ 网格（行=源 rank，列=目标 rank），All-to-All = **网格转置**：
rank `j` 收到的是网格第 `j` 列（所有源 rank 发给 `j` 的分片，按源 rank 顺序拼接）。

## 与相邻原语的关系（考点）

- 每 rank 只填**对角**分片 → 等价于 **Scatter**（一份数据切分发下去）。
- 每 rank 的 $W$ 个分片**全相同** → 等价于 **All-Gather**（`check_degenerates_to_allgather` 固化）。
- 所以 All-to-All 是这两者的一般化；反向说，Scatter/All-Gather 是 All-to-All 的退化特例。

## 两种实现路径

1. **循环拼接**：`out[j] = cat(inputs[q][j*k:(j+1)*k] for q)`——直观，判题独立参考用这条。
2. **reshape-transpose**：`stack(inputs).reshape(W, W, k).transpose(0, 1).reshape(W, W*k)`——向量化，
   参考解用这条。两条路径互相印证，正是 `check_matches_ref_random` 的用意。

$k = \text{len}(\texttt{inputs}[0])//W$ 是每个分片长度；别把 `W` 和 `k` 弄反会导致切片错位。

## 真实通信量

All-to-All 每 GPU 收发各 $(W-1)k$ 数据（自己的对角分片不用发），总计约 $2\cdot\frac{W-1}{W}S$，与
All-Reduce 同量级；但它是**全对全**流量模式，对网络 bisection bandwidth 最敏感，跨节点 All-to-All 常是
MoE 训练的瓶颈，工程上用分层（机内 NVLink + 机间 RDMA）all-to-all 来缓解。

## 判题口径

`test_cases.py`：样例、多组 $(W,k)$ 与独立参考逐 rank 对拍、元素多重集守恒、退化成 All-Gather 的恒等、
单 rank、不修改输入。
