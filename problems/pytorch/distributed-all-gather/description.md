# 集合通信：All-Gather

## 题目描述

**All-Gather** 是最基础的集合通信原语：每个 rank 手里有一份「本地分片」，操作后**每个 rank 都拿到
所有分片按 rank 顺序拼起来的完整结果**。它是张量并行前向、数据并行收集参数的底层积木。

> 本题没有真实进程组。把一个集合通信建模成**纯函数**：输入是「通信前各 rank 的本地数据」（一个列表，
> 第 $r$ 项是 rank $r$ 的分片），输出是「通信后各 rank 的数据」（同样是列表，第 $r$ 项是 rank $r$
> 拿到的结果）。你要实现的就是这个映射。

## 规则

- 实现 `solve(shards)`，`shards` 是长度 $W$ 的列表，`shards[r]` 是 rank `r` 持有的一维张量（各 rank 等长）。
- All-Gather 后，每个 rank 得到按 rank 顺序的拼接：
  $$ \text{full} = [\ \texttt{shards[0]} \Vert \texttt{shards[1]} \Vert \cdots \Vert \texttt{shards[}W\text{-1]}\ ] \quad (\dim 0 \text{ 拼接}) $$
- 返回**长度 $W$ 的列表**，每个元素都等于 `full`（所有 rank 结果相同）。
- 不得修改输入张量。

## 示例

```
输入：shards = [ [10, 11], [20, 21], [30, 31] ]   # 3 个 rank，各持长度 2
  full = [10, 11, 20, 21, 30, 31]
输出：[ [10,11,20,21,30,31], [10,11,20,21,30,31], [10,11,20,21,30,31] ]   # 3 份相同
```

## 约束

- `1 ≤ W ≤ 64`；每个分片为一维、等长、长度 ≥ 1
- 返回 `list[torch.Tensor]`，长度 $=W$，每个形状 $=(W\cdot\text{len},)$
- 分片 dtype 以 float32 为主；`torch.equal` 逐元素判等
