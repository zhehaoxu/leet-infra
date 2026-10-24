# 集合通信：Gather

## 题目描述

**Gather** 是 All-Gather 的「单目标版」：每个 rank 交出自己的一片，但拼接后的**完整结果只送到一个目标
rank**（`dst`），其余 rank 拿不到全体、**保持自己原来那片不变**。真实用法如各卡把本地统计 gather 到
rank0 汇总，而不必广播回所有人。

> 集合通信按纯函数建模：输入是「通信前各 rank 本地数据」列表（第 `q` 项 = rank `q`），输出是「通信后各
> rank 数据」列表。

## 规则

实现 `solve(inputs, dst)`：`inputs` 是长度 $W$ 的列表，每项一维张量、**长度相同**；`dst` 是目标 rank。
返回**长度 $W$ 的列表**，其中：

- 第 `dst` 项 $= \bigoplus_{q=0}^{W-1} \texttt{inputs}[q]$（各 rank 分片按 rank 顺序拼接成完整结果）；
- 第 `j`（`j != dst`）项 $= \texttt{inputs}[j]$（**原样保留**自己那片，gather 不给它们结果）。

不得修改输入张量。

## 示例

```
输入：inputs = [ [1,2], [3,4], [5,6] ], dst = 1
输出：[ [1,2],                      # rank0 非目标：保留原片
        [1,2,3,4,5,6],              # rank1=dst：收齐全体拼接
        [5,6] ]                     # rank2 非目标：保留原片
```

## 约束

- `1 ≤ W ≤ 64`；`0 ≤ dst < W`；各分片长度相同
- 返回 `list[torch.Tensor]`，长度 $=W$；`out[dst]` 长度 $=W\cdot\text{shard}$，其余各项等于对应输入
- 与 **All-Gather** 的区别：All-Gather 把完整结果给**所有** rank，Gather 只给 **`dst` 一个** rank
- 与 **Reduce** 的区别：Gather 是**拼接**（不聚合），数值上不做求和
