# 集合通信：Scatter

## 题目描述

**Scatter** 是 All-Gather 的逆操作：源 rank 手里有一份**完整数据**，通信后把它**均分成 $W$ 段**，
第 `i` 段只发给 rank `i`。于是每个 rank 最后只留下自己那一段。它常用于把整批数据按卡切分下发。

> 集合通信按纯函数建模：输入是源 rank 的完整张量与 rank 数，输出是「分发后各 rank 的数据」列表。

## 规则

- 实现 `solve(full, world)`：`full` 是一维张量（长度 $L$ 能被 `world` $=W$ 整除），`world` 是 rank 数。
- 把 `full` 均分成 $W$ 段，每段 `chunk = L / W`；rank `i` 得到第 `i` 段：
  $$ \text{out}_i = \texttt{full}[\,i\cdot\text{chunk} : (i+1)\cdot\text{chunk}\,] $$
- 返回**长度 $W$ 的列表**，第 `i` 项是 rank `i` 收到的那一段（各 rank 结果不同）。
- 不得修改输入张量。

## 示例

```
输入：full = [1,2,3,4,5,6], world = 3     # L=6, chunk=2
输出：[ [1,2], [3,4], [5,6] ]             # rank0←前段, rank1←中段, rank2←后段
```

## 约束

- `1 ≤ world ≤ 64`；`full` 为一维、长度能被 `world` 整除（chunk ≥ 1）
- 返回 `list[torch.Tensor]`，长度 $=W$，每项形状 $=(L/W,)$
- 与 Reduce-Scatter 的区别：Scatter **不做跨 rank 求和**，只是把一份数据切片分发
