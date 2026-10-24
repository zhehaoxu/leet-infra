# 集合通信：All-to-All

## 题目描述

**All-to-All** 是 Scatter / All-Gather 的一般化：每个 rank 手里有一批**按目标 rank 分好组**的数据，
通信后把「发给 rank j 的那一组」送到 rank j，于是 rank j 收集齐**所有 rank 发给它**的数据。MoE 专家
并行里把 token 路由到对应专家、张量并行的布局转换，核心就是它。

> 集合通信按纯函数建模：输入是「通信前各 rank 本地数据」列表（第 `q` 项 = rank `q`），输出是「通信后各
> rank 数据」列表。

## 语义

每个 rank `q` 持有一个一维张量 `inputs[q]`（长度 $W\cdot k$），把它均分成 $W$ 个分片，第 `j` 片
`inputs[q][j*k:(j+1)*k]` 是** rank `q` 要发给 rank `j` 的数据**。把这些分片排成一个 $W\times W$ 的网格
（行=源 rank，列=目标 rank），All-to-All 就是**对这个网格做转置**：

$$ \text{out}[j] \;=\; \bigoplus_{q=0}^{W-1} \texttt{inputs}[q][\,j\cdot k:(j+1)\cdot k\,] $$

（$\oplus$ 表示按源 rank `q` 顺序拼接）。即 rank `j` 收到网格的**第 `j` 列**。

## 规则

实现 `solve(inputs)`：`inputs` 是长度 $W$ 的列表，每项是一维张量、长度能被 $W$ 整除。
返回**长度 $W$ 的列表**，第 `j` 项 = 所有 rank 发给 `j` 的分片按源 rank 顺序拼接。不得修改输入。

## 示例

```
W=2, k=2:
输入：inputs = [ [10,20 | 30,40],                 # rank0: 发→0:[10,20], 发→1:[30,40]
                [50,60 | 70,80] ]                 # rank1: 发→0:[50,60], 发→1:[70,80]
输出：[ [10,20,50,60],                             # rank0 收齐「发给0」的(来自 rank0,rank1)
        [30,40,70,80] ]                            # rank1 收齐「发给1」的
```

## 约束

- `1 ≤ W ≤ 64`；每项长度 $=W\cdot k$（`k ≥ 1`），各 rank 分片大小一致
- 返回 `list[torch.Tensor]`，长度 $=W$，每项形状 $=(W\cdot k,)$
- 特例：所有 rank 都只填对角分片 → 退化成 Scatter；每 rank 的 $W$ 个分片都相同 → 退化成 All-Gather
