# 选择性激活重算：显存预算下的最小重算代价

## 题目描述

全量梯度检查点把**每一层**的激活都丢掉、反向时整层重算，省显存但重算开销大。Korthikanti 等人（2022，*Reducing Activation Recomputation in Large Transformer Models*）提出**选择性激活重算（selective activation recomputation）**：不是所有激活的重算代价都一样——像 LayerSoftmax、Dropout 这类逐元素算子占显存小、重算便宜，而注意力里的 $SQK$、$S^2V$ 这类大矩阵乘占显存大、重算贵。策略是**只保留（存下来）少数最占显存 / 最贵重算的激活，其余一律重算**，在同样的显存预算下把总重算时间压到最低。

把它抽成一道纯整数最优化：给定每个算子块的激活字节与重算代价，在**激活显存总预算**内决定保留哪些，使**被重算算子的总代价最小**。

## 模型（务必按此定义）

共 $N$ 个算子块，编号 $0,\dots,N-1$。第 $i$ 块：

- $\texttt{mem}[i]$：若**保留**它这一步的激活，需要占用的显存字节数（正整数）；
- $\texttt{cost}[i]$：若不保留它、反向时**重算**它所需的时间代价（正整数）。

选择保留集合 $K\subseteq\{0,\dots,N-1\}$，须同时满足：

1. **显存预算**：$\sum_{i\in K}\texttt{mem}[i]\le \texttt{budget}$；
2. **保底保留数**：$|K|\ge \texttt{min\_keep}$（至少保留这么多个算子块的激活）。

目标是最小化**重算总代价**：

$$ \min_{K}\ \sum_{i\notin K}\texttt{cost}[i] \;=\; \Big(\sum_i \texttt{cost}[i]\Big) - \max_{K}\sum_{i\in K}\texttt{cost}[i] $$

若不存在任何满足两条约束的 $K$（例如把 $\texttt{mem}$ 最小的 $\texttt{min\_keep}$ 个加起来就已超过 `budget`），返回**不可行**。

## 规则

实现 `solve(mem, cost, budget, min_keep)`：

- `mem`、`cost`：等长的正整数序列（列表即可），第 $i$ 项对应算子块 $i$；
- `budget`：非负整数，激活显存总预算；
- `min_keep`：非负整数，保留算子块数的下限。
- 返回**最小的重算总代价**（int）。
- 若问题不可行（无满足约束的保留集合），抛出 `ValueError`。

## 示例

```
mem  = [30, 10, 20, 40, 15]
cost = [50, 20, 30, 60, 10]
budget = 60, min_keep = 2
  保留 {0,1,2}：mem=60≤60, |K|=3≥2, kept cost=50+20+30=100 → 重算=170-100=70
  保留 {1,3}  ：mem=50, |K|=2, kept cost=20+60=80 → 重算=90（更差）
  最优重算代价 = 70
→ 70
```

## 约束

- `1 ≤ N ≤ 50`；`1 ≤ mem[i], cost[i] ≤ 2000`；`0 ≤ budget ≤ 2000`；`0 ≤ min_keep ≤ N`
- 返回单个 int；不可行抛 `ValueError`
- 这是纯组合最优化模型，不涉及真实计算图 / 算子 profiler
