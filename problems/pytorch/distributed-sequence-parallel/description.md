# 序列/上下文并行：激活切分与通信账

## 题目描述

张量并行沿 hidden 维切、流水线并行沿 layer 维切，而**序列并行 / 上下文并行（Sequence / Context Parallel）**沿**序列长度**维切：把一条长序列切成若干段分给不同 rank，每个 rank 只持有自己那段序列上的激活，从而把最占显存的激活（以及注意力对超长上下文的处理）摊薄到多卡上。Korthikanti 等（2022）的 sequence parallelism 与 Ring Attention（Liu 等, 2023）都是这条路。代价是：注意力、归约这类需要「看全序列」的算子，要在切分/不切分的区域之间用 **All-Gather + Reduce-Scatter** 来回搬运。

本题算一笔简化账：给定单卡整步激活总量、可沿序列切分的比例与并行度，求**每 rank 激活显存**与**每步通信字节**。

## 模型（务必按此定义）

记一步前向在单卡上的**总激活显存** $A=\texttt{total\_activation}$（字节，已含所有层）。

- **可切分比例** $f=\texttt{shardable\_fraction}\in[0,1]$：其中 $f\cdot A$ 是**能沿序列维切分**的激活，$ (1-f)\cdot A$ 是无法切分、必须在每个 rank 上**整份复制**的激活。
- **并行度** $C=\texttt{cp\_degree}$（rank 数）。把可切分部分均分到 $C$ 个 rank，每 rank 只留 $\frac{f\cdot A}{C}$；不可切分部分每 rank 各留整份 $(1-f)\cdot A$。于是每 rank 激活显存：
  $$ \text{per\_rank} = (1-f)\cdot A + \frac{f\cdot A}{C} $$
- **通信**：可切分区域进出「全序列视角」时要做集合通信，每个训练步把所有层的切分激活合计起来各做一次 **All-Gather** 与一次 **Reduce-Scatter**。沿 $C$ 个 rank 成环，做一次这类原语每 rank 搬运带环系数
  $$ c=\frac{C-1}{C} $$
  的量。于是每 rank 每步通信字节：
  $$ \text{comm} = 2\cdot(f\cdot A)\cdot c = 2\,f\,A\cdot\frac{C-1}{C} $$
  （2 = All-Gather + Reduce-Scatter 各一次；与 `distributed-zero-communication` 用的是同一个环系数口径。）

退化校验：$C=1$ 时 $c=0$、可切分部分不用摊，$\text{per\_rank}=A$、$\text{comm}=0$。

## 规则

实现 `solve(total_activation, shardable_fraction, cp_degree)`，返回 `(per_rank, comm)` 两个 float 字节数。

- `total_activation` $A$：单卡整步总激活字节数；
- `shardable_fraction` $f\in[0,1]$：可沿序列切分的比例；
- `cp_degree` $C\ge 1$：序列/上下文并行度（整数 rank 数）。
- $C=1$ 时 `comm` 必为 `0.0` 且 `per_rank == A`。

## 示例

```
A=1000, f=0.8, C=4
  per_rank = 0.2*1000 + 0.8*1000/4 = 200 + 200 = 400
  comm     = 2 * 0.8*1000 * (3/4) = 2*800*0.75 = 1200
→ (400.0, 1200.0)

A=8000, f=0.75, C=8
  per_rank = 2000 + 6000/8 = 2750.0
  comm     = 2*6000*7/8 = 10500.0
→ (2750.0, 10500.0)
```

## 约束

- `total_activation ≥ 0`；`0 ≤ shardable_fraction ≤ 1`；`1 ≤ cp_degree ≤ 4096`（整数）
- 返回 `(float, float)`，相对误差 $10^{-6}$ 内判等
- 只按上述简化模型计算，不涉及具体层的注意力环形交换调度、重叠与真实带宽
