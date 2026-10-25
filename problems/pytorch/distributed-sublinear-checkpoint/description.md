# 次线性激活检查点：√n 重算调度的显存峰值

## 题目描述

反向传播需要前向算出的激活。最省时间的做法是**全部存下来**（$n$ 层就占 $n$ 份激活显存）；最省显存的做法是**一层都不存**、反向时从头重算整条前向。梯度检查点（activation checkpointing）走中间路线：**每隔 $k$ 层存一个检查点**，反向走到某段时，从该段起点的检查点出发把中间激活**重算**出来。

Chen 等人（2016，*Training Deep Nets with Sublinear Memory Cost*）证明：只要合理地选检查点，$n$ 层网络的激活显存可以压到 $O(\sqrt n)$，代价只是多跑约 $\sqrt n$ 次前向。本题把这个结论量化成一个纯整数最优化：给定层数与每层激活字节，找出让**显存峰值最小**的检查点间隔 $k$，并报告峰值与全存基线。

## 峰值模型（务必按此定义）

把网络看作 $n$ 个**串行**的层，每层前向产生一份需占 $a=\texttt{act\_bytes}$ 字节的激活。采用「每隔 $k$ 层存检查点」的调度（$1\le k\le n$），反向遍历时的**激活显存峰值**为：

$$ \text{peak}(k) = \Big(\underbrace{\lceil n/k\rceil}_{\text{常驻的检查点数}} + \underbrace{k}_{\text{当前段重算窗口内最多同时存活的激活数}}\Big)\cdot a $$

- 第一项：整条链上按间隔 $k$ 布点，需要常驻的检查点个数上界为 $\lceil n/k\rceil$；
- 第二项：反向进入某一段时，从该段起点重算，段内最多同时存活 $k$ 份激活。

函数 $\lceil n/k\rceil + k$ 在 $k\approx\sqrt n$ 处取得最小值 $\approx 2\sqrt n$，这正是次线性 $O(\sqrt n)$ 显存的来源。

另给全存基线（不做检查点、每层都存）：

$$ \text{store\_all} = n\cdot a $$

## 规则

实现 `solve(n_layers, act_bytes)`，返回三元组 `(best_k, min_peak, store_all)`：

- `best_k`：使 $\text{peak}(k)$ 最小的 $k\in[1,n]$；**若多个 $k$ 并列最小，取最小的 $k$**；
- `min_peak`：$\text{peak}(\texttt{best\_k})$（字节，整数）；
- `store_all`：$n\cdot a$（字节，整数）。
- 三项都是整数（Python `int`，勿用浮点导致精度丢失）。

## 示例

```
n_layers=9, act_bytes=1
  peak(k) = ceil(9/k)+k:  k=1→10, k=2→7, k=3→6, k=4→7, ... k=9→10
  最小在 k=3，peak=6
  store_all = 9
→ (3, 6, 9)          # √9=3，峰值 6≈2√9，全存要 9

n_layers=100, act_bytes=1_000_000
  best_k=10, peak=(ceil(100/10)+10)*1e6=(10+10)*1e6=2e7, store_all=1e8
→ (10, 20000000, 100000000)
```

## 约束

- `1 ≤ n_layers ≤ 100000`，`1 ≤ act_bytes ≤ 1e9`（整数）
- 返回 `(int, int, int)`；用精确整数比较判等
- 只做上述峰值模型的枚举/最优化，不涉及真实框架的重算图与内存分配器
