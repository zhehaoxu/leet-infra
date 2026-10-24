# Sequoia：最优草稿树动态规划（Algorithm 1）

## 题目描述

给定一个 draft/verify 配置，Sequoia 要回答：**在投机预算（树大小）$N$、每个节点最多 $B$ 个分支的约束下，怎样的草稿树形结构能让单次迭代期望生成的 token 数最大？** **Sequoia**（Chen et al., 2024, [arXiv:2402.12374](https://arxiv.org/abs/2402.12374)）在**位置接受假设（positional acceptance）** 下给出了闭式期望，并用二维动态规划（其 Algorithm 1）离线求出最优树形——这题就让你实现这个 DP。

## 位置接受假设与期望

验证算法接受"某个已接受节点的第 $k$ 个孩子"的概率只取决于 $k$，记接受向量 $P=(P[1],P[2],\dots,P[B])$（本题入参 `P` 长度 $B$，`P[b-1]` 即 $P[b]$）。对树上节点 $v$，设 $\texttt{Path}(v)$ 为从根到 $v$ 路径上每一步是"第几个孩子"的索引序列，则其分数
$$f(v)=\prod_{i\in\texttt{Path}(v)} P[i],\qquad f(\text{root})=1.$$
验证一棵树 $\mathcal T$ 期望生成的 token 数（Prop 3.4）：
$$F(\mathcal T)=\sum_{v\in\mathcal T} f(v).$$

## DP：张量 T[n,b]

设 $T[n,b]$ = 所有满足「size $=n$ 且根的**直接孩子数（FirstBranch）恰好为 $b$**、每个节点出度 $\le B$」的树中最大的 $F(\mathcal T)$。初始化 $-\infty$，base：
$$T[1,0]=1,\quad T_{\max}[1]=1,\qquad T_{\max}[n]=\max_{0\le b\le B} T[n,b].$$
递归（把第 $b$ 个孩子的子树拆成 size $=m$ 的部分，其子树被计入的前提是第 $b$ 个孩子被接受，概率 $P[b]$）：
$$T[n,b]=\max_{1\le m\le n-1}\Big(T[n-m,\,b-1]+P[b]\cdot T_{\max}[m]\Big),\quad n=2..N,\; b=1..B.$$
答案 = $T_{\max}[N]$（size $\le N$ 中最优，实际取 $n=N$）。

返回 `float`：$T_{\max}[N]$。入参 `P` 长度 $B$，`P[b-1]` 对应 $P[b]$（第 $b$ 个孩子接受率，$b=1..B$）。

## 示例

```
N=3, B=1, P=[0.7]   （只能是一条链）
  f(root)=1, 深度1 = P[1]=0.7, 深度2 = P[1]^2=0.49
  F = 1 + 0.7 + 0.49 = 2.19                       -> 返回 2.19

N=4, B=2, P=[0.9, 0.5]  （允许分叉）
  最优树形经 DP 求得 F = 3.439                    -> 返回 3.439

N=1, B=3, P=[0.9, 0.6, 0.3]  （只有根，无投机）
  F = 1.0                                        -> 返回 1.0
```

## 约束

- $1\le N\le 32$，$1\le B\le N-1$（$N=1$ 时 $B$ 可任意），返回用 `abs(a-b) <= 1e-6*max(1,|b|)` 比较。
- $P[b]\in[0,1]$ 单调性不作要求；DP 覆盖所有 FirstBranch $\le B$、size $=n$ 的有序树取最大。
- $T[n,b]$ 对不可行状态（如 $b>n-1$）保持 $-\infty$，$T_{\max}[n]$ 只在可行 $b$ 上取最大。
