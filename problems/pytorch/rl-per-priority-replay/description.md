# RL 基建：优先经验回放采样与 IS 权重

## 题目描述

回放池（replay buffer）里攒着几万条经验，一次训练只用其中 $K$ 条——**怎么选**这 $K$ 条、以及**选出来之后给多大权重**，是 RL 基建的数据流问题而非算法问题。**PER**（Prioritized Experience Replay, Schaul et al., arXiv:1511.05952）把均匀采样换成按优先级的比例采样

$$P(i)=\frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}}$$

$\alpha=0$ 退回均匀、$\alpha=1$ 完全按比例。比例采样引入了分布偏差，于是给每条样本配一个**重要性采样（IS）权重**做修正：

$$\tilde w_i=\bigl(N\cdot P(i)\bigr)^{-\beta},\qquad w_i=\frac{\tilde w_i}{\max_j \tilde w_j}$$

$\beta=0$ 不修正、$\beta=1$ 完全修正（实践中 $\beta$ 从 0 线性退火到 1）；批内除以最大值归一是标准做法，保证权重不超过 1。

> 关于指数符号：论文原文的写法在不同资料里正反都有，**本题固定采用主流代码口径 $(N\cdot P(i))^{-\beta}$**（等价于「均匀分布 / 实际分布」的比值取 $\beta$ 次幂），别自己换符号。

**随机性显式入参**：不许在函数内部掷骰子。$[0,1)$ 上的均匀随机数由入参给定，你要复现的是「这些随机数落到哪些样本、各自权重多大」这条确定性链路——即 PER 里 sum-tree 之外的另半边。

## 规则

输入四个量：

| 名称 | 类型 | 含义 |
|---|---|---|
| `priorities` | float $(N,)$ | 池内每条样本的优先级 $p_i\ge 0$（通常是 $ \lvert\mathrm{TD}\rvert+\varepsilon$） |
| `uniform_draws` | float $(K,)$ | 外部抽好的均匀随机数 $u_k\in[0,1)$ |
| `alpha` | 标量 $\ge 0$ | 优先级指数 |
| `beta` | 标量 $\ge 0$ | IS 修正指数 |

三步计算：

1. **采样概率** $P(i)=p_i^{\alpha}/\sum_j p_j^{\alpha}$。若 $\sum_j p_j^{\alpha}=0$（全零优先级且 $\alpha>0$），退化为均匀 $P(i)=1/N$。
2. **落到哪个样本**：令前缀和 $F_i=\sum_{j\le i}P(j)$，第 $k$ 个随机数取下标为**最小的** $i$ 使 $F_i>u_k$（即「累积到严格大于」）；若因浮点误差 $u_k\ge F_{N-1}$，截到 $N-1$。
3. **IS 权重**：$\tilde w_k=(N\cdot P(\text{indices}[k]))^{-\beta}$，再除以本批 $\max_j \tilde w_j$。

返回三元组 `(indices, weights, probs)`：`indices` 为 int64 $(K,)$、`weights` 为 float32 $(K,)$、`probs` 为 float32 $(N,)$。空池（$N=0$）无样本可指，`indices` 填 $-1$、`weights` 填 $0$、`probs` 为空张量。

## 示例

`priorities=[1.0, 2.0, 3.0, 2.0]`、`alpha=1.0`、`beta=0.5`、`uniform_draws=[0.05, 0.40, 0.72, 0.95]`：

```
probs   = [0.125, 0.25, 0.375, 0.25]        # 前缀和 0.125 | 0.375 | 0.75 | 1.0
indices = [0, 2, 2, 3]                      # 0.05<0.125→0; 0.40<0.75→2; 0.72<0.75→2; 0.95<1.0→3
weights = [1.00000000, 0.57735026, 0.57735026, 0.70710677]
        # (4·P)^-0.5 = [1.414214, 0.816497, 0.816497, 1.0]，除以 max=1.414214
```

同一批随机数、只把 $\beta$ 换成 $1.0$（完全修正）：

```
weights = [1.00000000, 0.33333334, 0.33333334, 0.50000000]
```

$\alpha=0$ 时（任意 $\beta$）$P\equiv 1/N$，$\tilde w$ 全相等，故 `weights ≡ 1`、下标只由等宽分区决定：

```
probs = [0.25, 0.25, 0.25, 0.25]   indices = [0, 1, 2, 3]   weights = [1, 1, 1, 1]
```

## 约束

- CPU；$N\le 10^5$、$K\le 10^4$；`priorities` 非负；$0\le u_k<1$。
- 建议内部用 float64 累加、输出 float32（判定容差 `atol=1e-5, rtol=1e-5`；`indices` 用 `torch.equal`）。
- 边界：$N=0$（空池，无样本可采）时 `indices` 全为 $-1$、`weights` 全为 $0$、`probs` 为空张量；$K=0$ 时 `indices`/`weights` 为空但 `probs` 仍长 $N$；$N=1$ 时 `indices ≡ 0` 且 `weights ≡ 1`。
- 退化不变量：$\alpha=0\Rightarrow P$ 均匀且 $w\equiv1$；$\beta=0\Rightarrow w\equiv1$；优先级全相等 $\Rightarrow P$ 均匀且 $w\equiv1$；某条 $p_i=0$ 且 $\alpha>0\Rightarrow$ 该条永不被选中、其 $P(i)=0$。
- 本题只做**样本级**索引与权重，不涉及词表分布，与 `sampling-*` 合集的 top-k/nucleus 无交集；优势/损失公式在 `to-*` 合集。
