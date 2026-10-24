# ExpectedAttention：对未来注意力取期望

## 题目描述

**ExpectedAttention**（Mülbauer et al., 2024, [arXiv:2406.11430](https://arxiv.org/abs/2406.11430)）：真正的关键不是"过去的 query 打了多少注意力"，而是"**未来的 query 预计会给当前 key 多少注意力**"。用一个高斯模型 $\mathcal{N}(\mu, \Sigma)$ 拟合过去 query，然后对当前 key $k$ 求 $\mathbb{E}_q[e^{q^\top k / \sqrt{d}}]$。高斯积分给出解析式：

$$ \mathbb{E}\big[A(q, k)\big] = \exp\!\Big(\frac{\mu^\top k}{\sqrt{d}} + \frac{k^\top \Sigma\, k}{2 d}\Big) $$

再用 softmax 归一化、乘以 value 的范数（**信息量大的 value 更值得保留**），并在 sink 位置塞一个"全局 max + 1"以强制保留。

## 规则

给定：
- `mu` $(H, D)$：query 均值（**已应用 RoPE 平均旋转后**，本题假设由外部提供）
- `cov` $(H, D, D)$：query 协方差（同上）
- `keys` $(H, L, D)$, `values` $(H, L, D)$
- `n_sink` $s$：前 $s$ 个位置不参与打分、直接以最大值保底
- `epsilon` $\varepsilon$：vnorm 前的偏移

对每个 head $h$：

1. 去掉 sink：$k' = \text{keys}_{h, s:}$, $v' = \text{values}_{h, s:}$
2. 未归一化期望分数：
   $$ u_{h,n} = \frac{\mu_h^\top k'_n}{\sqrt{D}} + \frac{k_n'^\top \Sigma_h\, k'_n}{2D} $$
3. 沿 $n$ 做 softmax：$p_{h,n} = \text{softmax}_n(u_{h,n})$
4. 用 value 范数缩放：$r_{h,n} = (p_{h,n} + \varepsilon)\cdot \|v'_n\|_2$
5. 前 $s$ 位以 $m + 1$ 填充（$m$ 是**跨 head、跨位置**的 $r$ 全局最大值）

返回 shape $(H, L)$。

## 示例

```
H=2, L=6, D=3, n_sink=2, epsilon=0.1

mu  = [[0.5, 0.2, -0.1], [0.1, 0.4, 0.3]]
cov = [[[1.0,0.1,0.0],[0.1,0.8,0.05],[0.0,0.05,0.6]],
       [[0.7,0.0,0.2],[0.0,0.9,0.1],[0.2,0.1,0.5]]]
keys[h=0] = [[1,0,0],[0,1,0],[0.5,0.5,0],[1,1,1],[0.2,0.3,0.4],[0,0,2]]
keys[h=1] = [[0.5,0.5,0.5],[1,0,0],[0,1,0],[2,1,0.5],[0.3,0.3,0.3],[1.5,0,0]]
values[h=0] = [[0.1,0.2,0.3],[0.4,0.5,0.6],[1,0,0],[0,1,0],[1,1,0],[0.5,0.5,0.5]]
values[h=1] = [[0.2,0.1,0],[0.3,0.3,0.3],[2,0,0],[1,1,1],[0,2,0],[0.5,1,1.5]]

softmax over n=4 keys → 与 ||v|| 相乘 → 全局 m=0.930834
scores[h=0] = [1.930834, 1.930834, 0.322050, 0.470150, 0.404386, 0.278735]
scores[h=1] = [1.930834, 1.930834, 0.603884, 0.930834, 0.530053, 0.553043]
```

## 约束

- CPU float32；`use_covariance = True`、`use_vnorm = True`（题目固定）。
- `softmax` 沿最后维（`n`），不做跨 head 归一化。
- sink 位置的填充值是 `scores.max() + 1` — 跨 head 全局 max。
