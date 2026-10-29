# 稀疏化：剪枝微调的 STE 与软阈值重参数化反向

## 题目描述

剪枝之后做微调时，前向要用**剪过的**权重 \(w\odot m\)，但反向怎么办？二值掩码 \(m=\mathbb{1}[|w|\ge\tau]\) 对 \(w\) 的导数是 0，直接反传会让被剪权重永远收不到梯度、也无法「复活」。工程上的两种解法：

- **STE（straight-through estimator）**：前向用硬掩码，反向把掩码当常数，梯度 \(g\odot m\) 直通——保留位置的梯度照常更新，被剪位置恒为 0。这是最常用的剪枝微调做法，代价是梯度与前向不完全一致（Bengio et al. 2013 讨论的正是这种直通偏差）。
- **WSN（soft threshold weight reparameterization）**：把掩码换成连续松弛 \(m=\mathrm{clamp}(|w|/\tau,0,1)\)，前向 \(y=w\cdot m\) 在 \(|w|<\tau\) 区间里是 \(w|w|/\tau\) 的二次曲线，反向既含 \(m\) 项也含掩码本身对 \(w\) 的导数：当 \(|w|<\tau\) 时 \(\mathrm{d}y/\mathrm{d}w = 2|w|/\tau\)（乘上游梯度），\(|w|\ge\tau\) 时退化为 1。Kusupati et al. (ICML 2021) 用它把「剪不剪」变成可学习的阈值问题。

本题把上游梯度 `grad_out` 直接给你，只需按两种口径分别算出前向输出、掩码与对权重的梯度，不需要 autograd。

## 规则

给定 `weight`（float32，任意形状）、标量 `tau > 0`、上游梯度 `grad_out`（float32，同形状），以及 `mode`：

- `mode="hard_ste"`：
  - 掩码 \(m=\mathbb{1}[|w|\ge\tau]\)（**含等号**）；
  - 前向 \(y=w\odot m\)；
  - 梯度 \(g_w=g\odot m\)（掩码不参与求导）。
- `mode="wsn"`：
  - 掩码 \(m=\mathrm{clamp}(|w|/\tau,\,0,\,1)\)；
  - 前向 \(y=w\odot m\)；
  - 梯度 \(g_w=g\odot\big(m+\frac{|w|}{\tau}\cdot\mathbb{1}[|w|<\tau]\big)\)。

返回 `(y, grad_w, mask)`，三者形状同输入、float32。禁止 `torch.autograd`。

## 示例

```
weight   = [0.5, -1.5, 2.0, -0.25]
tau      = 1.0
grad_out = [1, 2, 3, 4]

hard_ste:
  mask  = [0, 1, 1, 0]           # |w| = [0.5, 1.5, 2.0, 0.25] 中 >= 1 的两位
  y     = [0, -1.5, 2, 0]
  grad  = [0, 2, 3, 0]

wsn:
  mask  = [0.5, 1.0, 1.0, 0.25]  # clamp(|w|/1.0, 0, 1)
  y     = [0.25, -1.5, 2, -0.0625]
  grad  = [1*(0.5+0.5), 2*1, 3*1, 4*(0.25+0.25)] = [1, 2, 3, 2]
```

## 约束

- CPU float32；`tau > 0`；`grad_out` 已给定，不得调用 autograd。
- `hard_ste` 在 \(|w|=\tau\) 处保留（掩码为 1），梯度照常直通。
- `wsn` 在 \(|w|\ge\tau\) 时掩码饱和为 1、梯度恰好等于上游梯度；在 \(|w|<\tau\) 时梯度系数为 \(2|w|/\tau\)。
- 两种模式下前向都必须满足 \(y=w\odot m\)。
