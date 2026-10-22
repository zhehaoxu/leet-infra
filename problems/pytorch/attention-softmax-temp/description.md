# 温度缩放与 Top-p 采样（Temperature + Nucleus Sampling）

## 题目描述

解码时先对 logits 做**温度缩放** softmax 控制分布尖锐程度，再用 **top-p（nucleus）截断**：只保留按概率降序累计首次达到 $p$ 的头部集合，其余置 0 并重新归一化。

$$ \text{softmax}_T(z)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} $$

- $T \to 0$：分布趋近 argmax 的 one-hot（更确定）；
- $T \to \infty$：分布趋近均匀（更随机）。

## 规则

- 实现 `solve(logits, temperature=1.0, top_p=None)`，沿最后一维逐行返回概率，行和为 1。
- 温度在 softmax **之前**作用于 logits：$z/T$。必须减每行最大值保证数值稳定（大 logits 不上溢）。
- **top-p nucleus**（`top_p` 给定且 $<1$ 时）：对每行按概率降序排列，累计概率 `cumsum`，丢弃满足 `(cumsum - 自身概率) >= top_p` 的尾部项——即**保留使累计首次达到 $p$ 的头部集合**，且无论如何**至少保留 1 项**（当最大单项概率已 $\ge p$ 时只剩 1 项）。保留项重新归一化，被丢弃项为 0，输出恢复原始顺序。
- `top_p=None` 或 `>=1.0` 时不截断。

## 示例

```
输入：logits=[[2,1,0.5,0.1]], temperature=1.0
输出：[[0.5745217, 0.2113547, 0.1281931, 0.0859304]]   # 标准 softmax

输入：logits=[[10,9,0,-5]], temperature=1.0, top_p=0.9
   softmax 前两项概率和≈1.0，第三项起累计已超 0.9 → 截断尾部
输出：[[0.7310585, 0.2689414, 0.0, 0.0]]   # 保留两项并重归一化
```

## 约束

- float32/float64 输入，支持任意前置批量维
- `top_p` 极小时也必须保留至少 1 项（不得出现整行 0）
