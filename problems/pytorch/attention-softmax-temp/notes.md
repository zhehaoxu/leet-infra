# 温度缩放 softmax 与 Top-p 采样详解

## 温度缩放

在 softmax 前把 logits 除以温度 $T$：

$$ p_i=\text{softmax}(z/T)_i $$

- $T>1$：分布更**平坦**（更随机、多样性高）；
- $T<1$：更**尖锐**，$T\to0$ 逼近 argmax one-hot；$T=1$ 即原始 softmax。
- 仍是先除温度、**再减行最大值**求 exp（数值稳定不变，见「Softmax」关）。缩放对减法顺序无碍，但 `logits/T` 一定在归一化前。

## Top-p（nucleus）截断

采样策略：不按固定 top-k，而是按**累计概率阈值** $p$ 取最小候选集（核），丢掉长尾，再在核内重归一化。相比 top-k，候选数随分布陡度自适应。

实现（本参考解）关键：

1. 先温度 softmax 得 `probs`，**降序排序**并记录索引 `idx`。
2. 累计 `cum = cumsum(sorted)`；判「保留」用
   $$ \text{remove}=(\text{cum}-\text{sorted})\ge p $$
   即**不含自己**的累计仍 $<p$ 的项留下——这样**保证至少保留 1 项**（否则首项自身就 $\ge p$ 会被全删）。这是最易写错的一行。
3. 被删位置 0，保留的重归一化，再用 `scatter_(-1, idx, kept)` 还原到原顺序。

## 口径提醒

`top_p>=1` 或 `None` 时不截断，直接返回温度分布。`cum - sorted` 是「排除自身前的累计」，写成 `cum >= p` 会把刚好跨过阈值的那一项也删掉，与 HuggingFace 实现不一致。
