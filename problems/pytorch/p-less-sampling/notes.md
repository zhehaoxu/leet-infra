# 解题说明 · P-less 采样

## 碰撞概率作阈值

按分布 $p$ 独立抽两个 token，两者**相同**的概率就是 $\sum_i p_i^2$（碰撞概率，等于
Rényi-2 熵的指数 $2^{-H_2}$）。P-less 的判据非常朴素：**连「随机抽中自己」的概率都不如
的 token，不值得留**——即丢掉 $p_i < \sum_j p_j^2$ 的长尾。

$$\text{thr} = \sum_i p_i^2,\qquad \text{keep}_i = [\,p_i \ge \text{thr}\,]$$

## 为什么免超参

- **尖分布**（$p_{\max}\to 1$）：$\sum p^2 \to 1$，门槛极高，几乎只剩 argmax。
- **平分布**（$V$ 个等概率 $1/V$）：$\sum p^2 = V\cdot(1/V)^2 = 1/V$，而每个 $p_i = 1/V$，
  恰好全部保留。

于是同一套代码在不同置信度下自动给出不同的截断宽度，无需像 top-p 那样为每个模型调 `p`。

## 关键不变量

- **argmax 必留**：$\sum_i p_i^2 = \sum_i p_i\cdot p_i \le p_{\max}\sum_i p_i = p_{\max}$，
  所以 $p_{\max}\ge\text{thr}$ 恒成立，`filtered` 分母恒正、不会全被清空。
- 阈值用的是**概率** $p$（softmax 之后），不是 logit；$\sum p^2$ 在概率单纯形上取值
  $\big[1/V,\,1\big]$。

## 实现要点

`thr = probs.pow(2).sum(-1, keepdim=True)`；`keep = probs >= thr`；`filtered = probs*keep`
再除以 `filtered.sum(-1, keepdim=True)`。采样与合集内其它题一致：对 `filtered` 原顺序 CDF
取首个 $\ge u$。注意用 `>=`（论文与官方实现保留等于阈值的项）。
