# 解题说明 · Top-n-sigma 采样

## 思想：让阈值跟着分布形状走

top-k 固定保留 `k` 个、top-p 固定保留质量 `p`，两者都不看分布本身的**陡峭程度**。
Top-n-sigma 把截断线放在「比最大值低 $n$ 个标准差」的位置：

$$\text{keep}_i = \big[\, z_i \ge z_{\max} - n\sigma \,\big],\qquad
\sigma = \sqrt{\tfrac1V\sum_i (z_i-\bar z)^2}$$

- 分布很**尖**（少数 logit 很高、其余接近 $-\infty$）→ $\sigma$ 大 → 阈值下探得多，但其实
  高分候选彼此拉得很开，低分长尾被 $n\sigma$ 一刀切掉。
- 分布很**平**（logit 挤在一起）→ $\sigma$ 小 → 阈值贴近 max，只留下最靠前的极少数。

这样无需为每个模型手调 `k`，一个 `n`（llama.cpp 默认 2.0）就能自适应。

## 与 llama.cpp 对齐的三个细节

1. **总体标准差**：`std = sqrt(acc / valid_count)`，分母是元素个数 `V`，**不是** `V-1`
   （`unbiased=False`）。用错样本方差会让阈值整体偏移、边界处保留集合不同。
2. **作用在 logit 域**：`mean/std/max` 都在原始 logits 上算，`-INF` 的槽位应被排除。本题
   输入无掩码，直接对整行统计即可。截断判据 `logit < max - n*std → 丢弃`，取反即保留集。
3. **先 softmax 再按 logit 阈值筛**：softmax 单调，保留 `logit>=cut` 等价于保留这些位置后
   重新归一化，与「把丢弃 logit 置 $-\infty$ 再 softmax」结果相同。

## 采样

保留集合归一化得到 `filtered`（原顺序、和为 1），再对 `filtered` 的原顺序 CDF 做逆变换
（首个 $\ge u$ 的位置）→ `chosen`。因为至少保留 argmax（它必 $\ge$ max $-n\sigma$），
`filtered` 分母恒正、`chosen` 恒在保留集合内。
