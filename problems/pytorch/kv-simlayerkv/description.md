# KV Cache：SimLayerKV 懒惰层二值保留

## 题目描述

多数跨层方法（PyramidKV 等）给每层一个**数值预算**；SimLayerKV（Zhang et al., 2024, [arXiv:2410.13846](https://arxiv.org/abs/2410.13846)；kvpress `NVIDIA/kvpress` 的 `kvpress/presses/simlayerkv_press.py` 中 `SimLayerKVPress`）把跨层压缩简化成每层独立的 **0/1 决策**："懒惰层"只需要头尾信息就能工作——最后若干条 query 的注意力若几乎全部落在**最前 \(n\_initial\) 个（attention sink）+ 最近 \(n\_recent\) 个** token 上，说明这层根本不用中间上下文，整段中间 cache 可直接丢弃；否则该层**全量保留**。没有 top-k、没有分数排序，只有一组注意力质量和一次阈值比较。

## 规则

输入：`last_attn` 形状 \((N, L)\)——第 \(l\) 层里**最后 \(n\_last\) 条 query** 对全部 \(L\) 个 key 位置的注意力权重，已对 batch、全部 query head 与窗口行取过平均；整数 `n_initial`、`n_recent`、`n_last`；标量 `lazy_threshold` \(\theta\in[0,1]\)。

对每一层 \(l\)：

1. **禁用分支**：若 \(\theta = 1.0\) 或 \(L\le n\_initial+n\_recent+n\_last\)，该层原样保留（掩码全真、压缩比 0）。前者是上游实现的真实行为：默认参数 \(\theta=1.0\) 即"关闭压缩"，必须显式调低阈值才会生效。
2. **懒惰判定**：
   \[ s_l=\sum_{j<\min(n\_initial,L)} a_{l,j}+\sum_{j\ge L-\min(n\_recent,L)} a_{l,j} \]
   \(s_l\) **严格大于** \(\theta\) 才算懒惰（\(s_l=\theta\) 不懒）。
3. **懒惰层保留集**：\([0, n\_initial)\ \cup\ [\,L-(n\_recent-n\_last),\ L)\)，其余位置全部驱逐；该层压缩比
   \[ r_l=\frac{L-n\_initial-(n\_recent-n\_last)}{L} \]
   非懒惰层保留全部、\(r_l=0\)。（进入第 3 步时必有 \(L>n\_initial+n\_recent+n\_last\)，两段保留区间不相交。）

返回两个张量：`mask`（bool，\((N, L)\)，真 = 保留）与 `ratios`（float32，\((N,)\)）。

## 示例

```
N=2, L=10, n_initial=2, n_recent=4, n_last=1, θ=0.5

layer0: [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1]
        s = (0.2+0.2) + (0.0+0.3+0.2+0.1) = 1.0 > 0.5 → 懒惰
        保留 [0,2) ∪ [10-3,10) = {0,1,7,8,9}，r = (10-2-3)/10 = 0.5
layer1: [0.1, 0.05, 0.2, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05, 0.05]
        s = (0.1+0.05) + (0.05+0.1+0.05+0.05) = 0.4 ≤ 0.5 → 不懒
        全保留，r = 0.0

mask   = [[T, T, F, F, F, F, F, T, T, T],
          [T, T, T, T, T, T, T, T, T, T]]
ratios = [0.5, 0.0]
```

## 约束

- CPU float32；`last_attn` 非负（注意力均值）；\(n\_recent>n\_last\ge1\)；\(N,L\ge1\)。
- 阈值比较是**严格大于**；\(\theta=0\) 且 \(s_l=0\)（全零注意力）时不懒惰。
- \(n\_initial\) 或 \(n\_recent\) 超过 \(L\) 时求和区间按 \(L\) 截断（只可能走禁用分支）。
- `mask` 用 bool 张量返回；懒惰层的掩码形如「左块 + 右块」，两个块之间为一段连续假值。
