# 滑动窗口注意力详解

## 掩码 = 因果 ∧ 窗口

位置 $i$ 可见的 $j$ 需同时满足两条：

$$ j \le i \quad(\text{因果}),\qquad i - j < \text{window}\quad(\text{回看上限}) $$

合成带宽 $w$ 的下三角对角带。用一维 `idx` 广播成 $(L,L)$ 比较矩阵最省事：

```
causal = idx[:, None] >= idx[None, :]
within = (idx[:, None] - idx[None, :]) < window
mask   = causal & within          # True = 可见
```

注意本合集沿用了「bool mask **True=可见**」的约定（与 SDPA 一致），
`masked_fill(~mask, -inf)` 才是屏蔽；写反会把上三角当成可见。

## 易错点

- **窗口边界用 `<` 而非 `<=`**：`i-j < window` 让窗口恰好含 `window` 个 token（含自己）；
  写成 `<=` 会多回看一个。
- **对角线恒可见**：`window>=1` 保证 $i-j=0<window$，所以每行至少有一个非 $-\infty$，
  softmax 不退化。若允许 `window=0` 会出现整行 $-\infty$ → NaN。
- **退化关系**：`window >= L` 时 `within` 恒真，等价纯因果注意力——可作回归用例。
- **局部性探针**：扰动远处 token 的 v，应只影响其后 `window` 个位置；这能抓出「只做了因果、
  忘了窗口」的半对实现。

## 应用与延伸

Mistral 7B 用 `window=4096` 的滑窗，让超出窗口的 token 靠**层间堆叠**扩大感受野
（每层各看 4096，多层后有效上下文远大于窗口）；Longformer 则用「滑窗 + 少量全局 token」
混合。滑窗使 KV cache 只需保留最近 `window` 个 token，是长序列推理省显存的关键。
