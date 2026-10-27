# KV Cache：ScissorHands 时间一致性滞回驱逐

## 题目描述

按**单帧**注意力分数剪 KV 缓存有一个致命问题：分数本身噪声很大，某个关键 token 可能因为一次性的注意力分散而被判低分，下一帧又恢复重要，但它已经被永久删掉。ScissorHands（Liu et al., NeurIPS 2023, arXiv:2305.17118）把这个现象总结为「**重要性的持续性假设**」（persistence of importance）：token 的重要性是缓慢变化的，因此**驱逐决策也应该带记忆**。

它的做法是给每个 token 维护一个「**连续低分计数**」：只有当重要性**连续 \(m\) 帧**都低于阈值 \(\tau\) 时才驱逐，任意一帧回升就把计数清零。这就是控制论里的**滞回（hysteresis）**——用一个「耐心窗口」滤掉单帧抖动，代价是对真正的失效反应慢 \(m-1\) 帧。

形式化地，第 \(s\) 步对 token \(j\)：

\[
c_s[j]=
\begin{cases}
c_{s-1}[j]+1, & \text{score}_s[j]<\tau\\[2pt]
0, & \text{score}_s[j]\ge\tau
\end{cases}
\qquad
\text{evict}_s[j]=1 \iff \big(c_s[j]\ge m\big)\ \wedge\ \text{尚未驱逐}
\]

驱逐是**不可逆**的：一旦在第 \(s\) 步被驱逐，后续所有步的存活掩码都为 0（真实缓存里它已经不存在了，后续分数不应该再有机会救它）。

## 任务

实现 `solve(scores, tau, patience)`：

- 输入：`scores`（float32 `(S, L)`，`S` 步、`L` 个 token 的逐步重要性分）；`tau`（阈值，标量）；`patience`（整数 \(m\ge 1\)）。
- 输出 `(evict_step, keep)`：`evict_step` 为 `(L,)` int64，首次被驱逐的步号，从未被驱逐为 `-1`；`keep` 为 `(S, L)` bool，`keep[s]` 是**第 `s` 步处理完驱逐之后**的存活掩码。
- 低分判定为**严格小于**：`score == tau` 不算低分，计数清零。
- `patience < 1` 抛 `ValueError`。

## 示例

```
scores = [[0.9, 0.1, 0.6, 0.2],
          [0.8, 0.2, 0.4, 0.3],
          [0.7, 0.3, 0.6, 0.9],
          [0.6, 0.4, 0.5, 0.8]]      τ = 0.5，m = 2

token0: 0.9,0.8,0.7,0.6  → 从不低分            → evict_step = -1
token1: 0.1,0.2,...      → 第 0、1 步连续低分    → evict_step = 1
token2: 0.6,0.4,0.6,...  → 第 1 步低分后回升     → evict_step = -1
token3: 0.2,0.3,...      → 第 0、1 步连续低分    → evict_step = 1

evict_step = [-1, 1, -1, 1]
keep[1]    = [True, False, True, False]
```

## 约束

- 驱逐后该 token 的计数不再更新、也不会被「救回」；`keep[s][j]` 在 `s >= evict_step[j]` 时恒为 0。
- `patience = 1` 时退化为普通的单帧阈值驱逐（`score < tau` 立刻驱逐）。
- 计数必须逐 token 独立：不同 token 的连续低分区间互不影响。
- 每步都要先更新计数、再判定驱逐；判定顺序不能颠倒，否则 `evict_step` 会整体偏移一步。
