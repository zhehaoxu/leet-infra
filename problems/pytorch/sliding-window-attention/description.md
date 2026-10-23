# 滑动窗口注意力（Sliding-Window / Local Attention）

## 题目描述

标准因果注意力里，位置 $i$ 能回看**全部**历史 token；滑动窗口注意力把回看范围限制为最近
`window` 个 token，把注意力复杂度从 $O(L^2)$ 降到 $O(L\cdot\text{window})$，是 Mistral、
Longformer 等处理长序列的关键手段。

## 规则

- 实现 `solve(q, k, v, window)`，单头：`q`、`k`、`v` 形状均为 `(B, L, d)`，返回 `(B, L, d)`。
- 缩放为 $1/\sqrt{d}$。
- 掩码同时满足**因果**与**窗口**：位置 $i$ 只能关注
  $\max(0,\, i-\text{window}+1) \le j \le i$ 的 key/value（即 $j \le i$ **且** $i-j < \text{window}$）。
- 对角线恒可见（每个 token 至少能看到自己），因此不会出现整行 $-\infty$ 的退化。
- 当 `window >= L` 时应退化为纯因果注意力。

## 示例

```
输入：q=k=v=[[[1,0],[0,1],[1,1]]]，window=2
输出：[[[1.000000, 0.000000],
        [0.330238, 0.669762],
        [0.669762, 1.000000]]]
（第 0 个 token 只看自己，输出=v0）
```

## 约束

- float32 输入；$B \ge 1$，$L, d \ge 1$，`window >= 1`
- $q, k, v$ 序列长度相等（$L_k = L_q = L$）
