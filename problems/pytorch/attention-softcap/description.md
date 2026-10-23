# 注意力 Soft-Capping（Gemma 式 logits 软化）

## 题目描述

当注意力 logits 极大时，$\text{softmax}$ 会塌缩成近似 one-hot，梯度消失、训练不稳。
Gemma 2 引入 **soft-capping**：在 softmax 之前先把 logits 用一个平滑的 `tanh` 压进
有界区间 $(-cap, cap)$，既限制极端值、又保留可导的软阈值。

$$ \text{logits}' = cap \cdot \tanh\!\left(\frac{\text{logits}}{cap}\right),\qquad
p = \text{softmax}(\text{logits}') $$

## 规则

- 实现 `solve(scores, cap)`，`scores` 为任意形状、在**最后一维**上做 softmax；`cap` 为正浮点数。
- 先做 $cap \cdot \tanh(scores/cap)$，再对最后一维做 softmax，返回归一化概率（每行和为 1）。
- `cap` 很大时 $\tanh(x/cap)\approx x/cap$，结果趋近直接对 `scores` 做 softmax。

## 示例

```
输入：scores=[[1.0, 2.0, 3.0]]，cap=1.0
     （capped = tanh([1,2,3]) = [0.761594, 0.964028, 0.995055]）
输出：[[0.286751, 0.351092, 0.362156]]
```

## 约束

- float32 输入；最后一维长度 $\ge 1$；`cap > 0`
- `scores` 可能含极端大/小值（如 $\pm 10^6$），实现须保证 `tanh` 后不出现 NaN/Inf
