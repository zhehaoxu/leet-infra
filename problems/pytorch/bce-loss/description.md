# 二元交叉熵损失（BCE Loss）

## 题目描述

二分类（或多标签）损失：输入是已经过 sigmoid 的**概率** $p\in(0,1)$，标签 $y\in\{0,1\}$。它衡量单个伯努利分布的预测与真值的差异，是多标签分类与 GAN 判别器的标配。

## 规则

- 实现 `solve(pred, target, reduction="mean")`，逐元素：
  $$ \mathcal L_i = -\big[\,y_i\log p_i + (1-y_i)\log(1-p_i)\,\big] $$
- `pred` 是概率（不是 logits），与 `target` 同形；`target` 为 0/1（可为 float）。
- 必须对 $p$ 做**截断**（clamp 到开区间），避免 $p=0/1$ 时 $\log 0=-\infty$。
- `reduction` 取 `"mean"` / `"sum"` / `"none"`；未知值抛 `ValueError`。
- 对内部概率（远离 0/1）结果须与 `F.binary_cross_entropy(pred, target, reduction=...)` 一致。

## 示例

```
输入：pred = [0.9, 0.2, 0.75, 0.4]
      target = [1.0, 0.0, 1.0, 0.0]
reduction="none" → [0.10536, 0.22314, 0.28768, 0.51083]
reduction="mean" → 0.28175
（y=1 只留 -log p、y=0 只留 -log(1-p)；预测越接近真值标签，损失越小）
```

## 约束

- `pred` 支持 float32 / float64；输出 dtype 与 `pred` 一致
- 不使用 `F.binary_cross_entropy`；需手写公式并做 log 截断
