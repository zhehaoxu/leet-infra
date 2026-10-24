# 交叉熵损失（Cross-Entropy Loss）

## 题目描述

多分类的核心损失：模型输出未归一化的 **logits**，标签是整数类别索引。交叉熵 = 真实分布（独热）与预测分布的 KL + 熵，等价于「真实类对数的负概率」$-\log p_{\text{true}}$。数值稳定须把 softmax 与 log 合并成 log-softmax。

## 规则

- 实现 `solve(logits, target, reduction="mean")`：
  - `logits` 形状 $(N, C)$（$N$ 个样本、$C$ 个类别）；
  - `target` 形状 $(N,)$，是该样本的真实类索引 $[0,C)$；
  - 单样本损失 $= -\log\text{softmax}(\text{logits})[\,\text{target}\,]$。
- 必须**数值稳定**：先减每行最大值再做 $\exp$/$\log$，不能直接 `log(softmax(x))`。
- `reduction` 取 `"mean"`（对 $N$ 个样本平均）/ `"sum"` / `"none"`；未知值抛 `ValueError`。
- 结果须与 `F.cross_entropy(logits, target, reduction=...)` 一致。

## 示例

```
输入：logits = [[2.0, 1.0, 0.1],
                [0.1, 2.0, 1.0],
                [1.0, 0.5, 3.0]]
      target = [0, 1, 2]
reduction="none" → [0.41703, 0.41703, 0.19673]
reduction="mean" → 0.34360
（即使 logits 大到上千，输出仍有限：稳定式的作用）
```

## 约束

- `logits` 支持 float32 / float64；`target` 为整型索引
- 不使用 `F.cross_entropy` / `F.nll_loss`，需手写稳定的 log-softmax + gather
