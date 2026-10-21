# Swish / SiLU

## 题目描述

实现 Swish 激活函数（PyTorch 中称 SiLU，Sigmoid Linear Unit），由 Google 在 2017 年提出，是 ReLU 的平滑替代，被 NAS 自动搜索确认为优于 ReLU 的通用激活：

$$
\text{swish}(x) = x \cdot \sigma(x)
$$

其中 $\sigma$ 是 Sigmoid。与 ReLU 相比它在负区间有一个小幅下凹（最小值约 $-0.278$ 出现在 $x \approx -1.28$），处处光滑可导。

## 规则

- 实现 `solve(x: torch.Tensor) -> torch.Tensor`，返回逐元素 $x \cdot \sigma(x)$。
- 可以调用 `torch.sigmoid` 作为原语；整体必须向量化。
- 输出形状、dtype 与输入一致。
- 大负数（如 $x = -1000$）时结果应趋近 0 且无 `NaN`（注意 `sigmoid` 稳定时 $x\cdot\sigma$ 自然稳定）。

## 示例

```
输入：[-3.0, -1.0, 0.0, 1.0, 3.0]
输出：[-0.14228, -0.26894, 0.0, 0.73106, 2.85772]
```

## 约束

- 输入为 float32 或 float64 张量，任意形状
