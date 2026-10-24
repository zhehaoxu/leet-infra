# 按轴求和归约（Reduce along Axis）

## 题目描述

归约（reduction）把某个轴「压掉」，用一个聚合函数（这里求和）沿该轴把多份数据合成一份。
理解「沿哪个轴归约、剩下哪些轴」是读写一切张量运算（softmax、注意力、BatchNorm、损失函数）
的地基。请**不使用 `torch.sum`**，靠切片累加实现沿任意轴的求和。

## 规则

- 实现 `solve(x, axis, keepdim=False)`，返回沿 `axis` 求和后的张量。
- `axis` 可为负（-1 表示最后一维）。
- `keepdim=False` 时该轴被移除；`keepdim=True` 时保留为长度 1。
- 结果须与 `torch.sum(x, dim=axis, keepdim=keepdim)` 一致。

## 示例

```
输入：x = [[1, 2, 3],
          [4, 5, 6]]
  axis=1（沿列方向、把每行压成标量）→ [6, 15]
  axis=0（沿行方向、把每列压成标量）→ [5, 7, 9]
  axis=1, keepdim=True             → [[6], [15]]
```

## 约束

- 支持任意维输入；沿轴 `axis` 求和
- 不得使用 `torch.sum / .sum() / .mean()` 等现成归约算子（靠切片逐项累加）
