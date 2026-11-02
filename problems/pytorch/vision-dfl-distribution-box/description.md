# DFL 分布式框回归账

回归一个距离值时，直接回归标量会把标注噪声当成真值。DFL 改成回归一个**分布**：把每条边的
距离离散成 `reg_max+1` 个 bin，网络输出 logit，解码时取期望，训练时用「左右两个相邻 bin 的
交叉熵加权和」监督——权重由目标值在 bin 内的位置决定。

## 接口

```python
def solve(logits, targets=None, stride=1.0, reg_max=7, weights=None) -> dict
```

`logits` 形状 `(N, S, reg_max+1)`（`S` 是边数，检测里通常为 4：l/t/r/b）。返回：

| 字段 | 含义 |
| --- | --- |
| `probs` | `(N, S, reg_max+1)`：每个 bin 的 softmax 概率 |
| `dist` | `(N, S)`：期望距离 `Σ i·p_i × stride` |
| `bin_index` | 长度为 `reg_max+1` 的下标张量 |
| `left` / `right` | 目标的左右 bin 下标 |
| `w_left` / `w_right` | 两条 CE 的权重，恒和为 1 |
| `loss` / `loss_mean` | 逐元素损失与均值 |

## 输入输出

```
输入：logits=全 0（1 条边、8 个 bin），targets=[[3.25]]
输出：w_left=[[0.75]], w_right=[[0.25]], left=[[3]], right=[[4]],
      loss=[[log 8]], dist=[[3.5]]
```

均匀分布下期望是 `0..7` 的中点 `3.5`；目标 `3.25` 落在 bin 3 与 4 之间，权重 `0.75 / 0.25`，
两条 CE 都是 `log 8`，凸组合仍是 `log 8`。

## 规则

- **解码是下标期望**：`dist = Σ_{i=0}^{reg_max} i · softmax(logits)_i × stride`。bin 下标从
  **0** 起，所以期望最大只能到 `reg_max × stride`，不是 `(reg_max+1) × stride`。
- **损失是两条 CE 的凸组合**：`left = int(target)`（向零截断）、`right = left + 1`、
  `w_left = right - target`、`w_right = target - left`，
  `loss = w_left · CE(left) + w_right · CE(right)`。不是「把 logit 插值后算一次 CE」，
  也不是「用一个软标签做一次 CE」。
- **整数目标退化为单个 CE**：`target` 为整数 `k` 时 `w_left = 1, w_right = 0`，
  右侧那一项权重为 0。
- **目标范围 `[0, reg_max-1]`**：`right` 必须可索引，所以 `target = reg_max` 非法（官方实现
  会在这里越界），负目标同样拒绝。
- **归一化分母是元素数**：`loss_mean = loss.sum() / loss.numel()`，即 `N × S`，不是正样本数。
- **`weights`**：给定时逐元素乘到 `loss` 上（乘在加权 CE 之后）。
- **数值稳定**：logit 加到 `1e4` 仍要给出均匀分布，softmax 必须减最大值。
- **校验**：`reg_max` 为正整数（`bool` 拒绝）、`logits` 为 3 维且最后一维等于 `reg_max+1`、
  `stride` 为有限正数、`targets`/`weights` 形状与 `dist` 一致。

## 提示

- `targets` 用 `.long()` 得到的是**向零截断**，正目标的截断等于 floor；这和 `round` 不同。
- 只给 `logits` 时是纯解码，账本里不应出现 `loss` 相关键。
