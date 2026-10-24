# PyramidKV：层间预算金字塔

## 题目描述

**PyramidKV**（[arXiv:2406.02069](https://arxiv.org/abs/2406.02069)）：Transformer 不同层对 KV 缓存的需求差异极大——**低层需要更长的上下文来还原词法与句法、高层只需要少数关键 token 做定位**。于是按层号画一条"金字塔"：底层最宽、顶层最窄、中间线性递减。

kvpress 的 `PyramidKVPress.get_layer_budget` 用如下公式产生每层保留 token 数：

- $\text{max\_cp} = W + Q(1 - r)$（$W$=window_size、$Q$=q_len、$r$=compression_ratio）
- $\text{min\_num} = (\text{max\_cp} - W)/\beta$
- $\text{max\_num} = 2(\text{max\_cp} - W) - \text{min\_num}$
- 若 $\text{max\_num} \geq Q - W$：$\text{max\_num} = Q - W$；$\text{min\_num} = 2(\text{max\_cp} - W) - \text{max\_num}$
- 若 $Q \geq \text{max\_num} \geq \text{min\_num} \geq W$ **不成立**：**回退 SnapKV** 每层同一预算 $\max(1, \text{round}(Q(1-r)))$
- 否则：$\text{steps} = (\text{max\_num} - \text{min\_num})/(\text{num\_layers} - 1)$（`num_layers==1` 时 steps=0），第 $l$ 层保留 $\text{round}(\text{max\_num} - l \cdot \text{steps})$

## 规则

实现 `solve(q_len, num_layers, compression_ratio, window_size, beta)` → 返回 shape $(\text{num\_layers},)$ 的 `torch.long` tensor 每层保留数。

## 示例

```
q_len=100, num_layers=4, cr=0.5, W=4, β=2
max_cp = 4 + 50 = 54
min = (54-4)/2 = 25
max = (54-4)*2 - 25 = 75
无封顶（75 < 96）；区间合法
steps = (75 - 25) / 3 = 16.667
budgets = [round(75), round(58.333), round(41.667), round(25)] = [75, 58, 42, 25]
```

## 约束

- `beta >= 1`；`round` 使用 Python 内建（**banker's rounding**：`round(0.5)=0`、`round(1.5)=2`）
- `num_layers == 1` 时 steps=0，直接返回 `[round(max_num)]`
- 若区间不合法：每层同一预算 `max(1, round(q_len * (1 - cr)))`
