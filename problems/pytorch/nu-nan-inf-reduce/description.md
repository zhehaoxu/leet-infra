# NaN 归约：忽略、计数与全 NaN 段的语义

## 题目描述

`torch.sum` 遇到一个 NaN 就把整段泼成 NaN，`torch.nanmean` 又只在整段全 NaN 时给你一条 warning 和一个 NaN。真实需求往往是第三种：掩码外的位置是 NaN（attention 里被 mask 掉的 key、稀疏 batch 里缺失的样本、多卡聚合时未落盘的 rank），要**跳过 NaN 继续算**，同时把「一个有效元素都没有」这种段单独定义清楚。本题要求手写四种跳过式归约，并让 inf 作为合法数值照常参与——很多人会把 `nanmean` 的分母误当成段长，或者以为 `nanmax` 会把 inf 也排除掉。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 记一段待归约数据为 `seg`，有效元素 `valid = seg 中不是 NaN 的元素`：
  - `sum` = \(\sum\) valid，**全 NaN 段给 0.0**（NaN 当加数被丢掉，不污染）；
  - `mean` = \(\sum\) valid / **len(valid)**，分母是有效个数而不是段长，全 NaN 段给 NaN；
  - `max` / `min` = max/min valid，全 NaN 段给 NaN。
- ±inf **不是** NaN：它照常参与。含 +inf 的段 `max` 是 +inf，同时含 ±inf 的段 `min` 是 -inf、`sum` 是 NaN（inf - inf）。
- 被归约轴长度为 0（空张量）等同于 count == 0：`sum` 给 0.0，其余给 NaN。
- `dim=None` 做整体归约返回 0-D 张量；`dim` 允许负轴，越界抛 ValueError；`mode` 只接受四种，`keepdim` 只接受 bool；非浮点输入抛 ValueError。

## 示例

```
x = [[1, nan, 3], [nan, nan, 2], [nan, nan, nan]]

dim=1  mean -> [2, 2, nan]     sum -> [4, 2, 0]    max -> [3, 2, nan]
dim=0  mean -> [1, nan, 2.5]   sum -> [1, 0, 5]    max -> [1, nan, 3]
dim=None  sum -> 6   mean -> 2   max -> 3   min -> 1
```

第一行有效元素是 1 和 3，所以 `mean` 是 \( (1+3)/2=2 \) 而不是 \(4/3\)；第三行一个有效元素都没有，`sum` 给 0.0 而 `mean`/`max` 给 NaN。列方向上第 0 列只有 1 有效、第 1 列全 NaN、第 2 列是 3 和 2，于是 `dim=0` 的 `mean` 是 `[1, nan, 2.5]`。含 inf 的一行 `[inf, nan, -inf]`：`max` 是 +inf，`min` 是 -inf，`sum` 是 NaN。

## 约束

- 支持任意维 float32/float64，输出 dtype 与输入一致
- oracle 用纯 Python 分段归约（先按 movedim 切片、过滤 NaN 再用内置 `sum/max/min`），与任何张量归约实现都不同
- 良态数据 1e-12 相对容差，示例数据要求逐位相等（含 NaN 位置必须同为 NaN）
