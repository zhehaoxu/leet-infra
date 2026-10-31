# PRM best-of-N 聚合账：步级打分的四种归并口径

## 题目描述

过程奖励模型（Process Reward Model, PRM，Lightman et al. 2023, arXiv:2305.20050 "Let's Verify
Step by Step"）给一条候选解答的**每一步**打分，而不是只给终答打分。best-of-N 场景下要拿这串
步级分数比较 N 条候选，就必须选一个**归并口径**（reduction rule）把变长序列压成标量——
口径选什么，排序就信什么：

- `last` 只看最后一步：局部、易被单步逆袭；
- `max` 看最好的一步：乐观，"高光"即入选；
- `min` 看最差的一步：悲观，短板即淘汰；
- `product` 全步连乘：任何一步趋近 0 都压垮整条链——这正是论文偏好 product 的机理：
  长链里隐藏一处失败步骤的代价被指数级放大。

> 为保证判题**确定性**，步级分数直接以数值列表传入；排序的并列规则钉死为「聚合分降序、
> 并列取 index 较小者」。

## 规则

- 实现 `solve(step_scores, answer_scores=None, rule="product", top_m=1)`：
  - `step_scores: list[list[float]]`，N 条候选各自的步级分数序列（长度可不同，每条至少 1 步）；
  - `answer_scores: None 或 list[float]`，与候选等长的终答打分；提供时
    `agg[i] = 步级聚合[i] × answer_scores[i]`；
  - `rule`：`"last" | "max" | "min" | "product"` 四选一；
  - `top_m: int ≥ 1`。
- 聚合后按 `(-agg[i], i)` 排序：
  - `best`：聚合分最大的候选 index（并列取较小）；
  - `top`：前 `top_m` 个 index（降序，并列取较小）；
  - `tie`：最高分是否并列（`agg.count(max) > 1` 严格计数）。
- 账本同时返回 `agg`（每条的聚合分）、`n_steps`（每条步数）、`min_len`（最短候选步数）。
- 校验（`ValueError`）：`rule` 不在四种之列；`top_m` 非正整数（含 bool）；`step_scores` 为空；
  某条候选无步；`answer_scores` 长度不匹配。

## 示例

```
输入：step_scores = [[0.9, 0.5, 0.8], [0.7, 0.9], [1.0, 0.1, 0.1, 0.1]], rule = "product"
  product: [0.36, 0.63, 0.001]
  best = 1（0.63 > 0.36 > 0.001），top(3) = [1, 0, 2]
  注意 max 口径下 best = 2（有 1.0 的高光步），last 口径下 best = 1
输出：agg = [0.36, 0.63, 0.001], best = 1, top = [1, 0, 2], tie = false
```

## 约束

- 纯 Python 即可（torch 无关）；分数为 `[0, 1]` 区间的浮点
- `top_m` 可大于 N（此时 `top` 就是全部候选的排序）；`bool` 是 `int` 的子类，必须显式排除
- 四种口径都必须实现并与论文同名；未知 `rule` 必须在聚合**之前**报错
