# 解题说明 · PyramidKV

## 预算守恒

设计目标是：所有层保留数之和 ≈ 每层同一预算时的总和（保证 compression_ratio 生效）。

$(\text{max} + \text{min}) \cdot \text{num\_layers} / 2$ 是等差数列求和；令其等于 $\text{max\_cp} - W$ 乘以层数（去掉 window 部分），得到 `max_cp = W + q_len*(1-cr)`。

## 封顶逻辑

当 `max_num` 太大接近 q_len 时，`min_num` 会被推得比 `window_size` 还小——此时把 `max_num` 截到 `q_len - window_size`，重算 `min_num`（保持总和不变）。

## 回退条件

若截断后仍不满足 `q_len >= max >= min >= window`，说明 q_len 太短——金字塔不成立，退化回 SnapKV 每层同一预算。

## round vs int

kvpress 里用 `round(x)` 而非 `int(x)`——Python 的 `round` 是 banker's rounding，与直觉的四舍五入有细微差异（对 0.5 特殊）。题目沿用 `round`。

## 与 LUKV / PerLayer 的关系

PyramidKV 只是把"每层保多少"公式化。`LUKVPress` 提供**任意 budget curve**、`PerLayerCompressionPress` 允许**每层不同 compression_ratio**——都属于同一族"层级预算分配器"，与"如何打分"（SnapKV / KNorm）正交。
