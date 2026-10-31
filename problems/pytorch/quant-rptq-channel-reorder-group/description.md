# 量化：RPTQ 通道重排与分组量化

## 题目描述

相邻通道的激活分布常常差异巨大。若按通道顺序直接分组，组内一个离群通道就会把该组的量化范围撑大，其余通道只能分到很少的整数格点。RPTQ（Yuan et al., ICLR 2023, arXiv:2304.01089）的做法是：先按**通道的量化跨度**重排通道，让跨度接近的通道进入同一组，再做组内量化。

给定逐通道激活上下界 `xmax/xmin` 与权重 `W`，本题要求：

1. 计算跨度 `span_i = xmax_i - xmin_i`，按 `(span_i, 通道号)` 升序**稳定**排序得到 `perm`；
2. 把 `perm` 顺序的 `C` 个通道**均衡**切成 `num_groups` 段：前 `C % num_groups` 段各多 1 个，段长写入 `counts`；
3. 每段取段内 `max/min`，按下式得到该组 `scale` 与 `zero`：
   - `scale = clamp((gmax - gmin) * 2^-bits, 1e-4, 1e4)`
   - `zero = clamp(round((gmax + gmin) * -0.5 / scale), -1e4, 1e4)`
4. 对 `W` 逐元素 fake-quant：`q = clamp(round(w / scale) + zero, -2^(bits-1), 2^(bits-1)-1) - zero`，重建 `w_hat = q * scale`，再按 `perm` 写回原通道顺序。

注意整数码被截在**对称区间** `[-2^(b-1), 2^(b-1)-1]` 上，正侧只有 `2^(b-1)-1` 个正码，因此组内正向最大值通常会先饱和。

## 任务

实现 `solve(xmax, xmin, W, num_groups, bits)`：

- `xmax`, `xmin`：float32 `(C,)`，逐通道激活上界/下界，要求 `xmin <= xmax`；
- `W`：float32 `(O, C)` 权重；
- `num_groups`：`1..C` 的整数；`bits`：`2..8` 的整数。
- 返回 `dict`：`perm` int64 `(C,)`、`counts` int64 `(G,)`、`scales` float32 `(G,)`、`zero_points` int64 `(G,)`、`dequant` float32 `(O, C)`。

**约定**：内部用 float64 计算，输出按上述 dtype；容差 `1e-5`。非法输入（类型错误、形状不符、`num_groups/bits` 越界、存在非有限值、`xmin > xmax`）抛 `ValueError`。

## 示例

```
xmax = [8.0, 0.5, 8.4, 0.6], xmin = [-8.0, -0.5, -8.4, -0.6]
W = [[8.0, 0.5, 8.4, 0.6], [-8.0, -0.5, -8.4, -0.6]]
num_groups = 2, bits = 4

span = [16.0, 1.0, 16.8, 1.2] -> perm = [1, 3, 0, 2], counts = [2, 2]
group 0: channels {1, 3}, gmax=0.6, gmin=-0.6 -> scale=0.075, zero=0
group 1: channels {0, 2}, gmax=8.4, gmin=-8.4 -> scale=1.05, zero=0
dequant = [[7.35, 0.525, 7.35, 0.525], [-8.4, -0.525, -8.4, -0.6]]
```

## 约束

- 排序必须是稳定排序：跨度相同时按通道号升序；不得依赖浮点不稳定的聚类。
- 分段必须均衡，余数落在前 `C % num_groups` 段（上游 PEG 变体直接丢弃余数，本题修正为不丢通道）。
- 每个组只能使用自己段内的 `max/min`，不能复用全局统计量。
- 重建结果必须按 `perm` 写回原通道顺序。
