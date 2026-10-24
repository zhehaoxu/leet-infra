# Roofline 算术强度与性能上限

## 题目描述

Roofline 模型用一个指标判断算子是**算力瓶颈（compute-bound）**还是**带宽瓶颈（memory-bound）**，
并给出在该硬件上能达到的性能上限。它是 `operator-fusion`（融合提升算术强度）背后的理论工具。

## 规则

- 实现 `solve(flops, nbytes, peak_flops, bandwidth)`，返回四元组
  `(arithmetic_intensity, ridge_point, attainable_flops, is_memory_bound)`。
- **算术强度（Arithmetic Intensity）**：每搬一字节做多少次浮点运算
  $$ AI = \frac{\text{flops}}{\text{nbytes}} \quad [\text{flop/byte}] $$
- **屋脊点（ridge point）**：硬件从 memory-bound 切换到 compute-bound 的临界算术强度
  $$ \text{ridge} = \frac{\text{peak\_flops}}{\text{bandwidth}} $$
- **可达性能（Roofline 上包络）**：
  $$ \text{attainable} = \min\!\left(\text{peak\_flops},\ AI \cdot \text{bandwidth}\right) $$
- **瓶颈判定**：$AI < \text{ridge}$ 为 memory-bound（`is_memory_bound=True`，此时
  $\text{attainable}=AI\cdot\text{bandwidth}<\text{peak}$）；否则 compute-bound
  （$AI \ge \text{ridge}$，$\text{attainable}=\text{peak\_flops}$，取等号归 compute-bound）。

## 示例

```
输入：flops=1e9, nbytes=1e7, peak_flops=2e14, bandwidth=1e12
  AI      = 1e9 / 1e7        = 100.0 flop/byte
  ridge   = 2e14 / 1e12       = 200.0 flop/byte
  100 < 200 → memory-bound，attainable = 100 * 1e12 = 1e14
输出：(100.0, 200.0, 100000000000000.0, True)
```

## 约束

- `flops, nbytes, peak_flops, bandwidth` 均为正数
- 返回四元组：前三项为 float，`is_memory_bound` 为 bool
- 判题对 float 用相对容差 `1e-6`；`AI == ridge` 的边界按 compute-bound 处理
