# 量化：PB-LLM 部分二值化

## 题目描述

把所有权重都压到 1～2 bit 会让**显著权重**的误差失控：LLM 权重里少数大幅度值对输出影响很大，而多数小幅度值并不敏感。PB-LLM（Partially Binarized LLMs, 2023, arXiv:2310.00034）的做法是**只把小权重二值化**，大权重仍用较高位宽表示：

1. 按 `|W|` 选出幅度最小的一部分元素，标记为 `mask`；
2. `mask` 内的元素走 1-bit 的二值（xnor）通路；
3. 其余元素走 `high_bits` 位的非对称量化；
4. 两条通路的重建值按 `mask` 逐元素拼起来。

本题固定 `groupsize=-1`（整张权重作为一个 low 组），且只实现 RTN 分支，不做 GPTQ 的 Hessian 误差补偿。

**掩码**：把 `|W|` 全局展平后升序排序，取第 `int(W.numel() * low_frac)` 个值为阈值 `thresh`，`mask = |W| <= thresh`。相等值会一起入选，因此实际比例可能略大于 `low_frac`。

**low 通路（xnor）**：先 `masked = W * mask`（未入选位置置 0，**零也计入统计**），逐行计算

- `low_mean = masked.mean(dim=1)`
- `low_scale = |masked - low_mean|.mean(dim=1)`

重建为 `sign(w - low_mean) * low_scale + low_mean`（`torch.sign(0) == 0`）。

**high 通路（逐行非对称）**：`xmin=min(row.min(), 0)`、`xmax=max(row.max(), 0)`，二者同时为 0 时置为 `(-1, +1)`；`maxq=2^high_bits-1`；`high_scale=(xmax-xmin)/maxq`；`high_zero=round(-xmin/high_scale)`。重建：

`q = clamp(round(w / high_scale) + high_zero, 0, maxq)`，`w_hat = (q - high_zero) * high_scale`。

## 任务

实现 `solve(W, low_frac, high_bits)`：

- `W`：float32 `(O, C)` 权重，非空且有限；
- `low_frac`：`0 <= low_frac < 1` 的数值，低精度比例；
- `high_bits`：`2..8` 的整数，高位通路的位宽。
- 返回 `dict`：`mask` bool `(O, C)`；`low_mean`、`low_scale`、`high_scale` float32 `(O,)`；`high_zero` int64 `(O,)`；`dequant` float32 `(O, C)`。

**约定**：内部用 float64，输出 float32，容差 `1e-6`。非法输入（`high_bits` 越界、`low_frac` 不在 `[0,1)`、`W` 非二维浮点/为空/含非有限值）抛 `ValueError`。

## 示例

```
W = [[1.0, -0.5, 0.25, -0.25, 0.1, -0.1]], low_frac = 0.5, high_bits = 4
|W| 排序 = [0.1, 0.1, 0.25, 0.25, 0.5, 1.0]，索引 int(6*0.5)=3 -> thresh = 0.25
mask = [False, False, True, True, True, True]
low: masked = [0, 0, 0.25, -0.25, 0.1, -0.1] -> low_mean = 0, low_scale = 0.7/6
high: xmin=-0.5, xmax=1.0 -> high_scale = 1.5/15 = 0.1, high_zero = round(0.5/0.1) = 5
dequant = [1.0, -0.5, 0.116667, -0.116667, 0.116667, -0.116667]
```

## 约束

- 掩码阈值必须在**整张 `W`** 上取（本题 `groupsize=-1`），并按 `int(numel * low_frac)` 取排序下标。
- low 的 `mean/scale` 必须基于 `W * mask`（未入选位置按 0 参与均值），不能只统计入选元素。
- high 的 `maxq` 是 `2^bits - 1`（全 1 的无符号码），`zero` 由 `-xmin/scale` 四舍五入到偶数。
- 最终重建必须逐元素按 `mask` 选择，不能把两条通路混合相加。
