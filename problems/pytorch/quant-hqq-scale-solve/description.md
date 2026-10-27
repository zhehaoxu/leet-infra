# 量化：HQQ 半二次闭式 scale 求解

## 题目描述

RTN（round-to-nearest）先按 `absmax` 定死 scale，再用它把权重四舍五入——scale 是**先验**选出来的，round 的量化误差没有被优化。**HQQ**（Badri & Shaji, 2024, arXiv:2403.07134）把量化写成一个**交替优化**问题：

\[
\min_{s,\ z,\ q}\ \|W-s\,(q-z)\|_2^2
\quad\text{s.t.}\quad q=\operatorname{round}(W/s)+z,\ \ z\in\{0,\dots,2^b-1\}
\]

固定 \(q\) 与 \(z\)（整数）时，对 \(s\) 的最优解是**闭式**的（最小二乘 / Wiener 解）：

\[
s^\star=\frac{\langle W,\ q-z\rangle}{\|q-z\|_2^2}
\]

于是每组权重只需几次「round → 闭式更新 scale」的迭代，就能得到比 RTN 更小的重构误差。zero-point \(z\) 由组内最小值对齐得到（\(z=\operatorname{clamp}(\operatorname{round}(-W_{\min}/s),0,2^b-1)\)），本题不搜索 \(z\)。

## 任务

实现 `solve(W, bits, iters)`：

- 输入：`W` 为 float32 `(G, N)`（每行一组）、`bits` 为 2～8 的整数、`iters` 为非负整数（交替迭代次数）。
- 对每组（group）独立执行：
  1. `lo, hi = W.min(), W.max()`；若 `hi == lo`（常量组）：`scale = 1.0`、`zero = 0`、重建值全为 `lo`，跳过迭代；
  2. 初始 `s = (hi - lo) / (2^bits - 1)`，`z = clamp(round(-lo / s), 0, 2^bits-1)`；
  3. 重复 `iters` 次：`q = clamp(round(W/s) + z, 0, 2^bits-1) - z`，再取闭式 `s = <W,q> / <q,q>`（若 `<q,q> == 0` 保持 `s` 不变）；
  4. 用最终的 `s` 重算一次 `q`，重建 `deq = q * s`。
- 输出 `(scale, zero, deq)`：`scale` float32 `(G,)`；`zero` int64 `(G,)`；`deq` float32 `(G, N)`。

**约定**：内部 float64，输出 float32；容差 \(10^{-5}\)。`bits` 越界或 `iters < 0` 抛 `ValueError`。

## 示例

```
W = [[0, 1, 2, ..., 15]], bits = 4, iters = 2
lo=0, hi=15 → s=1, z=0；q = W 精确（整数格点）；s = <W,W>/<W,W> = 1
输出：scale=[1.0], zero=[0], deq=[[0, 1, 2, ..., 15]]
```

## 约束

- 闭式 scale 必须用 \(\langle W,q\rangle/\langle q,q\rangle\)，不能退回 `(hi-lo)/(2^b-1)`。
- `z` 一旦由初值确定就固定不动（HQQ 的简化版：不搜索 zero-point）。
- 常量组按上述退化规则处理（`scale=1, zero=0, deq=lo`）。
