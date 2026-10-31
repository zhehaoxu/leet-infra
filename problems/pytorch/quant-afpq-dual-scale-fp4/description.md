# 量化：AFPQ 双侧浮点 scale 量化

## 题目描述

FP4/FP3 的量化码表不是等距的：小数值附近格点密，大数值附近格点疏——这正是浮点表示的优势。但如果正负两侧共享同一个 scale，权重分布不对称时，小的一侧会白白浪费码点。

AFPQ（Asymmetric Floating Point Quantization for LLMs, 2023, arXiv:2311.01792）给**正负两侧各一个 scale**：

- `scale_pos = xmax / (maxq/2)`，只由正侧极值决定；
- `scale_neg = |xmin| / (maxq/2)`，只由负侧极值决定。

正部与负部各自除以自己的 scale 后，在码表上取最近的码，最后把两部分重建相加。这样正负极值悬殊时，小的一侧仍能落在密集的码点上。

本题使用源码中的两张码表（`bits=4` 时 `maxq=12`，`bits=3` 时 `maxq=8`）：

- 4-bit：`[-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]`
- 3-bit：`[-4, -2, -1, -0, 0, 1, 2, 4]`

码表里同时有 `-0` 与 `0`，二者数值相同；取最近码时若距离相同，取码表中**更靠前**的那个（即 `argmin` 的首个匹配）。

## 任务

实现 `solve(W, bits)`：

- `W`：float32 `(G, N)`，逐行（输出通道）独立量化；
- `bits`：只能是 `3` 或 `4`。
- 每行：
  1. `xmin = min(row.min(), 0)`，`xmax = max(row.max(), 0)`；若二者都为 0，令 `xmin=-1`、`xmax=+1`；
  2. `scale_pos = xmax / (maxq/2)`，`scale_neg = |xmin| / (maxq/2)`；
  3. 把元素拆成 `x_pos = max(w, 0)` 与 `x_neg = min(w, 0)`，分别除以 `scale_pos/scale_neg`，在码表上取最近码得到 `q_pos/q_neg`；
  4. `dequant = q_pos * scale_pos + q_neg * scale_neg`。
- 返回 `dict`：`scale_pos`、`scale_neg` float32 `(G,)`，`q_pos`、`q_neg`、`dequant` float32 `(G, N)`。

**约定**：内部用 float64，输出 float32，容差 `1e-6`。整行没有正数（或没有负数）时，该侧 `scale=0`，对应 `q` 与重建贡献都取 0，**不得做 0/0 除法**。`bits` 非 3/4、`W` 非二维浮点张量、空张量、含非有限值时抛 `ValueError`。

## 示例

```
W = [[3.0, -0.3, 0.2]], bits = 4, maxq = 12, half = 6
xmax=3.0, xmin=-0.3 -> scale_pos=0.5, scale_neg=0.05
q_pos = [6, 0, 0.5]     (0.2/0.5=0.4，最近码 0.5)
q_neg = [0, -6, 0]      (-0.3/0.05=-6，精确落在码点)
dequant = [3.0, -0.3, 0.25]
```

对比单 scale：若正负共用一个 scale `3.0/6=0.5`，则 `-0.3/0.5=-0.6` 只能取到 `-0.5`，重建 `-0.25`，误差 `0.05`；双侧 scale 把 `-0.3` 精确表示。

## 约束

- 码表必须与给定表完全一致，`q_pos/q_neg` 只能取表中的值。
- 距离相同时取码表中更靠前的码，不能取更靠后的码。
- 单侧缺失（全正/全负行）按零贡献处理；全零行按 `(-1, +1)` 退化。
- 内部 float64 计算，避免 float32 除法在最近码边界上翻转。
