# 量化：BAQ 损失幂律位宽分配

## 题目描述

固定位宽量化给所有权重列同样的码字数，但校准数据上的损失（敏感度）在通道之间差异很大：敏感通道多分几位、不敏感通道少分几位，能在平均位宽不变时显著降低误差。BAQ（arXiv:2506.05664）的做法是：按每列的损失，用一条**幂律**把它映射到码字数，再对权重做非对称量化。

对损失向量 `loss`（长度为列数 `C`）、参考位宽 `R_ref`（整数）与均值缩放系数 `ratio`（正数），固定指数 `th = 1.95`：

\[
\bar{\ell} = \text{ratio}\cdot\text{mean}(\text{loss}),\qquad
t_c = \max\!\left(\frac{\text{loss}_c}{\bar{\ell}}\cdot 2^{th\cdot R_{ref}},\ 1\right)
\]

\[
\text{levels}_c = \text{round}\!\left(t_c^{1/th}\right),\qquad
\text{maxq}_c = \text{levels}_c - 1,\qquad
\text{bits}_c = \log_2(\text{levels}_c)
\]

- 当 `loss_c` 恰好等于均值时，`levels_c = 2^R_ref`，该列正好取参考位宽；
- `levels_c` 不必是 2 的幂（例如 5 → `maxq = 4`、`bits ≈ 2.3219`），它直接就是该列可用的码字数；
- `max()` 保证 `levels_c >= 1`：损失远小于均值的列会退化成 `maxq = 0`（0 bit）。

随后逐元素量化。动态范围按**行**（输出通道）取，码字数按**列**取：

\[
xmax_r = \max(\max_j W_{r,j}, 0),\qquad
xmin_r = \min(\min_j W_{r,j}, 0)
\]

整行为 0 时取 `xmin = -1, xmax = +1`。于是

\[
scale_{r,c} = \frac{xmax_r - xmin_r}{maxq_c + 10^{-8}},\qquad
zero_{r,c} = \text{round}\!\left(\frac{-xmin_r}{scale_{r,c}}\right)
\]

\[
q_{r,c} = \text{clamp}\!\left(\text{round}\!\left(\frac{W_{r,c}}{scale_{r,c}}\right) + zero_{r,c},\ 0,\ maxq_c\right),\qquad
\hat{W}_{r,c} = scale_{r,c}\cdot(q_{r,c}-zero_{r,c})
\]

## 任务

实现 `solve(W, loss, R_ref, ratio)`：

- `W`：float32 `(O, C)`，非空且有限；
- `loss`：float32 `(C,)`，非负且有限，均值大于 0；
- `R_ref`：`int`，取值 `1..8`；
- `ratio`：`int`/`float`，有限且大于 0；
- 返回 `dict`：
  - `bits` float32 `(C,)`：每列位宽 `log2(levels)`；
  - `maxq` int64 `(C,)`：每列最大码值 `levels - 1`；
  - `scale` float32 `(O, C)`：逐元素 scale；
  - `zero` int64 `(O, C)`：逐元素零点；
  - `dequant` float32 `(O, C)`：反量化结果。

**约定**：内部用 float64，输出 float32，容差 `1e-6`；`round` 均为「四舍五入到偶数」（`torch.round`）。scale 的分母是 `maxq + 1e-8`（上游写法，不是 `maxq + 1`）。`W`/`loss` 形状或类型非法、`loss` 含负值或非有限值或均值为 0、`R_ref` 不是 `1..8` 的整数、`ratio` 非有限或非正时抛 `ValueError`。

## 示例

```
W = [[0.8, 0.6, 0.9],
     [0.2, 0.6, 0.3]]
loss = [0.3, 1.0, 1.7], R_ref = 2, ratio = 1.0

loss_mean = 1.0, 2^(th*R_ref) = 2^3.9 = 14.9285
t = [4.4786, 14.9285, 25.3785]
t^(1/1.95) = [2.157, 4.000, 5.251] -> levels = [2, 4, 5]
maxq = [1, 3, 4], bits = [1, 2, log2(5) = 2.3219]

row 0: xmin = 0, xmax = 0.9, diff = 0.9 -> scale = [0.9, 0.3, 0.225]
row 1: xmin = 0, xmax = 0.6, diff = 0.6 -> scale = [0.6, 0.2, 0.15]
zero = [[0, 0, 0], [0, 0, 0]]

q       = [[1, 2, 4], [0, 3, 2]]
dequant = [[0.9, 0.6, 0.9], [0.0, 0.6, 0.3]]
```

## 约束

- 位宽只由列损失决定，同列的所有行共享同一个 `maxq`；动态范围按行取，因此 `scale` 是 `(O, C)` 的逐元素矩阵。
- `t` 必须先 clamp 到 `>= 1` 再开 `th` 次根；不能让 `levels` 取到 0。
- `maxq = 0` 的列，码值全为 0，该列反量化结果全为 0（`ratio` 很大或损失极小时会出现）。
- 所有取整（`levels`、`zero`、`q`）都用 half-to-even；上游的 GPTQ 误差补偿、`q > 1e3` 回退分支不在本题范围。
