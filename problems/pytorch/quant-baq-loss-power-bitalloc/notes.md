# 量化：BAQ 损失幂律位宽分配——算法详解

## 1. 为什么用损失的幂律分配位宽

BAQ（arXiv:2506.05664）的核心观察是：量化误差对每个输入通道（列）的敏感度不同，而固定位宽把码字数平均分给了所有列。它用校准数据上每列的损失 `loss_c` 作为重要度，构建一条把「相对损失」映射到「码字数」的曲线。本题采用上游的幂律形式，指数固定为 `th = 1.95`：

\[
\text{levels}_c = \text{round}\!\left(\left(\max\!\left(\frac{loss_c}{\bar\ell}\cdot 2^{th R_{ref}},1\right)\right)^{1/th}\right)
\]

关键在于归一化因子 `2^{th R_ref}`：当 `loss_c = \bar\ell` 时，括号内恰为 `2^{th R_ref}`，开 `th` 次根后是 `2^{R_ref}`，于是 `levels = 2^{R_ref}`、`maxq = 2^{R_ref}-1`——**平均损失的列正好拿到参考位宽**。损失每降为原来的 `2^{-th}`，码字数减半、位宽减 1；指数 `th > 1` 让曲线在均值附近更平缓，不会因为损失的小幅波动就增减一位。

## 2. 位宽是连续量，不是整数

`levels_c` 是四舍五入后的**码字数**，可以是任意正整数：`levels = 5` 时 `maxq = 4`，`bits = log2(5) ≈ 2.3219`。因此 `bits` 输出是浮点数，表示「这一列实际用了几位」；真正进入量化器的是 `maxq`（最大码值），码值范围 `[0, maxq]` 一共有 `maxq + 1 = levels` 个码字。

`t_c` 在开根之前先 `max(·, 1)`：这保证 `levels >= 1`，也就是 0-bit 下界。损失远小于均值的列（或 `ratio` 很大时）会落到 `levels = 1, maxq = 0`，此时 `q` 只能取 0，整列反量化为 0。

## 3. 动态范围按行、码字数按列

这是 BAQ 源实现里容易看错的一点：`fasterquant_baq` 用 `W.max(dim=1)` / `W.min(dim=1)` 取**每行**的极值，而 `R_vec`、`maxq_scalar` 是按**列**的。所以

\[
scale_{r,c} = \frac{xmax_r-xmin_r}{maxq_c + 10^{-8}},
\qquad
zero_{r,c} = \text{round}\!\left(\frac{-xmin_r}{scale_{r,c}}\right)
\]

`xmax`/`xmin` 都向 0 夹紧，保证量化区间覆盖 0（对 ReLU 后大量非负的权重也成立）。整行为 0 时上游把 `xmin`/`xmax` 改成 `-1/+1` 避免 `diff = 0`；随后 `diff == 0` 再加 `1e-8` 的兜底在本流程里实际不会触发。`zero = round(-xmin/scale)` 把 `xmin` 映射到码值 0，`q` 再夹到 `[0, maxq_c]`——因为每列码字数不同，同一个 `scale` 矩阵的列之间不能再互相比较码值。

分母用 `maxq + 1e-8` 而不是 `maxq + 1` 是上游原样：它让 `xmax` 落在码值 `maxq` 附近（而不是 `maxq+1`），最后一级台阶略微覆盖不到 `xmax`，误差是 1e-8 量级，可忽略。

## 4. 边界行为

- `maxq = 0`：码值全 0，反量化值为 `scale · (0 - zero)`。通常 `scale` 极大、`zero ≈ 0`，结果就是全 0。
- `ratio` 放大 `loss_mean`：`t_c` 整体按 `ratio^{-1/th}` 缩小，所有列位宽等比例下降；`ratio` 减半则反向。
- 全零行：`xmin=-1, xmax=+1`，`diff=2`，但零权重仍会量化到 `zero` 对应的码值并精确还原为 0。
- 上游 `quantize_vec` 里 `q > 1e3` 时改用 `x_mean` 的兜底分支是 GPTQ 逐列补偿路径的补丁，本题不实现误差补偿，因此也不实现该分支。

## 5. 与本合集其他题的分工

- `gptq-column-quantize`：GPTQ 逐列量化与 Hessian 误差补偿，位宽固定；
- `quant-awq-clip-scale-search`：搜索裁剪系数以最小化输出误差；
- `quant-squeezellm-sensitive-kmeans`：用聚类选择敏感通道并保留高精度，仍以固定码字位置为核心；
- `quant-rptq-channel-reorder-group`：通道重排 + 分组，组内统一位宽；
- 本题：**按每个输入通道的损失分配码字数**（位宽可变、可为 0），再对每行动态范围做非对称量化——只做分配与量化，不含校准前向、Hessian 补偿与回退分支。

## 6. 验证方法

- `check_example`：手算 2×3 示例，逐字段核对 `maxq`、`zero`、`bits`、`scale` 与 `dequant`。
- `check_allocation`：位宽严格按幂律公式计算且随损失单调不减；损失等于均值时取参考位宽 `R_ref`。
- `check_row_stats`：`scale[r,c]` 必须等于该行动态范围除以该列 `maxq + 1e-8`，`zero` 由该行 `xmin` 决定。
- `check_matches_oracle`：三组不同尺度的随机输入与纯 Python oracle 全字段对齐。
- `check_quant_grid`：反量化值必须落在 scale 的整数格上，码值为整数且处于 `[0, maxq]`。
- `check_boundary_allocation`：极小损失退化为 0 bit 且整列为 0；大损失得到更多码字；`ratio` 放大 16 倍使参考位宽从 4 降到 2。
- `check_zero_rows`：全零行按 `(-1, +1)` 取范围，零权重仍反量化为 0。
- `check_validation`：形状、类型、负/全零/非有限损失、`R_ref` 与 `ratio` 非法时抛 `ValueError`。
