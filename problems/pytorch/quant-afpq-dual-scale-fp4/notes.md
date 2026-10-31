# 量化：AFPQ 双侧浮点 scale 量化——算法详解

## 1. 单 scale 的盲点

FP4/FP3 的码表在零附近密集、在极值附近稀疏，本来就是为了匹配权重的钟形分布。但只要正负两侧共用一个 scale，这个优势就会被不对称分布吃掉：假设正侧极值是 3.0、负侧极值只有 -0.3，共用 `scale=3.0/6=0.5` 时，负侧所有值都挤在 `-0.5` 附近，`-0.3` 只能重建为 `-0.25`。

AFPQ（Asymmetric Floating Point Quantization for LLMs, 2023, arXiv:2311.01792）的改动非常直接：正负两侧各用**自己的极值**算 scale。上游实现中 `sym=False`、`two_scale=True`，`find_params` 先做 `xmin=min(min,0)`、`xmax=max(max,0)`，再分别除以 `maxq/2`；`quantize_2scale` 把 `x` 拆成正部与负部，各自在码表上取最近码，`xq = q_pos*scale_pos + q_neg*scale_neg`。

## 2. 码表与 scale 的对应

码表的最大码恰好是 `maxq/2`：

- 4-bit：`maxq=12`，`half=6`，码表 `[-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]`；
- 3-bit：`maxq=8`，`half=4`，码表 `[-4, -2, -1, -0, 0, 1, 2, 4]`。

因此 `scale_pos=xmax/(maxq/2)` 会把正侧极值精确映射到最大正码 `maxq/2`；负侧同理映射到最负码。除以 `maxq/2` 而不是别的常数，是为了让极值不溢出码表。

码表里的所有数（含 `±0.5`、`±1.5`）都是二进制精确值，float32 可以无损表示，所以 `q_pos/q_neg` 的 float32 输出与 float64 中间结果逐位一致，判题可以精确比较。

## 3. 拆分正负部的必要性

直接对 `x/scale` 取最近码是不行的：两侧 scale 不同，必须先拆成 `x_pos=max(x,0)` 与 `x_neg=min(x,0)`，各自除以自己的 scale，再分别取码。重建时两部分相加；因为其中一个必为 0，加法不会重复计入。

本行没有正数时 `xmax` 被 `max(...,0)` 压成 0，`scale_pos=0`；上游此时会出现 `0/0=NaN` 的未定义路径。本题明确规定：**scale 为 0 的一侧，码与贡献都取 0**，避免 NaN 传播，也让单侧行（全正或全负）有确定结果。

## 4. 平局、零与退化

- 码表同时包含 `-0`、`0`，数值相等；`argmin` 取首个匹配，因此零附近的实际码是 `-0`，重建仍为 0。
- 其他平局（例如 `q=1.75` 到 `1.5` 与 `2` 等距）同样取码表中更靠前的 `1.5`。本题把「首个匹配」写进约定，避免实现之间漂移。
- 全零行：`xmin=xmax=0`，按源码规则置为 `(-1,+1)`，两侧 scale 都是 `1/(maxq/2)`，重建为 0。
- `bits ∈ {3,4}`；`W` 必须二维、非空、有限；内部 float64，输出 float32。

## 5. 与本合集其他题的分工

- `quant-nf4-quantile-grid`：NF4 的分位构造与等距码表不同，本题固定使用 FP4/FP3 码表；
- `symmetric-quantize` / `asymmetric-quantize`：均匀整数格点的 fake-quant，本题是非均匀浮点码表；
- `quant-double-quant-scale`：对 scale 再做一次量化，本题只优化 scale 的**分侧**选择；
- `gptq-column-quantize`：跨列误差补偿，本题逐行独立、无补偿；
- 本题：AFPQ 的**双侧 scale + 最近码取整**，以及最近码平局规则。

## 6. 验证方法

- `check_example`：3.0/-0.3/0.2 的手算示例，覆盖双侧 scale 与码表取整。
- `check_codebook`：3/4-bit 下 `q_pos/q_neg` 只能取对应码表的值，且不超过 `maxq/2`。
- `check_scales`：四类行（不对称、全零、全负、全正）的 `scale_pos/scale_neg` 公式。
- `check_matches_oracle`：30 组随机配置（含单侧行）与纯 Python 逐标量 oracle 对拍。
- `check_benefit_vs_single_scale`：正负极值悬殊时双侧误差远小于单 scale。
- `check_tie_picks_first_code`：`q=1.75` 必须取 `1.5`，验证平局规则。
- `check_one_sided_rows` / `check_all_zero_row`：单侧行不做 0/0、全零行按 `(-1,+1)` 退化。
- `check_validation`：`bits`、维度、空张量、非浮点、非有限值非法时抛 `ValueError`。
