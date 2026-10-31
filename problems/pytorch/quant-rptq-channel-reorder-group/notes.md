# 量化：RPTQ 通道重排与分组量化——算法详解

## 1. 分组量化的盲点

逐组量化的误差取决于组内动态范围。若按通道顺序切段，相邻通道的动态范围可能相差几个数量级：一个离群通道会把组 scale 撑大，其余通道只剩很少的整数格点，重构误差集中爆发。

RPTQ（Yuan et al., ICLR 2023, arXiv:2304.01089）的观察是：通道的量化敏感度与其激活范围强相关，把范围相近的通道放进同一组，可以显著压低组内 range。原仓库 `quantize/reorder_utils.py` 的默认实现用 sklearn `KMeans(n_clusters, n_init=10, random_state=0)` 在二维点 `(xmax_i, xmin_i)` 上聚类，再按 cluster label 排序；同一文件还提供了 PEG 变体 `peg_tensor_calc_reorder_index`：按 `xmax-xmin` 排序后等量切段。

## 2. 组内量化的公式

每组把 `[gmin, gmax]` 映射到 `2^bits` 个等宽格点：

\[
s=\operatorname{clamp}\!\left((g_{\max}-g_{\min})\,2^{-b},\ 10^{-4},\ 10^{4}\right)
\]

为了让组中点对齐零，零点取

\[
z=\operatorname{clamp}\!\left(\operatorname{round}\!\left((g_{\max}+g_{\min})\cdot\frac{-0.5}{s}\right),\ -10^{4},\ 10^{4}\right)
\]

fake-quant 先做 `w/s` 取整并加 `z`，再截断到有符号整数区间：

\[
q=\operatorname{clamp}\!\left(\operatorname{round}(w/s)+z,\ -2^{b-1},\ 2^{b-1}-1\right)-z,
\qquad \hat w=q\cdot s
\]

正侧只有 \(2^{b-1}-1\) 个正码，比负侧少一个，所以组内正向最大值会先饱和。示例中 `8.4` 落在 `scale=1.05` 的组里：`8.4/1.05 = 8`，被截到 `7`，重建为 `7.35`。`CLIPMIN=1e-4` 防止零跨度组除零，`1e4` 上限防止 scale 爆炸。

## 3. 为什么用 PEG 口径而不是 KMeans

上游默认路径依赖 sklearn KMeans 的浮点累加。两个到簇心距离几乎相等的通道，在 torch 与纯 Python 实现下可能被分到不同簇，`perm` 随之不同——这对确定性判题是致命的。PEG 是同一个实现文件里的确定性替代：只比较浮点跨度、稳定排序、等量切段，任何语言结果一致。

本题采用 PEG，并把上游「`counts=[C//g]*g`、丢弃余数」的行为修正为均衡分段：前 `C % g` 段各多 1 个，保证 `sum(counts) == C`，不会有通道被漏掉。这是本题与上游实现的唯一有意偏差，已在题面写明。

## 4. 边界与退化

- 跨度全为 0：`scale` 落到 `1e-4`；若通道值也为 0，则 `zero=0`、重建为 0；若常量非零（例如 `xmax=xmin=2`），`zero` 会按公式被 `±1e4` 截断，重建由 `(q-zero)*scale` 决定，结果可能明显偏离常量——这是上游公式的固有行为，本题照实保留。
- `num_groups=1`：所有通道一组，`scale` 由全局最大/最小决定。
- `num_groups=C`：每组一个通道。
- 输入要求：`bits ∈ [2,8]`、`num_groups ∈ [1,C]`、`xmin <= xmax`、全部有限。
- 输出 dtype：`perm/counts/zero_points` 为 int64，`scales/dequant` 为 float32；内部全程 float64。

## 5. 与本合集其他题的分工

- `gptq-column-quantize`：跨列误差补偿（Hessian 逆），本题不做补偿；
- `quant-squeezellm-sensitive-kmeans`：优化非均匀格点位置，本题格点均匀；
- `quant-double-quant-scale` / `quant-nf4-quantile-grid`：scale 的二次量化 / 分位格点，本题固定指数格点；
- `llm-int8-outlier-split`：把通道拆成 outlier 与 normal 两部分；本题是**重排 + 均衡分组**，不拆通道；
- 本题：确定性通道重排（permutation）+ 均衡分组 + 组内非对称 fake-quant。

## 6. 验证方法

- `check_example`：手算示例，覆盖正侧饱和（`8.4 -> 7.35`）。
- `check_perm_is_span_sorted`：`perm` 是排列，且按 `(跨度, 通道号)` 稳定升序。
- `check_balanced_groups`：段长只允许 `C//g` 或 `C//g+1`，余数落在前 `C%g` 段。
- `check_group_scales`：每组 `scale/zero` 由段内 `max/min` 唯一确定。
- `check_matches_oracle`：40 组随机配置与纯 Python 逐标量 oracle 对拍（`perm/counts/scale/zero/重建`）。
- `check_quantization_grid`：重建值落在整数格点上，反解出的整数码被 clamp 在 `[qmin, qmax]`。
- `check_non_identity_and_single_group` / `check_constant_channels`：非恒等重排、单组退化、零跨度与常量通道。
- `check_validation`：类型、范围、形状、有限性、顺序非法时抛 `ValueError`。
