# 量化：SqueezeLLM 敏感度加权非均匀格点——算法详解

## 1. 为什么格点要「非均匀 + 加权」

均匀量化（`symmetric-quantize` 一族）假设权重同等重要、分布均匀。两组现实打破它：

- **分布不均**：LLM 权重近似高斯，中间密两端稀，等距格点在 0 附近浪费分辨率；
- **重要性不均**：SqueezeLLM（Kim et al., ICML 2023, arXiv:2306.07629）用 Fisher 信息（输出对权重的二阶敏感度）给每个权重打分，量化误差落在高敏感权重上代价成倍放大。论文的敏感度近似为 \(\partial \mathcal L/\partial w\) 的平方（一次反向即可得），据此构造 **sensitivity-aware k-means**：加权目标让格点向高敏感、高密度区域聚拢。

## 2. 加权 Lloyd 的两步（判题考点）

目标

\[
\min_{\{c_j\}, a}\ \sum_i s_i\,(w_i-c_{a_i})^2
\]

- **分配**：固定中心，逐点取最近簇——\(a_i=\arg\min_j|w_i-c_j|\)，并列取**较小**簇号（`argmin` 语义）；
- **更新**：固定分配，每个中心的最优解是**加权均值**

\[
c_j^\star=\frac{\sum_{i:a_i=j}s_iw_i}{\sum_{i:a_i=j}s_i}
\]

  （对 \(c_j\) 求导：\(-2\sum_{i:a_i=j}s_i(w_i-c_j)=0\)）。这是 `check_weighted_centroids` 的核心断言：返回的每个中心必须等于其簇内加权均值（容差 \(10^{-4}\)）——用算术均值或忘记乘 \(s_i\) 会立刻失败。
- **空簇**：没有点时保留原中心（否则会出现 NaN 或中心塌缩）；初始化用位置分位点保证不会一上来就空簇。

## 3. 单调性与排序

1D 的 Lloyd 每轮都降低目标（分配步 + 加权均值步各自最优），因此迭代是单调的。每次更新后把中心升序排序（分配在下轮重算），保证：
- 输出 `centroids` 单调，便于下游做「最近码二分查找」；
- `assign` 与 `centroids[assign]` 的对应关系明确（`check_weighted_centroids` 断言 `deq == centroids[assign]`）。

`check_beats_uniform_grid` 给出宏观对比：同样的 k 下，学出的格点加权 MSE 必须显著低于 `linspace(min, max, k)` 的等距格点；这是非均匀格点存在的意义，也防止「只把 k-means 跑成 k 个分位点」的退化解。

## 4. 敏感度语义的定向断言

`check_sensitivity_shifts_centroid` 用 k=2、`w=[0,1,2,3,4]` 两组敏感度对比：

- base：`sens` 全 1 → 中心在 1 与 3（左侧簇含 0,1,2）；
- hi：把 3、4 的敏感度抬到 100 → 右侧中心被拉向 4（加权均值 \((3+4\cdot100)/101\approx3.99\)），左侧中心不动。

这既验证了加权均值的方向，也验证了「敏感度只影响所在簇」的局部性。

## 5. 与其余量化题的分工

- `quant-hqq-scale-solve`：格点**均匀**，只优化 scale（闭式）；
- `quant-nf4-quantile-grid`：格点由**正态分位**固定，只优化编码；
- 本题：格点位置**由数据 + 敏感度学出来**（非参数、非闭式），是三者中唯一的「学习式」码本；
- `gptq-column-quantize`：跨列误差补偿，不改格点分布。

## 6. 验证方法

- `check_example`：4 点、k=2、1 轮手算。
- `check_weighted_centroids`：加权均值 + 升序 + `deq` 自洽。
- `check_beats_uniform_grid`：加权 MSE < 均匀格点的 50%。
- `check_uniform_sens_matches_lloyd`：`sens` 全 1 时时标量重算对拍。
- `check_sensitivity_shifts_centroid`：敏感度定向移动 + k/iters/sens 五类非法入参。
