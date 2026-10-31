# 量化：BiLLM 结构拆分与残差二值——算法详解

## 1. 题目从哪来

BiLLM（arXiv:2402.10685，`Aaronhuang-778/BiLLM`）把 1-bit 权重量化拆成「结构化显着段 + 两条 1-bit 残差通路」。仓库里与本题对应的实现是：

- `binary.py::high_order_residual`：逐行残差二值；
- `utils/autosearch.py::structural_searching`：第一段列数 `i` 与第二段幅值切分点的两段搜索；
- `utils/mask.py::generate_structural_mask`：按切分值生成 `mask1/mask2`；
- `bigptq.py`：把三组掩码交给 `Binarization.quantize`，并可选做 GPTQ 误差补偿。

本题只保留前三者的确定性计算：不读校准数据、不用 Hessian、不做 GPTQ 补偿、不训练任何参数。`up_lim` 对应上游 `structural_guassian_distribution(..., up_lim=30)` 的参数，在本题里由调用方给出。

## 2. 算法步骤

记 `HOR(x, mask, order)` 为逐行残差二值。对每一行，只在 `mask` 选中的列上统计：

1. `r = x`（第一轮）；
2. `mu = mean(r over mask)`；
3. `sigma = mean(|r - mu| over mask)`；
4. 重建 `r_hat = sign(r - mu) * sigma + mu`，写回这些列；
5. 若 `order=2`，令 `r = x - r_hat` 再执行 2–4，并把两轮结果相加。

第一段搜索：

- `count_c = sum_r |W[r, c]|`，按 `(count, 原列号)` 降序稳定排序，取前 `up_lim` 列为候选；
- 对 `i = 1 .. up_lim-1`，令 `mask3` 为候选前 `i` 列，计算 `HOR(W, ~mask3, 2) + HOR(W, mask3, 2)` 与 `W` 的 MSE；
- 严格小于才更新，得到最优 `i`。

第二段搜索：

- 固定 `mask3`，先算 `group3 = HOR(W, mask3, 2)`；
- 把 `mask3` 列置零后展平 `|W|` 并升序排序；
- 在 81 个分位点 `0.10 + 0.01k`（线性插值）上得到候选切分值；
- 对每个切分值生成 `mask2 = (~mask3) & (|W| >= split)`、`mask1 = (~mask3) & (|W| < split)`，计算 `HOR(mask1,1) + HOR(mask2,1) + group3` 的 MSE；
- 严格小于才更新，最优切分值写入 `split`。

## 3. 源码头细节

- **逐行而非逐列**：`nanmean(masked_x_tensor, dim=1)` 的统计维度是行；同一行共用 `mu/sigma`，所以 `mask1` 里一行三个值会一起被拉向该行的均值。
- **被屏蔽位置的处理**：上游用 `torch.where(mask, residual, nan)`，再做 `nanmean`；`torch.sign(nan)` 在 PyTorch 中返回 0，于是 `binary * mask` 在掩码外得到 0，不会污染结果。本题用显式置零实现同一语义，避免依赖 `nan` 行为。
- **空行**：某行没有任何入选列时，`nanmean` 为 `nan`，上游用 `torch.where(isnan, 0, ...)` 置 0；`order=2` 的两轮都输出 0。
- **分位点包含显着列的 0**：上游展平的是 `origin_matrix * (~mask3)`，显着列变成 0 参与排序；当 `mask3` 列数较多时，低分位点会被大量 0 占住，`split` 可能正好为 0，此时 `mask1` 为空、全部剩余列进入 `mask2`。这是合法输出，不能人为抬高分位。
- **边界归属**：`mask2` 用 `>= split`，`mask1` 用 `< split`；`split` 本身常常等于某个 `|W|` 值（分位插值命中数据点），等号归属必须按此规则。
- **两段误差口径**：第一段比较的是 `group3 + group4`（`group4` 是 `~mask3` 上的 order=2 重建），第二段比较的是 `group1 + group2 + group3`。第一段的 `group4` 只是搜索用的临时量，不进入最终输出。

## 4. 确定性边界

- **并列**：`torch.topk` 对相等列范数的返回顺序不保证稳定，本题固定为「原列号升序」；两段搜索都用严格小于，并列保留先出现者。实现里必须写死这两条，否则相同输入可能得到不同掩码。
- **分位网格**：上游 `torch.linspace(0.10, 0.90, 81)` 默认 float32，分位值因此带 ~1e-8 抖动；本题约定用 float64 的 `0.10 + 0.01k` 并线性插值，掩码在非临界输入上完全一致。这一差异只影响切分值的末位，不影响算法语义。
- **数值精度**：统计量、误差、`split` 内部全部 float64；输出 `reconstructed`/`split` 为 float32。比较 `mask2 = |W| >= split` 时用的是 float64 的 `split`，不要先把 `split` 降到 float32 再比较，否则恰好落在切分点上的元素可能翻转。
- **无随机性**：整个流程没有随机数、没有校准、没有优化器；同一输入必得同一输出。

## 5. 与本合集其他题的分工

- `quant-pbllm-partial-binarize`（order 620）：全局幅度阈值一次划出二值/高位两段，二值段用 `sign(x - mean)`，高位段用逐行非对称整数；没有列结构选择，也没有多阶残差。
- 本题：先用**列范数结构化搜索**定 2-bit 显着段，再在剩余权重上做**幅值两段 + order-1 残差**，显着段自身用 **order-2 残差**；输出三组掩码与重建矩阵。
- `quant-squeezellm-sensitive-kmeans`、`quant-nf4-quantile-grid` 等码本题：本题不产生码本，也不做分位量化，只做拆分与符号重建。
- 1-bit 稀疏近零组（上游 `method="prune"`）与 GPTQ 补偿不在本题范围；需要补偿的题另有 GPTQ 系列题覆盖。

## 6. 验证方法

- `check_example`：2×6、`up_lim=3` 手算示例，核对显着列、`split`、三个掩码与重建矩阵。
- `check_matches_oracle`：五组形状/`up_lim`（含 `up_lim=2` 与 `up_lim=列数`）与纯 Python oracle 对齐。
- `check_salient_columns_by_norm`：显着列必须按列绝对值和挑选，`up_lim=2` 只能选出第 1 名，输出升序。
- `check_masks_partition`：三个掩码互斥、并集覆盖全部元素，`mask3` 恰好是整列，`mask1/mask2` 非空。
- `check_split_is_masked_quantile`：`split` 必须落在「显着列置零后的 |W|」的 81 个分位点上。
- `check_ties_stable`：列范数全相等时按原列号升序取候选，显着列是 `[0, 1, ...]` 的前缀。
- `check_boundary_up_lim`：`up_lim=2` 只选 1 列，`up_lim=列数` 最多选 `C-1` 列，单行矩阵可用。
- `check_validation`：维度、空张量、非浮点、非有限值、`up_lim` 非整数或越界时抛 `ValueError`。
