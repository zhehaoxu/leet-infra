## 题意

Wanda（581 题）用 `|W|·RMS(激活列)` 给权重打显著性分。GBLM-Pruner（ICLR 2024）主张：剪枝决策还应该看**训练梯度**——一个权重若在微调样本上梯度很大，说明模型还想动它，剪掉代价高。

本题在给定的逐样本梯度上复现 GBLM 的融合显著性并完成逐行剪枝。判题输入：

- `weight`：形状 `(out, in)` 的浮点权重矩阵；
- `activations`：形状 `(tokens, in)` 的校准激活（每行一个 token）；
- `grad_samples`：形状 `(samples, out, in)` 的逐样本梯度张量；
- `sparsity`：非结构化剪枝比例，落在 `[0, 1]`；
- `scale`：梯度缩放因子（上游默认 100，判题显式给定，须为正数）；
- `gradient_inv`：是否使用梯度倒数变体；
- `prune_n` / `prune_m`：N:M 结构化参数，`prune_n > 0` 时启用。

## 计算规则

**第 1 步：激活列均方**。`scaler_row = mean(activations², dim=tokens)`，得长度 `in` 的向量；逐列开平方得 `√s_row`。

**第 2 步：梯度 L2 聚合**。对每个样本先乘 `scale`，再按样本维求平方和后开方：

```
G[i,j] = sqrt( Σ_s (grad_samples[s,i,j] · scale)² )
```

这对应上游 `gradient_computation.py` 先累积 `Σ|g·scale|²`、最后统一 `sqrt` 的两段式；**不是**逐样本取 max，也**不是** L1 聚合 `Σ|g|`。

**第 3 步：融合显著性**。令 `base = |W| · √s_row`（Wanda 项，逐元素，`√s_row` 按列广播）：

- 非 inv：`metric = base + |W| · G`（两证据相加，梯度项**不带** `√s_row` 因子）；
- inv 变体：`metric = base / (G + 1e-8)`——梯度大处的显著性被压低（上游 `--gradient_inv` 开关，`small_value=1e-8` 防零除；零梯度位置的分母恰为 `1e-8`，metric 成为巨值，**不会被剪**）。

**第 4 步：掩码与置零**：

- 非结构化（`prune_n = 0`）：对 `metric` **逐行**做 stable sort，每行取前 `int(in · sparsity)` 个最小值的列索引，对应位置置零；
- N:M 结构化（`prune_n > 0`）：要求 `0 < prune_n < prune_m`、`in % prune_m == 0`、`sparsity == 0.5`；把列按每 `prune_m` 个分块，每块内取最小的 `prune_n` 个置零（与 581 题同一窗口规则，但作用在融合 metric 上）。

返回字典（键与 581 题同族，多一个 `grad_l2`）：

| 键 | 含义 |
| --- | --- |
| `metric` | 融合显著性矩阵，`(out, in)` |
| `grad_l2` | 第 2 步的 `G`，`(out, in)` |
| `mask` | 布尔掩码，`True` = 被剪 |
| `pruned_weight` | 置零后的权重 |
| `sparsity` | 实际稀疏度 = mask 均值 |

## 手算示例

```
W = [[1, 2, 3, 4],
     [4, 3, 2, 1]]
A = [[1, 0, 1, 0],
     [1, 1, 0, 0]]      # 2 个 token → mean 口径 s_row = [1, 0.5, 0.5, 0]
grads（2 个样本，scale = 1）:
  样本1 = [[.1,0,0,0],[0,0,0,.2]]
  样本2 = [[.1,0,0,0],[0,0,.3,0]]
```

- `G`：第 0 行 `[0.1√2, 0, 0, 0]`，第 1 行 `[0, 0, 0.3, 0.2]`；
- `√s_row = [1, 1/√2, 1/√2, 0]`；
- `metric` 第 0 行 = `[1+0.141421, 1.414214, 2.121320, 0]` → 剪 `{3, 0}`；
- `metric` 第 1 行 = `[4, 2.121320, 2.014214, 0.2]` → 剪 `{3, 2}`；
- `pruned_weight = [[0, 2, 3, 0], [4, 3, 0, 0]]`，实际稀疏度 0.5。

**scale 是真考点**：同一输入把 `scale` 改成 10，第 0 行 metric 变为 `[2.428427, 1.414214, 2.121320, 0]`，剪枝集合从 `{3, 0}` 变为 `{3, 1}`——因为两项相加后不同缩放因子改变了两证据的相对权重。

**gradient_inv 反转示例**：`W = [[10, 8]]`、`A = [[1, 1]]`、单样本梯度 `[1, 20]`：非 inv 得 `[11, 28]` 剪第 0 列；inv 得 `[10/1, 8/20] = [10, 0.4]` 剪第 1 列——梯度大、幅值小的权重在 inv 口径下反而先被剪。

## 边界与约束

- `weight`/`activations` 同 dtype（float32 或 float64）且列数一致；
- `grad_samples` 三维、至少 1 个样本、后两维与 `weight` 同形；
- `sparsity ∈ [0, 1]`；`scale > 0`；`prune_n, prune_m ≥ 0`；
- N:M 模式额外要求 `0 < prune_n < prune_m`、`in % prune_m == 0`、`sparsity == 0.5`；
- 内部以 float64 计算（上游为 fp16 累积，判题以高精度口径对拍）；
- 并列值按 stable sort（索引小者优先）；违例抛 `ValueError`。

## 与现有题的边界

- 581 `sparsity-wanda-activation-metric`：只有 Wanda 项，无梯度融合、无 inv 变体、无 `scale`；
- 474 `sparsity-sparsegpt-reconstruct`：SparseGPT 走 OBS 逐列重构，不是打分排序；
- 218 `sparsity-nm-2-4`：只考 2:4 窗口本身，本题 2:4 只是 metric 的作用域；
- 217 `sparsity-magnitude-prune`：纯幅值全局阈值，逐行口径与融合项均不同。
