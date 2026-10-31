## 题意

LLM-Pruner（2305.11627）的剪枝单位不是单个权重，而是**依赖组**：多头注意力里 q/k/v 的一个头与 o_proj 对应行块是耦合的——动头必须四者一起动。它用一阶泰勒显著性 `salience = w · g`（逐元素）给组打分：梯度与权重的乘积近似「移除该参数对损失的直接影响」。

本题给定多头注意力的四个投影矩阵与其梯度，复现每头的标量重要性与移除头的选择。判题输入：

- `wq` / `wk` / `wv` / `wo`：四个 `(d, d)` 权重矩阵（q/k/v/o 投影，MHA 布局：头 `h` 对应 q/k/v 的列块 `[h·head_dim, (h+1)·head_dim)` 与 o 的行块同区间）；
- `gwq` / `gwk` / `gwv` / `gwo`：对应的梯度矩阵（同形状同 dtype）；
- `head_dim`：每头维度，`d` 必须能被它整除；
- `mode`：`"param_first"` 或 `"vectorize"`（上游 `--taylor` 的两种口径）；
- `group_reduction`：`"sum"` 或 `"mean"`（上游 `--grouping_strategy` 的归并方式）；
- `num_prune`：要移除的头数。

## 计算规则

**第 1 步：逐元素泰勒显著性**。`salience_X = w_X · g_X`（四个矩阵各自逐元素相乘）。

**第 2 步：行/列方向归约**（上游 attention 分支：o_proj 是 out_channels 依赖、q/k/v 是 in_channels 依赖）：

| mode | o_proj（行方向） | q/k/v（列方向） |
| --- | --- | --- |
| `param_first` | `\|salience_o\|.sum(dim=1)` | `\|salience_X\|.sum(dim=0)` |
| `vectorize` | `\|salience_o.sum(dim=1)\|` | `\|salience_X.sum(dim=0)\|` |

`param_first` 先取绝对值再求和（幅值证据）；`vectorize` 先求和再取绝对值（**符号抵消**：同块内正负泰勒项相消，块重要性可能骤降为 0）。

**第 3 步：头块聚合**。把第 2 步的 `(d,)` 向量按 `head_dim` 分块求和，得到每头的块贡献 `(n_heads,)`——o 的行块贡献与 q/k/v 的列块贡献各自聚合。

**第 4 步：跨源归并与选择**。四个贡献堆叠成 `(4, n_heads)`，按 `group_reduction` 归并（`sum` 求和 / `mean` 均值，两者只差常数 4 倍，选择不变但数值不同），得每头标量重要性 `head_importance`。升序 **stable** 排序，取最小的 `num_prune` 个头移除；并列值取索引小者。

返回字典：

| 键 | 含义 |
| --- | --- |
| `head_importance` | 每头标量重要性，`(n_heads,)` float64 |
| `per_source` | 四个源（o/q/k/v）各自的头块贡献，`(4, n_heads)` |
| `pruned_heads` | 被移除头的升序索引列表 |
| `n_heads` | 头数 `d / head_dim` |

## 手算示例

```
w = [[ 1,  2,  3,  4],
     [ 5,  6,  7,  8],
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]]   # 四个矩阵同；梯度全 1；head_dim=2
```

- `salience = w`（梯度全 1）；o 行和 `[10, 26, 42, 58]` → 头块 `[36, 100]`；
- q/k/v 列和 `[28, 32, 36, 40]` → 头块 `[60, 76]`；
- `head_importance = [36+60+60+60, 100+76+76+76] = [216, 328]`；
- `num_prune=1` → 移除头 0。

**vectorize 抵消示例**：仅 q 非零，列块 0（头 0）为 `[1,-3,1,-3]`、列块 1（头 1）为 `[2,-2,2,-2]`：

- `param_first`：`|·|` 列和 `[6, 8]` → 头块 `[6, 8]`；
- `vectorize`：列和 `[0, 0]` → 头块 `[0, 0]`——同输入两种口径重要性完全不同。

**选择反转示例**（`head_dim=1`）：头 0 的 q 列 `[3, -3]`（绝对值和 6、求和后 abs 0），头 1 的 q 列 `[2, 0]`：`param_first` 得 `[6, 2]` 剪头 1；`vectorize` 得 `[0, 2]` 剪头 0——先 abs 还是后 abs 决定剪哪个头。

## 边界与约束

- 四个权重、四个梯度两两同形状 `(d, d)` 且同 dtype（float32/float64），权重与梯度 dtype 一致；
- `d % head_dim == 0`（行、列两个方向都要整除）、`head_dim ≥ 1`；
- MHA 布局要求 `d_out == d_in`（o 行块与 q/k/v 列块对齐）；
- `num_prune ∈ [0, n_heads]`；`mode`/`group_reduction` 枚举合法；
- 并列按 stable sort（小索引优先）；`num_prune=0` 返回空列表；违例抛 `ValueError`。

## 与现有题的边界

- 470 `sparsity-head-importance`：头重要性是 `\|\|W_h\|\|_F · \|\|X_h\|\|_F`（激活范数、无梯度），本题是泰勒乘积 `w·g` 且有 group 依赖结构；
- 582 `sparsity-owl-outlier-layerwise`：OWL 考层间稀疏度分配，本题考组内耦合打分；
- 472 `sparsity-ste-mask-backward`：STE 是掩码直通估计的梯度路径，不做组选择；
- 473 `sparsity-movere-reactivation`：移动/复活机制，方向相反（恢复而非移除）。
