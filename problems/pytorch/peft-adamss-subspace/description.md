# PEFT：AdaMSS 自适应多子空间散射适配

## 题目描述

**AdaMSS**（Adaptive Multi-Subspace orthogonal fine-tuning，NeurIPS 2025，官方仓 `jzheng20/NeurIPS2025-AdaMSS`，已入 HF PEFT `tuners/adamss`）在冻结奇异方向的骨架上**按组分段训练**：对带 bias 的增广权重 \(W' = [W \mid b]\ (out, in{+}1)\) 做 SVD，右奇异因子投影 \(\mathrm{newB} = V_r'^\top\ (r, in{+}1)\) **冻结**，左因子 \((out, r)\) 不整体训练——而是把 \(r\) 个 SVD 成分（out 维方向）用 KMeans 聚成 \(K\) 组，每组一个独立低秩子空间

$$A_i\ (\mathrm{rank}_i, r),\quad B_i\ (\mathrm{seg}_i, \mathrm{rank}_i),\quad \mathrm{rank}_i = \max(1, \min(\mathrm{seg}_i,\ \mathrm{subspace\_rank}))$$

不同子空间**异构秩**、也可整段冻结——这正是「自适应多子空间」的含义。

`adamss forward` 的确定性链（有 bias 时输入拼全 1 列，**输出维仍是 out_features**）：

```
x_aug     = [x | 1]                          # (..., in+1)
projected = x_aug @ newB.T                   # (..., r)   冻结投影
h_i       = projected @ A_i.T                # (..., rank_i)
y_i       = h_i @ B_i.T                      # (..., seg_i)
cat       = [y_0, y_1, ...]                  # (..., Σ seg_i)
delta[..., scatter_index] = cat              # 散射到 out 维段行
```

merge 主干（`get_delta_weight` / `get_delta_bias` 共用）：

```
combined = cat([B_i @ A_i], axis=0) @ newB    # (Σ seg_i, in+1)
dW_full[scatter_index, :] = combined          # (out, in+1) 增广 delta
weight delta = dW_full[:, :-1]                # bias delta = dW_full[:, -1]
```

**bias 是增广全 1 列对应的最后一列，不是额外一行**。scatter_index 由官方 `update_layer` 末尾按**可训练子空间顺序**拼接各段索引得到；未训练组的行在每个 delta 中保持全零。KMeans/SVD/PCA/QR 全部离线完成，锚点只钉确定性矩阵账。

## 规则

- `mode="build_scatter_index"`：seg 映射（组号 → 成分索引列表，划分 \(0..r{-}1\)）与 train_idx 入参，按训练组顺序拼接，返回一维行索引。
- `mode="rank_clamp"`：seg_lens 与 subspace_rank 入参，逐段返回 \(\max(1, \min(L, \mathrm{subspace\_rank}))\)（空段/零秩护栏：下限 1）。
- `mode="forward_delta"`：全链；`with_bias=True` 时输入拼全 1 列、经 newB 的 bias 列折算，输出维不变。
- `mode="get_delta_full"`：增广 delta \((out, in{+}1)\)（未剥 bias 列）；scatter_index 之外的行全零。
- `mode="split_delta"`：拆分 \((weight, bias)\)——with_bias 时取 `[:, :-1]` 与最后一列，否则全矩阵 + 全零 bias。
- newB/A_i/B_i/seg/train_idx 全部显式入参；全程 float64。

## 示例

\(out=6, in=5, r=5\)，成分聚 3 组 \([0,1],[2,3],[4]\)，subspace_rank=2，全训：

```
rank_actual = [2, 2, 1]            # max(1, min(seg, 2))
scatter_index = [0, 1, 2, 3, 4]
forward_delta(x, newB, A, B, sc, 6) -> (2,3,6) = x @ dW.T   # 互证
get_delta_full(newB_b, A, B, sc, 6) -> (6, 6)                # 增广形态
split_delta(dW_full, True) -> (dW_full[:, :-1], dW_full[:, -1])
```

部分训练（跳过组 1）：scatter_index = [0, 1, 4]，行 2、3 恒零。

## 约束

- forward 与 merge 数学同源：\(x\, @\, W_\Delta^\top\ (+\ b_\Delta) =\) forward（测试互证）。
- rank_clamp 满足：subspace_rank 大于段长时取段长；空段取 1。
- 输出维恒等于 out_dim（bias 折在输入侧，不是输出 +1）。
