# 解题笔记：LoSparse 低秩+结构化稀疏分解步

## 回源定位

- 源文件：yxli2123/LoSparse（ICML 2023 官方仓，compressics/IntelLabs 均不存在）`utils.py` 399 行。训练侧 PLATON 剪枝器（ipt/exp_avg_unc/structured_prune 的 mean/sum/max/min 行列聚合）是训练收敛类机制，触红线不入；只收**静态分解与前向**。
- `low_rank_decomposition`（L7-66）：svd(full_matrices=False) → rank=count_nonzero(S) → parameter_ratio 优先 ceil(pr·HW/(H+W))、否则 ceil(rank·rank_ratio) → √diag(S) 拆分到 L/R 两侧，max 取 `[:, :k]`、min 取 `[:, -k:]`、random 用 random.choices（随机性红线剔除）。
- `substitute_layer_weights`（L183-194）：`S = W − L@R` 作稀疏残差放入 LinearLoSparse。
- `prune_sparse`（L136-139）：`nonzero_idx = nonzero(sparse.weight.sum(dim=1))`（out 行和）→ `sparse_weight_pruned = S[nonzero_idx]`。
- 前向两路（L91-126）：未剪枝 `LRX + sparse(x)`；剪枝后 `x @ pruned.T` scatter 到 `[..., nonzero_idx]`（SX 缓存是 Deberta Q/K 双用工程，不入）。
- 与 `quant-gear-lowrank-outlier` 划界：GEAR 是「SVD 低秩 + 残差按幅度 top-k 元素稀疏」（非结构化、量化侧）；LoSparse 是「低秩 + **行级结构化**剪枝 + 训练中联合微调」，本题考预算公式/√拆分/行剪枝 scatter，机制不重叠。

## 数学要点

1. **√diag 拆分**：W ≈ U_k diag(s_k) V_k^T = (U_k diag(√s)) @ (diag(√s) V_k^T) = L@R，两侧参数均衡（列/行范数均 √s_i）；漏 √ 的变异被列范数断言与 oracle 同时抓。
2. **参数预算公式**：k = pr·HW/(H+W) 使 (H+W)·k ≈ pr·HW——低秩分支的参数量占比。正比例经 ceil 恒 ≥1，k<1 是死代码只查上界 k>min(H,W)。
3. **秩亏计数**：`count_nonzero(S)` 对 float32 SVD 极脆——重复行给出 ~1e-8 非零奇异值仍计满秩；只有**精确零行**（尾部行置零）才产生精确 0 奇异值。测试构造必须用零行，不能用重复行。
4. **行剪枝的自动落选**：掩码行全零 → 行和 0 → nonzero 自动排除 → `S_masked[idx]` 与 `S_res[idx]` 逐元素相等（等价变异 m-pruned-nomask 的依据）；`abs().sum` 与 `sum` 对 nonzero 选择也精确等价（|s|≠0 ⟺ s≠0，等价变异 m-nonzero-abs）。
5. **scatter 前向恒等式**：`scatter(x@S_pruned.T, idx) == (x @ (S·mask).T)`——剪枝是稀疏优化的实现细节，不改变数学。

## 实现要点

- 校验顺序：weight 2 维 → x 3 维且入维 = W → criteria 二选一 → parameter_ratio 正数或 None → rank_ratio 正数 → neuron_mask 1D 长度 H 且 0/1 → bias 1D 长度 H 或 None → 分解 → k 上界 → 掩码剪空。
- 分解用 float32 SVD（跟随官方，不做 double 提升）；oracle 侧 svd 作共享原语（与 770 同分层模式）。
- scatter 的 zeros dtype 用 x.dtype，`SX_scattered[..., nonzero_idx] = x @ sparse_pruned.T`。

## 测试要点

- oracle 镜像记账 + 共享 svd；L/R/残差/nonzero/前向全对拍。
- 重构性质：L@R == 前 k（或后 k）SVD 三元组重构；L 列范数 == √s_i（√ 拆分专属断言）。
- criteria 对比：min 与 max 的 L 不同；rank_ratio 路径必须用**秩亏权重**（rank < min(H,W)）才能区分 `rank·ratio` 与 `min(H,W)·ratio` 变异。
- 剪枝恒等式：剪枝前向 == 全量残差前向 + 按掩码置零（scatter 语义）；nonzero_idx == nonzero((残差·掩码).行和)。
- 未剪枝前向恒等：Y == x @ weight.T（分解是精确线性拆分 W = LR + S）。
- 秩亏：零行构造（精确 0 奇异值）→ rank/is_full_rank。
- 校验 battery：2 维/3 维/入维/random/比例 0 与 bool/掩码长度与 0-1/bias 长度/k 上界(pr=5)/掩码全零。

## 变异清单（35 条，全部 CAUGHT；等价剔除 2 条）

预算 7：parameter_ratio 覆盖翻转/None 反转/ceil→floor/公式分子分母互换/rank_ratio 用 min(H,W)/rank 计满秩/is_full_rank 恒真。SVD 6：√ 丢弃、L/R 拆分互换、max↔min、min 切头、k−1、Vh 转置。残差 3：符号反、全 SVD 残差、残差=W。剪枝 5：掩码不乘/掩码按列、nonzero 按列/abs（等价剔除）。前向 6：LR 顺序/因子互换、scatter 换全量/不转置、未剪枝残差项换 R.T、bias 丢弃。校验 13：weight/x 维度、入维、criteria、比例正数、掩码 0-1/长度、bias、剪空、消息。

等价剔除 2 条：`m-nonzero-abs`（|s|≠0 ⟺ s≠0，nonzero 选择不变）；`m-pruned-nomask`（掩码行全零 → 行和 0 → 自动落选，从原残差取幸存行逐元素相等）。

经验：① **count_nonzero 的秩亏构造要用精确零行**——重复行在 float32 SVD 下留 ~1e-8 残留奇异值仍计满秩；② **替换串别丢 if 的冒号/右括号**——产生语法错误后判题器返回 0/0，被 `passed < total` 的存活判定误判为 SURVIVED（变异脚本先验检查：替换后源码必须可编译）；③ 秩预算分支的变异分化需要 rank < min(H,W) 的权重（满秩下 `rank·ratio` 与 `min(H,W)·ratio` 同值）；④ nonzero 类断言对 abs/符号变换天然免疫（非零选择不变），这类变异直接按等价记录不硬造用例。
