# 稀疏：LoSparse 低秩+结构化稀疏分解步（SVD 因子拆分 + 参数预算 + 神经元行剪枝）

## 题意

LoSparse 用「低秩矩阵 + 结构化稀疏矩阵之和」逼近权重：SVD 低秩部分压走神经元间的相干成分，结构化（神经元/行级）剪枝删走非相干成分——低秩防止剪枝丢表达神经元，剪枝增强低秩的多样性。实现 `solve(weight, x, parameter_ratio=0.15, rank_ratio=0.1, remove_criteria="max_eigenvalue", neuron_mask=None, bias=None)`：

```text
# 1. 分解 (官方 low_rank_decomposition)
U, S, Vh = svd(W, full_matrices=False)      # 奇异值降序
rank = count_nonzero(S)
parameter_ratio 非 None: k = ceil(pr · H·W / (H + W))   # 参数量占比, 覆盖 rank_ratio
否则:                    k = ceil(rank · rank_ratio)    # 秩占比
max_eigenvalue:   L = U @ √diag(S)[:, :k],  R = √diag(S)[:k, :] @ Vh
min_eigenvalue:   L = U @ √diag(S)[:, -k:], R = √diag(S)[-k:, :] @ Vh
(√diag 拆分使 L/R 参数量均衡; L @ R = 所选奇异三元组的重构)

# 2. 残差与剪枝
S_res = W − L @ R                           # 稀疏残差
S_masked = S_res · neuron_mask[:, None]     # 模拟训练后按行置零
nonzero_idx = nonzero(S_masked.sum(dim=1))  # 行和非零的幸存神经元
sparse_pruned = S_masked[nonzero_idx, :]

# 3. 前向 (x 为 (B, L, in))
未剪枝: Y = x @ R.T @ L.T + x @ S_res.T + bias
剪枝后: Y = x @ R.T @ L.T + scatter(x @ sparse_pruned.T, nonzero_idx) + bias
```

返回 `dict(L, R, sparse_residual, reduced_rank, rank, is_full_rank, nonzero_idx, sparse_pruned, output)`。

## 关键语义

- **parameter_ratio 覆盖 rank_ratio**：非 None 时走参数量预算 `ceil(pr·H·W/(H+W))`（k 隐含 < min(H,W)），None 才用 `ceil(rank·rank_ratio)`——rank 用 `count_nonzero(S)`（秩亏权重的奇异值精确为 0 才会计入）。
- **√diag 拆分是均衡因子**：L 列范数 = R 行范数 = √s_i，L@R 恰为所选三元组重构；漏掉 √ 会把谱平方塞进因子。
- **min_eigenvalue 取尾部**：SVD 降序下 `[-k:]` 是最小奇异三元组（与 max 路径结果不同）。
- **剪枝是行级结构化**：按 `sum(dim=1)`（输出神经元行和）判存活；被掩码行全零、和恒 0 自动落选——从残差原矩阵取幸存行与从掩码矩阵取等价。
- **scatter 前向**：剪枝后只算幸存行，再按 `nonzero_idx` scatter 回全宽输出——数值上等价于「全量残差前向 + 按掩码置零」。
- **官方 `random` criteria 触随机性红线剔除**；官方 `raise NameError` → 判题域 ValueError。

## 判题域约定

- 官方 PLATON 训练产生 `masked_fill_`；判题域以显式 `neuron_mask` (0/1 行掩码) 模拟训练置零，残差本身不训练更新。
- 权重 (out, in) 行主序 nn.Linear 约定，x 为 (B, L, in)；bias 原样加。
- 官方 log/打印与 SX 训练缓存（Deberta Q/K 双用）不入。
- 官方不校验；判题域补权重/x 形状/criteria/比例正数/掩码 0-1 与长度/bias 长度/k 上界/掩码剪空等显式校验。

## 例子

H=12、W=8、pr=0.15：k = ceil(0.15·96/20) = 1；L (12,1)、R (1,8)，L@R = 最大奇异三元组重构；残差 (12,8)。掩码下幸存行按行和判出，前向 scatter 回 (B,L,12)。秩亏权重（尾部行精确置零）时 rank=2、is_full_rank=False。

数据规模：H ≤ 32、W ≤ 32，x (B ≤ 4, L ≤ 16)，float32。
