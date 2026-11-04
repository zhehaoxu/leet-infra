# PEFT：FRoD 旋转投影对角加稀疏增量

## 题目描述

**FRoD**（Frequency-domain Rotational Decomposition，AAAI 2026，已入 HF PEFT v0.20.0 `tuners/frod`）把「同类别各层」的增量权重塞进一个**共享正交基**里：跨层拼接联合分解得到方阵旋转 \(V\)，每层只留自己的对角系数 \(\lambda\)（秩 1 视角）加一个**极稀疏非对角矩阵 \(L\)**，\(\Delta W = U\,(S+L)\,V^{\top} - W\)（预校正回原权重）。与 LoRA 的逐层独立 \(A,B\) 相比，FRoD 的可训练面从 \(O(r\cdot(in+out))\) 压到「对角线 + \(s\) 个稀疏位」。

官方 `layer.py` 的确定性账（锚点全部可离线重算）：

```
z   = x @ V                      # 旋转到共享基
z_S = z @ S.T                    # 稀疏分支（S 非对称，官方 _sparse_activation_mm 是 S @ zᵀ 再转置）
z_L = z * λ                      # 对角分支（逐元素）
out = base(x) + (z_S + z_L) @ U.T
```

三个关键口径：

1. **共享投影 \(V\)**：同类别各层 \(W\) 沿行拼接成 \((\sum out, in)\)——行多列少走 SVD（\(V=V_{vh}\)，行正交）；行少列多走 QR + 岭平均 + `eigh`（\(V=R^{\top}\) 的特征向量组装）。float32 计算。
2. **稀疏 COO**：\(in^2\) 网格**去掉对角线**展平，`prng.randperm` 取前 `nnz = max(1, floor(in²·sparse_rate))` 个位置作为 `frod_s_indices`，values 零初始化。
3. **层内账**：\(U\,\mathrm{diag}(\lambda) = W V^{\top}\)（方阵 `solve(V, Wᵀ)`，非方阵 `pinv` 回退）、\(\lambda\) = \(WV^{\top}\) 的**列范数**、\(U\) 列归一化；\(S+L\) 装载后 `get_delta_weight` 回 \(\Delta W = U(S+L)V^{\top}-W\)。

**\(S=0\) 时 \(\Delta W \equiv 0\) 精确成立**（\(U\,\mathrm{diag}(\lambda)V^{\top} = W\) 是恒等重构），测试以此互证分解域参数化的零起点语义。

## 规则

- `mode="shared_projection"`：各层 \(W\) 列表入参，返回共享 \(V\)（\((in,in)\)，与层数无关；瘦阵 SVD 路径返回行正交 \(V\)）。
- `mode="sparse_indices"`：\(in\) 与 `sparse_rate`、`prng_seed` 入参，返回 `nnz×2` COO 索引（`nnz = max(1, floor(in²·rate))`，非对角、确定性）。
- `mode="sparse_dense"`：values + indices + `s_size` 入参，散回稠密 \((in,in)\)（缺省位为 0）。
- `mode="solve_u_lambda"`：\(W\) 与 \(V\) 入参，返回 `{"u", "lambda"}`（\(\lambda\) 是 \((in,)\) 列范数）。
- `mode="forward_delta"`：返回 \(x\,\Delta W^{\top}\)（与 \(xV(S+L)^{\top}U^{\top}\) 互证）。
- `mode="get_delta_weight"`：返回 \(U(S+L)V^{\top}-W\)。
- 全程 torch：V/S 索引 float32（官方路径），W/x/ΔW float64。

## 示例

\(in=6\)，两层 \((5,6),(4,6)\)：

```
shared_projection([W1, W2]) -> V (6,6)        # 15+8=23 行 > 6 列 → 宽阵路径
sparse_indices(6, 0.1, seed=0) -> 3 个非对角 (row,col)
get_delta_weight(U, lam, s_vals, s_idx, V, W) -> U(S+diag λ)Vᵀ - W
```

## 约束

- `s_indices` 无重复、无对角（\(i\ne j\)）。
- 瘦阵路径 \(V V^{\top}=I\)（行正交）；宽阵路径列近似正交。
- `get_delta_weight` 与 `forward_delta` 转置自洽：\(x\,\Delta W^{\top} = xV(S+L)^{\top}U^{\top}\)。
- 病态 \(V\)（大条件数）时方阵路径走 `solve` 而非 `pinv`（后者 rtol 截断小奇异方向，数值可区分）。
