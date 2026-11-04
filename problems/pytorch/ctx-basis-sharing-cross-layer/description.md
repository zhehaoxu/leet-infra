# 上下文压缩：Basis Sharing 跨层共享基低秩分解

## 题目描述

**Basis Sharing**（Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression，arXiv:2410.03765，ICLR 2025，官方仓 `TUDa-HWAI/Basis_Sharing`）不满足于逐层独立 SVD 压缩：把**不同层的同名投影权重拼成一张宽矩阵后联合分解**——组内所有层复用同一组共享基向量（basis），每层只留自己的窄系数矩阵。参数量从 \(m \cdot out \cdot in\) 降到 \(in \cdot k + m \cdot k \cdot out\)，大压缩比下优于逐层 SVD 与参数共享基线。

官方压缩主干（`group.py: Group._init_basis_coefficient`，llama 路径权重先转置）：

```
W_cat = [W_1.T | W_2.T | ... ]             # (in, m·out) 宽矩阵
S    = chol(H).T                           # H 为校准 Hessian（SPD）
w    = S @ W_cat                           # SVD-LLM 白化预处理
u, sigma, v = svd(w)                       # thin SVD
basis = invs @ u[:, :k] @ diag(sigma[:k])  # (in, k) 共享基
C_l   = v.T[:, seg_l][:k, :]               # (k, out_l) 每层系数
```

重建 \( \hat W_l = (B\, C_l)^\top \)；前向 \(y = x\,B\,C_l\)（Basis→Coefficient 链）。官方系数更新（`update_model`，basis 被再训练后重解）：

```
C_cat = (W_cat.T @ H @ B) @ (B.T @ H @ B)^-1    # 最小二乘闭式解，按层列段切
```

**关键恒等**：\(H = S^\top S\) 时该目标等价 \(\lVert S W_{cat} - U_k \Sigma_k C\rVert_F^2\)，其最优解恰为初建 SVD 系数——**两阶段在 SVD basis 下数值恒等**；basis 变化后 LS 重解（正规方程 \(B^\top H (W - BC) = 0\)）。校准复用映射（同输入投影共享 Hessian）：q/v 用 k 的 H、gate 用 up 的 H、其余用自身。SVD/Cholesky 因子全部显式入参。

## 规则

- `mode="scaled_concat"`：各层 \((out_l, in)\) 权重与上三角 Cholesky 因子 \(S\) 入参，返回 \((W_{cat},\ S\,W_{cat})\)，均 \((in,\ m \cdot out)\)——白化作用在**拼接后的宽矩阵**上。
- `mode="build_shared_basis"`：u/σ/invs 与 num_basis 入参，返回 \(B = S^{-1}U_{:k}\,\mathrm{diag}(\sigma_{:k})\ (in, k)\)（σ 降序取前 k 列）。
- `mode="initial_coefficients"`：\(v^\top\ (in, m \cdot out)\) 与 out_dim 入参，逐层列段 \([i \cdot out, (i{+}1) \cdot out)\) 取前 k 行，返回 \([(k, out_l)]\)。
- `mode="least_squares_update"`：闭式解 \((W_{cat}^\top H B)(B^\top H B)^{-1}\) 转置后按层列段切分。
- `mode="reconstruct"` / `mode="forward_shared"`：逐层 \((B\,C_l)^\top\) 与 \(x\,B\,C_l\)。
- `mode="calib_source"`：校准复用映射。
- 全程 float64；不现跑 SVD/Cholesky。

## 示例

\(in=8, out=6\)，组内 \(m=3\) 层，\(k=4\)：

```
scaled_concat([W1,W2,W3], S) -> w_cat (8,18), w = S@w_cat
build_shared_basis(u, sigma, invs, 4) -> B (8,4)
initial_coefficients(v.T, 6, 4) -> [(4,6)]×3
reconstruct(B, C) -> W_hat_l (6,8)；forward_shared(x, B, C) -> x@B@C_l
```

满秩 \(k=in\) 时 \(B_{full} C = W_{cat}\) 精确恢复（\(S\) 可逆）；截断时重建是近似。

## 约束

- 满秩互证：\(B_{full}\ C_{full} = W_{cat}\)、满秩 LS 解 = 全量 \(v^\top\)（两路互证）。
- SVD basis 下 LS 重解与初建系数恒等（\(H = S^\top S\) 恒等）；扰动 basis 下 LS 满足正规方程且与初建可观测不同。
- 前向与重建同源：\(x\,B\,C_l = x\,\hat W_l^\top\)。
- q/v→k、gate→up 的校准映射恒等（同输入投影）。
