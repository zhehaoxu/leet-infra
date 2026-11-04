# SVD-LLM 白化域截断压缩

## 题意

普通 SVD 压缩有个反直觉的问题：**截断较小的奇异值反而可能带来更大的压缩损失**（激活存在 outlier 时，权重的谱与损失不成正比）。SVD-LLM（ICLR 2025）的解法是先做 truncation-aware 数据白化——把权重搬到白化域再做 SVD，让「截断小奇异值」与「低压缩损失」直接挂钩。

给你四个输入（校准累积 H 的 Cholesky 因子 L 由上游 profiling 产出，不在判题域）：

- `weight`：待压缩权重 W，形状 (M, N)；
- `scaling_diag_matrix`：白化矩阵 L（(N, N)，下三角 Cholesky 因子）；`direct_update=True` 时不传；
- `x`：校准输入 (P, N)，`y`：稠密模型对应输出 (P, M)。

## 你要实现的管线

### 1. 白化域 SVD

```
W_scale = W @ L
U, S, Vᵀ = svd(W_scale)          # full_matrices=False
```

### 2. 参数预算截断

```
k = int(M·N·ratio / (M+N))        # 压缩后参数量 ≈ ratio 倍
k < 1 → 报错（k 上界无需校验：ratio ≤ 1 时 k 恒 < min(M,N)）
```

### 3. √Σ 因子拆分（v 侧带 L⁻¹ 补偿）

```
Σ_k = diag(S[:k]),  √Σ_k = sqrt(Σ_k)
svd_u = U[:, :k] @ √Σ_k
svd_v = √Σ_k @ (Vᵀ[:k, :] @ L⁻¹)   ← truncation-aware 核心：右乘 L⁻¹ 把白化抵消回去
direct_update=True 时跳过白化：svd_v = √Σ_k @ Vᵀ[:k, :]（且不需要 L）
new_w = U[:, :k] @ Σ_k @ (Vᵀ[:k, :] @ L⁻¹)
```

L 非满秩时官方防御：`+1e-6·I` 后再求逆。

### 4. 闭式更新（layer-wise closed-form update）

```
x' = (x @ svd_v_coreᵀ) @ Σ_k      # svd_v_core = Vᵀ[:k,:] @ L⁻¹（白化路径）
updated_uᵀ = lstsq(x', y).solution
```

### 5. fasterprune 因子与相对误差

```
append_u = updated_uᵀᵀ @ √Σ_k
append_v = √Σ_k @ svd_v_core
err(Ŵ) = ‖y − x@Ŵᵀ‖_F / ‖y‖_F     # error 用 new_w，updated_error 用 append_u@append_v
```

## 输入约束

- `weight` / `x` / `y` 均为 2 维浮点张量，形状满足 x (P, N)、y (P, M)；
- 白化路径下 `scaling_diag_matrix` 必须是 (N, N) 2 维浮点张量；`direct_update=True` 时不得传入；
- `ratio` 是 (0, 1] 内的有限数值（bool 不算数）。

返回 `dict(k, svd_u, svd_v, new_w, updated_u_t, append_u, append_v, error, updated_error)`。

## 提示

- 白化路径下 `svd_u @ (svd_v @ L) ≈ W @ L` 的 k 秩最优逼近——这就是「截断小奇异值 ⇔ 低损失」的实现载体；
- 闭式更新后误差应不劣于截断误差（`updated_error ≤ error`）；
- H 的累积与 Cholesky 在上游 profiling 阶段，L 直接作为输入。
