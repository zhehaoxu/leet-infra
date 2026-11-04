# Zero-Sum SVD 全局奇异分量选择

## 题意

LLM 低秩压缩要给每一层选一个秩。逐层独立决策（每个模块都剪到同一比例）忽略了一个事实：**不同模块的截断损失敏感度差异巨大**，全局统筹分配才能在总参数预算下压低整体损失。Zero-Sum SVD（ICML 2026）的名字来自它的核心选择策略：每接受一个奇异分量的截断就累积一个预测损失变化 δ，并始终偏好正 δ（「回血」的候选），让累计 Σδ 被压在 0 附近——**假设一个模块上的损失增加能被另一个模块上的收益抵消**。

给你一组模块（每个含 `weight` (M, N)、`grad` (M, N)、白化 Cholesky 因子 `L` (N, N) 或 None），以及全局剪枝比例 `global_prune_ratio`、秩下限比例 `keep_rank_ratio`、δ 估计配置（`beta_mixing` / `delta_order`）。

## 你要实现的管线

### 1. 白化域谱 + 梯度投影

```
A = W @ L                        # L 为 None 时 A = W
U, Σ, Vᵀ = svd(A)                # full_matrices=False, 按 Σ 降序重排
H = G @ L⁻ᵀ                      # 梯度白化（triangular solve）
grad_σ = diag(Uᵀ H V)            # 一阶损失敏感度（每个奇异分量一个）
```

### 2. 截断损失估计 δ（tail 候选从最小 σ 起）

```
first_order:  δ = −σ · grad_σ
second_order: δ = −σ · grad_σ + 0.5 · σ² · grad_σ²
beta_mixing:  δ = −sign(g) · |σ|^β · |g|^(1−β)     (β ∈ [0,1])
```

### 3. 亏盈平衡秩约束

```
r* = break_even_rank = max(1, (M·N − 1) // (M + N))
min_rank = max(1, ceil(keep_rank_ratio · r*))
每个模块最多剪 min(M, N) − min_rank 个奇异分量
```

r* 是低秩因子 (M+N)·r 参数量恰好追平稠密 (M·N) 的平衡点：秩高于 r* 时低秩分解不省参数。

### 4. 零和符号平衡选择（核心循环）

```
while dense_removed < stage_budget:
    prefer_positive = (running_sum ≤ 0)
    在偏好符号堆（δ 符号匹配）中找 |δ| 最小的
    模块候选；偏好堆空则落到另一符号堆
    （官方 pop_prefer_then_any 语义）；
    accept：consumed += 1（取该模块当前最小奇异分量），
            running_sum += δ、truncated += σ²
```

`|δ|` 相同时按模块名升序 tie-break（确定性）。所有分量耗尽仍不满预算时停止。

### 5. 参数记账与终止

```
effective_dense_delta(r_now):
    r_now > r*:  0                       # 高于平衡秩, 低秩因子不省参数
    r_now == r*: M·N − r*·(M+N)          # 恰在平衡点, 一次结清全部节省
    r_now < r*:  M+N                     # 平衡点以下, 每降一秩省 M+N
stage_budget = 初始参数总量 − int((1 − ratio) · 初始参数总量)
```

返回 `dict(ranks, keep_masks, truncated_sigma_sq, running_sum, dense_removed, stage_budget, selections)`。

## 输入约束

- `modules` 非空 dict，键为非空字符串，值恰含 `weight`/`grad`/`L` 三个键；`weight`/`grad` 为同形状 2 维浮点张量，`L` 为 None 或 (N, N) 2 维浮点张量；
- `global_prune_ratio` ∈ [0, 1)、`keep_rank_ratio` ∈ [0, 1]、`beta_mixing` ∈ [0, 1] 或 None、`delta_order` ∈ {first_order, second_order}（bool 不算数）。

## 提示

- 偏好堆空时**必须落到另一符号堆**，而不是直接终止——这是官方 `pop_prefer_then_any` 的语义；
- 敏感度 `grad_σ = diag(UᵀHV)` 的 V 来自**白化域** A=W@L 的分解，不是 W 自身；
- H = G@L⁻ᵀ 等价于对 L 解 triangular solve 后转置，用 `solve(L, Gᵀ).t()` 实现。
