# Gated Linear Attention：chunk 并行形式与朴素递推的恒等账

## 题意

Gated Linear Attention（2312.06635，fla-org/flash-linear-attention 落地）在线性注意力上加了逐 key 通道的 log 衰减门 `g`：朴素递推为 `h_t = h_{t-1}·diag(exp(g_t)) + k_t ⊗ v_t`、`o_t = (q_t·scale)·h_t`（scale 缺省 `K^-0.5`，即官方 `naive_recurrent_gla` 口径）。本题要求用**chunk 并行形式**实现同一递推：把 `[0, T)` 切成左闭右开的段，每段先用段内含入 cumsum `S` 把段前状态的一次性贡献与段内两两贡献拆开算，并输出每个 chunk 更新完的状态账本。chunk 并行结果必须与朴素递推逐位对拍。

## 求解

对每个 chunk `[c0, c1)`（长度 l）：

```
S = cumsum(g[c0:c1], dim=0)                  # 段内含入 cumsum，[l, K]
# 段间贡献：读段前状态 h_prev，衰减 exp(S_t)
o[c0:c1] += (q·scale ⊙ exp(S)) @ h_prev
# 段内贡献：A[t,j] = Σ_i q_t,i·k_j,i·exp(S_t,i − S_j,i)，下三角含对角
A = (q·scale ⊙ exp(S)) @ (k ⊙ exp(−S)).T
o[c0:c1] += tril(A) @ v[c0:c1]
# 状态更新：段末衰减（按 key 通道逐行）+ 带剩余衰减的 k 累积
h = exp(S_end)·h_prev（逐行） + (k ⊙ exp(S_end − S)).T @ v
```

关键结构点：

1. **`g` 是 `[T, K]` 逐 key 通道量**：cumsum 与所有 exp 都按通道独立进行，不存在标量衰减。
2. **段末衰减按行缩放**：`exp(S_end)[:, None] * h`——直接 `[K] * [K,V]` 会被 torch 广播到尾维（按列缩放），K=V 时静默出错（本题实测坑）。
3. **对角含在段内项里**：`A[t,t] = q_t·k_t`（`S_t − S_t = 0`），mask 用含对角下三角。
4. **不均匀尾 chunk**：末段长度不足 chunk_size 时 l 取实际值，`S_end = gc[-1]` 语义不变。
5. 全程 float64 计算（题面钉死），保证与朴素递推对拍到 1e-8。
