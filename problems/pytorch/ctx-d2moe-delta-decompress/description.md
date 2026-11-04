# 上下文压缩：D2-MoE Delta 解压缩（Fisher 合并 + 白化低秩）

## 题目描述

**D²-MoE**（Delta Decompression for MoE-based LLMs Compression，arXiv:2502.17298，官方仓 `lliai/D2MoE`）利用专家权重的高度相关性做零训练压缩：把每个专家权重分解为**共享 base 权重 + 专家独有 delta**——base 用 Fisher 信息加权合并捕获共享成分，delta 按低秩性质用白化 SVD 压缩并可跨专家共享 V 因子。本题钉 delta 解压缩的确定性矩阵账；base 侧的半动态结构化剪枝依赖动态 probe 输入，不在账内。

**Fisher 合并**（`merge_experts` fisher 分支，w1/w3/w2 各自独立）：

```
coef_j = fisher_j * freq_j / Σ_l (fisher_l * freq_l)
W_base = Σ_j coef_j * W_j
delta_j = W_j - W_base        # fisher=0 的专家系数为 0，不贡献 base
```

**参数预算秩**（`svd_delta`）：\(k = \lfloor r \cdot c \cdot ratio / (r + c) \rfloor\)——\((r{+}c) \cdot k \approx ratio \cdot r \cdot c\) 是「压缩后参数量占比」映射；ratio=1 时 \(k = rc/(r{+}c)\) 仍 < min(r,c)，**不存在满秩重建**。

**svdllm 白化 SVD + √Σ 拆分**：

```
W_s = M @ scale;  SVD -> u_k, Σ_k, vt_k
svd_u = u_k @ diag(sqrt(Σ_k))
svd_v = diag(sqrt(Σ_k)) @ vt_k @ inv(scale)
svd_u @ svd_v = (M @ scale)_k @ inv(scale)    # 白化域截断重建去白化
```

**共享 V**（`delta_share_V`）：组内 delta 按行堆叠 \((m \cdot r, c)\) 一次 SVD，\(V = \mathrm{diag}(\sqrt{\Sigma_k})\,vt_k\)，各专家只留行切片 \(U_j = us[j \cdot r:(j{+}1)r, :k]\,\mathrm{diag}(\sqrt{\Sigma_k})\)——满足 \(U_j V = [\,Us\,\mathrm{diag}(\Sigma_k)\,vts_k\,]_j\)（堆叠低秩重建的行切片）。

**合并前向**（SwiGLU）：

```
gate = x @ (W1_base + U1_j V1).T；up = x @ (W3_base + U3_j V3).T
h = relu(gate) * up；out = h @ (W2_base + U2_j V2).T
```

路由权重与 top-k 选择沿用原 MoE，在 adapter 外乘。SVD 因子全部显式入参。

## 规则

- `mode="fisher_merge"`：weights/fisher/freq 入参，返回 (base, deltas, coef)；coef 归一化、fisher=0 系数为 0。
- `mode="delta_rank"`：shape 与 ratio 入参，返回 \(\lfloor rc \cdot ratio/(r{+}c) \rfloor\)。
- `mode="ta_svd_split"`：白化域 SVD 因子与 inv(scale) 入参，返回 (√Σ 拆分的 svd_u, svd_v)。
- `mode="shared_v_split"`：行堆叠 SVD 因子入参，返回 (V_shared, 各专家 U 行切片)。
- `mode="forward_merged"`：base 三权重与三对 delta 因子入参，返回 SwiGLU 合并前向。
- 全程 float64；不现跑 SVD。

## 示例

\(in=6, h=8\)，2 专家，freq=[0.7,0.3]，w1 的 fisher=[1.5,0.5]：

```
coef = [1.5·0.7, 0.5·0.3] / (1.05+0.15) = [0.875, 0.125]
delta_rank([8,6], 1.0) = int(48/14) = 3        # 参数预算，非满秩
ta_svd_split: svd_u (8,k) @ svd_v (k,6) = (Δ@scale)_k @ inv(scale)
shared_v_split: U_j V = [Us diag(Σ) vts]_j 行切片
```

## 约束

- 白化抵消：√Σ 拆分乘积 = 白化域截断 SVD 重建右乘 inv(scale)。
- 共享 V 恒等：\(U_j V\) 逐段等于堆叠矩阵低秩重建的对应行切片。
- 合并前向与「base+delta 重建权重」的普通 SwiGLU 同源（互证）。
- fisher=0 的专家 delta 仍保留（base 不含它，信息在 delta 路径）。
