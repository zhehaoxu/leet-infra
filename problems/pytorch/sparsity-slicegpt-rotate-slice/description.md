# 稀疏：SliceGPT 正交旋转-切边步（PCA damping + shortcut 链式记账）

## 题意

SliceGPT 不把权重置零也不量化，而是给整个隐空间换基：用校准信号相关矩阵的特征向量做正交旋转，把隐空间旋到「主成分坐标」后**直接切掉尾部低能量维度**（整体降秩 d → d'），再用 shortcut 旋转矩阵把被切开的相邻块链回一起。实现单层 rotate-and-slice `solve(layer_inputs, mlp_ln_inputs, layer_outputs, w_qkv, w_o, w1, w2, attn_in_dim, attn_out_dim, mlp_out_dim, o_bias=None, ignore_masks=None)`：

```text
# pca (官方 pca_calc, 对三组显式信号各做一次)
H = Σ_batch Σ_token x xᵀ          # mask==0 的 token 先置零
damp = 0.01 · mean(diag(H)); H 对角加 damp   # 防退化
torch.linalg.eigh(H) → 按特征值降序重排列向量 → Q (列 = 主成分)

# 1. Q_in (layer_inputs): 注意力输入旋转
W_qkv ← (W_qkv @ Q_in)[:, :attn_in]     # float64 旋转后转回 float32
attn_shortcut = Q_in.T[:attn_in, :]     # float32

# 2. Q_mid (mlp_ln_inputs): 注意力输出 / MLP 输入旋转
attn_shortcut ← attn_shortcut @ Q_mid[:, :attn_out]
W_o ← (Q_mid.T @ W_o)[:attn_out, :];  b_o ← (Q_mid.T @ b_o)[:attn_out]
mlp_shortcut = Q_mid.T[:attn_out, :]    # float32
W_1 ← (W_1 @ Q_mid)[:, :attn_out]       # mlp 输入维 ≡ attn 输出维

# 3. Q_out (layer_outputs): MLP 输出旋转
mlp_shortcut ← (mlp_shortcut @ Q_out)[:, :mlp_out]
W_2 ← (Q_out.T @ W_2)[:mlp_out, :]
```

返回 `dict(w_qkv, w_o, o_bias, w1, w2, attn_shortcut_q, mlp_shortcut_q, q_in, q_mid, q_out, eig_vals)`——三组降序特征值随权重一并返回（damping 只在对角、可观测）。

## 关键语义

- **切的是秩不是权重值**：W 在旋转坐标系里切行/列，剩余部分原样保留——与 magnitude/Wanda 的逐元素置零、SparseGPT 的逐列重构完全不同族。
- **damping 只加对角**：`H[diag,diag] += damp`，非对角不动（加满 H 会改变特征向量）。
- **eigh 升序 → 降序重排**：argsort 降序同时重排特征值与列向量；切边取的是 `[:, :k]` = **最高能量**主成分。
- **shortcut 两段拼接**：`attn_shortcut = Q_in.T[:attn_in] @ Q_mid[:, :attn_out]`、`mlp_shortcut = (Q_mid.T[:attn_out] @ Q_out)[:, :mlp_out]`——先切 Q_in 的行/Q_mid 的列，matmul 用 float32（Q 先转 dtype）。
- **顺序块维度一致性**：MLP 输入维恒等于注意力输出维（官方 ForwardSlicingScheduler），一个 `attn_out_dim` 管两处。
- **纯函数式**：掩码置零前 clone，入参（信号/权重/掩码）一律不改。

## 判题域约定

- 官方在真实模型上前向取块间信号；判题域 `layer_inputs / mlp_ln_inputs / layer_outputs` 三组 list[(B,S,d)] 显式入参（每组逐 batch 累积 H）。
- 切边维度显式入参（scheduler 不入）；`final_orientation='pca'`（random 分支不入）。
- 权重旋转在 float64 做后转回 float32；shortcut 拼接按官方用 float32。
- 特征分解约定：必须 `torch.linalg.eigh`（同环境同后端符号确定），测试谱非退化（相邻特征值 gap > 1e-3）。
- 官方不校验；判题域补信号组/掩码对齐与 0/1/权重形状/切边维度等显式校验。

## 例子

d=8、attn_in=5、attn_out=6、mlp_out=7、两个 batch 带掩码：三组 PCA 各得 8×8 正交 Q（特征值降序、非退化）；W_qkv 从 (8,8) 切到 (8,5)、W_o (6,8)（含偏置切到 (6,)）、W_1 (16,6)、W_2 (7,16)；attn_shortcut (5,6)、mlp_shortcut (6,7)。掩码 0 的 token 不进 H；Q_in 对 layer_inputs 满足 `H @ q = q·diag(vals)`。

数据规模：d ≤ 32，f ≤ 4d，batch 组数 ≤ 4，权重 float32 值域 [-1, 1]。
