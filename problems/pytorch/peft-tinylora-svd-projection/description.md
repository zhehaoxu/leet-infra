# PEFT：TinyLoRA 冻结 SVD 因子与随机投影组合

## 题目描述

**TinyLoRA**（Learning to Reason in 13 Parameters, arXiv:2602.04118，Meta）把 LoRA-XS 的「冻结 SVD 因子 + 中间小矩阵 R」推到极致：R 不再独立训练，而是写成**固定随机投影矩阵的线性组合** \(R=\sum_i v_i P_i\)——唯一可训练参数是长度 \(u\) 的向量 \(v\)，论文里小到 13 个 bf16 值就能让 Qwen2.5-8B 在 GSM8K 上拿到 91%。官方参考实现 `ANRGUSC/TinyLoRA`，并已通过 PR #3024 进入 HF PEFT（`tuners/tinylora`）。

本题复刻 PEFT `TinyLoraLayer` 的确定性账：

- **SVD 分摊**（`_init_svd`）：\(W=U\,\mathrm{diag}(S)\,V^\top\)，\(\sqrt{S}\) **均分两侧**——\(A=\mathrm{diag}(\sqrt{S_{:r}})\,V^\top_{:r,:}\ (r,d_\text{in})\)、\(B=U_{:,:r}\,\mathrm{diag}(\sqrt{S_{:r}})\ (d_\text{out},r)\)；秩被短边截断 \(r_\text{act}=\min(r,\min(d_\text{out},d_\text{in}))\)。均分使 \(A,B\) 范数平衡。
- **投影初始化**（`_init_projection`）：\(P\in\mathbb{R}^{u\times r\times r}\sim\mathcal{N}(0,1/r)\)（std \(=1/\sqrt r\)，Johnson–Lindenstrauss 惯例），`torch.Generator().manual_seed(base_seed + layer_idx)` 逐层确定；P 是冻结 buffer 不是参数。
- **R 组合**（`_compute_R`）：`einsum("i,ijk->jk", v, P)`；v 零初始化 ⇒ 训练起点 \(\Delta W=0\)。
- **合并/前向**：\(\Delta W = B@R@A\)；前向 delta \(=x\,A^\top R^\top B^\top\)（LoRA-XS 式三次线性：A→R→B），两者数学等价。
- **weight_tying 分组**（model.py）：目标层按注入顺序拿确定层号，`num_groups = max(1, round(n·(1−wt)))`（wt=0 不共享、wt=1 全共享）、`group_size = max(1, n//num_groups)`、`group_idx = min(idx//group_size, num_groups−1)`；同组层共享一个 v——可训练参数 \(=\) num_groups×u。

## 规则

- `mode="svd_split"`：入参 U/S/Vh（满 SVD 因子，离线算好）与 r；返回 \((A,B,r_\text{act})\)。
- `mode="init_projection"`：入参 u、r、seed；返回 P \((u,r,r)\)，同 seed 重放逐位一致。
- `mode="compute_R"`：v 与 P 入参，返回 \(R\ (r,r)\)。
- `mode="delta_weight"` / `mode="forward_delta"`：B、R、A（及 x）入参。
- `mode="tie_groups"`：n 与 wt 入参，返回 `{"num_groups", "group_size", "group_indices"}`。
- 全程 float64；dropout=0。

## 示例

\(d_\text{out}=6, d_\text{in}=8, r=4, u=6, seed=7\)：

```
svd_split(U, S, Vh, 4) -> A:(4,8), B:(6,4), actual_r=4
init_projection(6, 4, 7) -> P:(6,4,4)
compute_R(v, P) -> R:(4,4)
delta_weight(B, R, A) -> dW:(6,8)，且 x @ dW.T == x@Aᵀ@Rᵀ@Bᵀ
tie_groups(6, 0.5) -> {num_groups: 3, group_size: 2, group_indices: [0,0,1,1,2,2]}
```

\(r=8\) 超过短边（如 \(5\times4\) 的权重）时 \(r_\text{act}\) 截断为 4。

## 约束

- SVD 因子全部显式入参（判题不跑 SVD）。
- \(B\) 列范数 \(=A\) 行范数 \(=\sqrt{S_i}\)（均分账的直接推论，测试抽查）。
- 分组账满足：组数 \(=\) 不同 group_idx 个数，可训练参数 \(=\) 组数×u。
