# 解题说明 · xKV 跨层共享奇异向量

## 一句话

把一层组内所有层的 K 沿通道轴拼成一个矩阵，只做**一次** SVD：位置侧的 U 整组共享，每层只存自己那段 `S·V`。

## 为什么 U 能共享

单层 key 转置成 `[S, H*D]`（行是位置、列是通道），组内沿 `dim=2` 拼接后是 `[S, gs*H*D]`。它的每一**行**是同一个位置在各层各 head 上的整条描述，所以 SVD 的行空间就是「位置码」：

\[
A = U\,\mathrm{diag}(S)\,V^\top,\qquad A_l = U \cdot SV_l,\quad SV_l = (S V^\top)[:,\ l\,H D : (l+1) H D]
\]

跨层冗余正落在位置码上：U 只存一份（`num_groups * S * rank`），`S·V` 每层一份（`layers * H * D * rank`）——上游 `scripts/calc_kv_comp_rate.py` 的压缩比公式就是这两项之和。

## 实现四步

1. 组内 `(gs, S, H, D) → permute(1,0,2,3) → reshape(S, gs*H*D)`，列序「层 → head → dim」；
2. `torch.linalg.svd(..., full_matrices=False)` 取前 `r = min(rank_k, S, gs*H*D)` 个三元组；
3. `SV = diag(S[:r]) @ Vh[:r]`，按每 `H*D` 列切给一层，reshape 成 `[H, rank, D]`；
4. `recon = U @ SV`，残差取整组的 Frobenius 范数。

## 四个易错口径

1. **秩按拼接后的宽度算**：上限是 `min(S, gs*H*D)`，可以超过单层通道数——这正是「rank 随 group_size 等比放大」能成立的原因。
2. **拼接列序**：上游的每个 key 已是 `[bsz, S, nh*hd]`，所以是「层在前、层内 head-major」。写成按 head 跨层交错，`S=1` 的用例会立刻发现 `coef` 对不上原 K。
3. **尾组**：SVD 只在 `layer_idx % gs == gs-1` 触发，`U_k` 又按 `num_layers // gs` 分配；层数不整除时尾层既没有 U 也没有 `S·V`，全程静默。本题直接抛 `ValueError`。
4. **确定性**：`torch.svd_lowrank` 是随机化算法，同一输入两次结果不同（上游注释自己写着 "Have deterministic issue but faster"）。`linalg.svd` 之后仍有符号自由度，本题钉死「U 每列绝对值最大的分量为正、并列取最小行号」，并同步翻转 `S·V` 的对应行，保证 `U @ SV` 不变。

## 判分点

- `u` 的列正交归一，且逐列满足上面的符号约定；
- `recon` 严格等于 `u @ coef`（gauge 一致性），也对得上 numpy 独立重算的截断重建；
- `err` 等于 Eckart–Young 尾巴 `sqrt(Σ_{j>r} σ_j²)`，随 `rank_k` 单调不增；
- 组与组互不影响：改第 0 组不能动第 1 组的任何输出；组内两层相同时两块 `S·V` 逐元素相等（拼接矩阵是 `[B|B]`，奇异值放大 `√2`）。

## 边界

- `rank_k` 超过 `min(S, gs*H*D)` 时截断到上限，不报错、不补零列；`coef` 的第三维是 `r` 而不是 `rank_k`。`rank_k` 很小时整组只剩一个方向，别的层系数整块归零——这是共享秩预算的必然（题面示例正是这局面）。
- 一次 `linalg.svd` 是 `O(S · (gs*H*D)^2)`，只在 prefill 时做；decode 侧的 `u @ sv` gather-gemm 是 CUDA 系统代码，不在本题范围。

## 出处

`abdelfattah-lab/xKV`（arXiv 2503.18893）:

- `xKV_SR/models/kv_cache_xkv.py:63-79` `fast_svd`：L68 `torch.svd_lowrank(tensor.float(), q=rank)`、L71 `diag_embed(S_trunc) @ Vt_trunc`、L73-74 `A`/`B` 的形状注释；
- 同文件 L196 `combined_key = torch.cat(keys, dim=2)`、L211 组内 `fast_svd`、L215-223 `self.U.append` 与 `SVh_trunc.view(bsz, rank, nh*gs, hd).transpose(1,2)` 后逐层 split；
- 同文件 L1243-1249 缓冲宽度 `nh*hd*gs` 与 `start = (layer_idx % gs) * nh * hd`、L1253 `if layer_idx % self.group_size == (self.group_size - 1)`；
- 同文件 L1238-1239 `U_k = zeros(num_layers // gs, ...)`、`SV_k = zeros(num_layers, nh, hd, rank)`：U 按组、SV 按层的存储账；
- 同文件 L1263-1264 K/V 分别用 `rank_k`/`rank_v`、L1271-1280 逐层切分（K 路径存 `[bsz, nh, hd, rank]`，V 路径存 `[bsz, nh, rank, hd]`，本题采用后者）；
- `README.md:99-100`：`--layer_group_size` 默认 1、xKV 用 2/4；`--rank_k/--rank_v` 默认 256/768 且随组宽等比放大；`scripts/calc_kv_comp_rate.py:24-27` 的显存账；
- `efficiency/tests/test_svd_reconstruction_accuracy.py:36-39`：拿 `sqrt(Σ tail σ²)` 当重建误差标尺（本题 `err` 的口径）。

**本题自己的补充**（上游没写、题面显式钉死）：精确 `torch.linalg.svd` 替代随机化 `svd_lowrank`、符号约定与并列规则、层数不整除直接报错、`err` 作为返回键、`[层][head][dim]` 的展平顺序。

## 相关题目

- `kv-palu-rank-budget`：同样按层组分配秩，但考组间预算怎么分；本题考组内一次 SVD 的分解与共享。
- `kv-lococo-conv-recombine`：跨层共享的是「合并后的一张 K/V」，本题共享奇异向量、每层仍留自己的系数；`kv-wkvd-least-squares-merge` 解最小二乘，本题只做截断 SVD 与切分。
