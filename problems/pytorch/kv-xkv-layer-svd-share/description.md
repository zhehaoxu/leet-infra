# KV Cache：xKV 跨层共享奇异向量

## 题目描述

相邻层的 key 高度相关，xKV（2503.18893）据此把低秩压缩的对象从单层换成**一组层**：把组内若干层的 K 沿通道轴拼起来做**一次** SVD，得到的左奇异向量 U 整组共用，每层只留自己那一段系数 \(S\cdot V\)。组内层数 `layer_group_size` 默认 1（退化成逐层 SVD），xKV 用 2 或 4；K 的秩默认 256，V 的秩默认 768。

关键在 SVD 的**朝向**。上游先把单层 key 从 `[bsz, H, S, D]` 转置成 `[bsz, S, H*D]`，组内 `torch.cat(..., dim=2)` 得到 `[bsz, S, gs*H*D]` 再送进 SVD：

- 行是**位置**，列是「层 → head → dim」展平后的**拼接通道**；
- 因此 U 是位置侧因子 `[S, rank]`，对组内所有层是同一份——这就是「共享」的由来；
- 每层的 `S·V` 只是 `[rank, gs*H*D]` 里属于自己的那段列切片，reshape 成 `[H, rank, D]` 存起来；
- 重建是 `U @ SV_l`，逐层一次小矩阵乘，不用还原整组宽度。

秩是对**拼接后的矩阵**截断的：`r = min(rank_k, S, gs*H*D)`。它可以超过单层的通道数 `H*D`——加宽层组、同步加倍秩，正是 xKV 的等压缩比对照口径。

三个容易写错的细节：

1. 拼接列序是「层在前、层内 head-major」，不是按 head 跨层交错；
2. 层数必须被 `layer_group_size` 整除：上游只在 `layer_idx % gs == gs-1` 时才触发 SVD，尾组永远等不到这次合并（`U_k` 又按 `num_layers // gs` 分配），等于静默丢层；
3. 上游用随机化的 `torch.svd_lowrank`，结果不可复现（代码注释原文 "Have deterministic issue but faster"）。本题改用精确的 `torch.linalg.svd`，并钉死一个符号约定。

## 任务

实现 `solve(k, layer_group_size=2, rank_k=256)`：

- `k`：`(num_layers, S, H, D)` float32/float64，逐层 K（batch 恒为 1），四维都须 ≥ 1；
- `layer_group_size`、`rank_k`：正整数（bool 不算）。层数不被整除抛 `ValueError`。
- 组数 \(G=\texttt{num\_layers}/\texttt{gs}\)，有效秩 \(r=\min(\texttt{rank\_k}, S, \texttt{gs}\cdot H\cdot D)\)。
- 内部 float64，返回输入 dtype；不得修改 `k`。

**符号约定（确定性）**：对每组里 U 的每一列，若绝对值最大的分量为负就整列取反，同时把 `S·V` 的对应行一起取反；绝对值并列时以**最小行号**那个分量为准。这样 `u @ coef` 不变，而 U 本身逐位可复现。

返回 4 键字典：

- `u`：`(G, S, r)`，每组共享的位置侧奇异向量；
- `coef`：`(num_layers, H, r, D)`，第 l 层的 `S·V` 切片 `[l*H*D, (l+1)*H*D)` 重排成 `[H, r, D]`；
- `recon`：`(num_layers, S, H, D)`，等于 `u[l//gs] @ coef[l]`；
- `err`：`(G,)`，每组 `‖A - U_r·S_r·V_r‖_F`。

## 示例

```
layer_group_size = 2, rank_k = 1        # L=2 → 一个组
k[0] = [[2, 0], [0, 0]]                 # S=2, H=1, D=2
k[1] = [[0, 0], [0, 1]]

A = [k[0] | k[1]] = [[2, 0, 0, 0],      # (S=2, gs*H*D=4)，列序 [层][head][dim]
                     [0, 0, 0, 1]]
sigma = [2, 1]                          # 两层各占一个正交方向
r = min(1, 2, 4) = 1                    # 只留 sigma=2 那一列
u_r = [[1], [0]]        SV_r = [[2, 0, 0, 0]]
coef[0] = [[2, 0]]                      # (H=1, r=1, D=2)：前 H*D 个分量
coef[1] = [[0, 0]]                      # 后 H*D 个分量，这一层没分到秩预算
recon  = [[[2, 0], [0, 0]], [[0, 0], [0, 0]]]
err    = ||A - u_r @ SV_r||_F = sigma_2 = 1.0
```

秩预算是**整组**的：`k[1]` 那一层的方向奇异值更小，就被同一个 `rank_k=1` 挤掉了。把 `rank_k` 调到 2，`err` 归零、`coef[1] = [[0, 0], [0, 1]]`——不存在「每层各自 rank_k」这回事。

## 约束

- **秩按拼接后的宽度截断**：`r` 的上限是 `min(S, gs*H*D)`，可以大于 `H*D`。逐层 SVD 永远拿不到这多出来的秩。
- **共享的是 U，不是 S·V**：`u` 只有 `G` 份，`coef` 有 `num_layers` 份。测试用「组内两层相同」验证两块系数逐元素相等，用「S=1」把列序钉死（此时 `u=[1]`，`coef` 必须逐元素等于原 K）。
- **残差是 Eckart–Young 的尾巴**：精确 SVD 截断下 `err = sqrt(Σ_{j>r} σ_j²)`，随 `rank_k` 单调不增，秩吃满时为 0。上游同一个仓库的 `test_svd_reconstruction_accuracy.py` 也是拿这个量当标尺。
- **符号不影响判分，但必须钉死**：测试只比 `recon`/`err` 与 `u` 的列符号约定，不比 numpy 的原始 U——同奇异值内的旋转是 gauge，符号翻转会让 U 和 `S·V` 同时变号。
- **不要对整块层做一次 SVD**：组与组之间必须互不影响，改第 0 组不能动到第 1 组的 `u`。
- **空序列、非四维、整型 dtype、`layer_group_size` 不是 int** 一律 `ValueError`。
