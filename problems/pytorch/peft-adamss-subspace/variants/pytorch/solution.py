"""AdaMSS 的子空间散射适配账。

回源: HF PEFT src/peft/tuners/adamss/layer.py
（jzheng20/NeurIPS2025-AdaMSS, NeurIPS 2025; Adaptive Multi-Subspace
orthogonal fine-tuning）。机制：对带 bias 的增广权重矩阵
W' = [W | b] (out, in+1) 做 SVD，冻结右奇异因子投影 newB = V'_r^T
(r, in+1)，可训练部分是左因子 UU (out, r)——但 UU 不直接整体训练，
而是把 r 个 SVD 成分（out 维方向）用 KMeans 聚成 K 组（seg 分段），
每段一个独立低秩子空间 A_i (rank_i, r)、B_i (seg_len_i, rank_i)，
rank_i = max(1, min(seg_len_i, subspace_rank))。不同子空间可以训练
不同秩、也可以整段冻结（train_subspace_index）——多子空间异构秩。

forward 链（有 bias 时输入拼全 1 列，输出维仍是 out_features）:
  x_aug    = [x | 1]                        # (..., in+1)
  projected = x_aug @ newB.T                # (..., r)   冻结投影
  h_i      = projected @ A_i.T              # (..., rank_i)
  y_i      = h_i @ B_i.T                    # (..., seg_len_i)
  cat      = [y_0, y_1, ...]                # (..., sum seg_len)
  delta[..., scatter_index] = cat           # scatter 到 out 维段行

merge 链（get_delta_weight / get_delta_bias 共用主干）:
  combined = cat([B_i @ A_i], axis=0) @ newB   # (sum seg_len, in+1)
  dW_full[scatter_index, :] = combined         # (out, in+1) 增广 delta
  weight delta = dW_full[:, :-1]，bias delta = dW_full[:, -1]
  （bias 是增广全 1 列对应的最后一列，不是额外一行）

KMeans/SVD/PCA/QR 全部离线完成；seg/训练段选择/A_i/B_i/newB 均
为显式入参，锚点只钉确定性矩阵账。

六入口复刻:
- build_scatter_index / rank_clamp / forward_delta /
  get_delta_full / split_delta / solve dispatch
"""

import numpy as np


def build_scatter_index(seg, train_idx):
    """update_layer 末尾的 scatter_index: 按可训练子空间顺序拼接 seg 段索引。

    seg: {i: [成分索引...]}（划分 0..r-1）；train_idx: 训练的组号。
    官方为 torch.cat([seg_result[train_subspace_index[i]] ...])。
    """
    return sum((list(seg[i]) for i in train_idx), [])


def rank_clamp(seg_lens, subspace_rank):
    """固定 rank 模式的实际秩: max(1, min(seg_len, subspace_rank))。

    官方 actual_rank = min(len(seg_indices), subspace_rank)，
    为 0 时置 1（空段或 subspace_rank=0 的护栏）。
    """
    return [max(1, min(L, subspace_rank)) for L in seg_lens]


def forward_delta(x, newB, A_list, B_list, scatter_index, out_dim,
                  with_bias=False):
    """adamss forward 的适配器增量路径。

    x: (..., in)；newB: (r, in[+1]) 冻结投影；A_list[k]: (rank_k, r)；
    B_list[k]: (seg_len_k, rank_k)；输出 (..., out_dim)。
    with_bias 时输入拼全 1 列（用 newB 的 bias 列），输出维不变。
    """
    xx = x
    if with_bias:
        ones = np.ones(x.shape[:-1] + (1,), dtype=x.dtype)
        xx = np.concatenate([x, ones], axis=-1)
    projected = xx @ newB.T
    outs = []
    for A_i, B_i in zip(A_list, B_list):
        h = projected @ A_i.T          # (..., rank_i)
        outs.append(h @ B_i.T)         # (..., seg_len_i)
    cat = np.concatenate(outs, axis=-1)
    delta = np.zeros(x.shape[:-1] + (out_dim,), dtype=x.dtype)
    delta[..., scatter_index] = cat
    return delta


def get_delta_full(newB, A_list, B_list, scatter_index, out_dim):
    """merge 主干: 增广 delta 权重 (out, in[+1])（未剥 bias 列）。

    combined = cat([B_i @ A_i], axis=0) @ newB —— blocks 各自乘 newB
    再散布；零初始化 (out, in+1) 后 dW_full[scatter_index, :] = combined。
    """
    blocks = [B_i @ A_i for A_i, B_i in zip(A_list, B_list)]
    combined = np.concatenate(blocks, axis=0) @ newB
    dW_full = np.zeros((out_dim, newB.shape[1]), dtype=newB.dtype)
    dW_full[scatter_index, :] = combined
    return dW_full


def split_delta(dW_full, with_bias):
    """get_delta_weight / get_delta_bias 的收尾拆分。

    with_bias: weight = dW_full[:, :-1]（out, in），bias = dW_full[:, -1]
    （out,）；无 bias 时 weight 即 dW_full，bias 为全零（out,）。
    """
    if with_bias:
        return dW_full[:, :-1], dW_full[:, -1]
    return dW_full, np.zeros(dW_full.shape[0], dtype=dW_full.dtype)


def solve(mode=None, **kw):
    if mode == "build_scatter_index":
        return build_scatter_index(kw["seg"], kw["train_idx"])
    if mode == "rank_clamp":
        return rank_clamp(kw["seg_lens"], kw["subspace_rank"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["newB"], kw["A_list"], kw["B_list"],
                             kw["scatter_index"], kw["out_dim"],
                             with_bias=kw.get("with_bias", False))
    if mode == "get_delta_full":
        return get_delta_full(kw["newB"], kw["A_list"], kw["B_list"],
                              kw["scatter_index"], kw["out_dim"])
    if mode == "split_delta":
        return split_delta(kw["dW_full"], kw["with_bias"])
    raise ValueError("unknown mode: %s" % mode)
