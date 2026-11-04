"""AdaMSS (official HF PEFT tuners/adamss, NeurIPS 2025;
jzheng20/NeurIPS2025-AdaMSS) -- adaptive multi-subspace scatter ledger.

Dispatch by the ``mode`` keyword:

- mode="build_scatter_index": the update_layer scatter account
  (kwargs: seg {group: [component indices]}, train_idx [trained group
  ids]) -> the out-row indices concatenated in TRAINED-subspace order;
  untrained groups leave their rows at zero in every delta
- mode="rank_clamp": the fixed-rank actual-rank account
  (kwargs: seg_lens [int], subspace_rank int) ->
  per-subspace [max(1, min(L, subspace_rank))]; the >=1 floor is the
  official guard for empty segments / zero rank
- mode="forward_delta": the adamss forward adapter path
  (kwargs: x (..., in), newB (r, in[+1]), A_list [(rank_i, r)],
  B_list [(seg_len_i, rank_i)], scatter_index, out_dim,
  with_bias=False) -> with_bias appends a ones column to the input
  (folded through newB's bias column), projected = x_aug @ newB.T,
  per-subspace h = projected @ A_i.T then y = h @ B_i.T, concat and
  scatter into (..., out_dim); the output dim stays out_dim
- mode="get_delta_full": the merge trunk shared by
  get_delta_weight / get_delta_bias
  (kwargs: newB, A_list, B_list, scatter_index, out_dim) ->
  combined = cat([B_i @ A_i], axis=0) @ newB scattered into the
  zero-init (out, in[+1]) augmented delta; rows outside scatter_index
  stay zero
- mode="split_delta": the weight/bias split
  (kwargs: dW_full (out, in[+1]), with_bias) -> (weight, bias) where
  with_bias gives weight = dW_full[:, :-1] and bias = dW_full[:, -1]
  (the augmented ones column), otherwise the full matrix and a zero
  bias vector

SVD/KMeans/PCA/QR are all done offline; seg, the trained-subspace
choice, A_i/B_i and newB are explicit inputs -- only the deterministic
scatter accounts are replicated.
"""

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
    raise NotImplementedError


def rank_clamp(seg_lens, subspace_rank):
    raise NotImplementedError


def forward_delta(x, newB, A_list, B_list, scatter_index, out_dim,
                  with_bias=False):
    raise NotImplementedError


def get_delta_full(newB, A_list, B_list, scatter_index, out_dim):
    raise NotImplementedError


def split_delta(dW_full, with_bias):
    raise NotImplementedError


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
