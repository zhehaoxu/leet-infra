"""GraLoRA 的子块低秩账（granular LoRA）。

回源: SqueezeBits/GraLoRA（官方仓，arXiv:2503.15477 训练细节见论文
arXiv:2505.20355），实现取 peft/src/peft/tuners/gralora/layer.py
（已进入 HuggingFace PEFT 的 tuners/gralora）。model.py/config.py
均为 PEFT 标准脚手架，确定性核心只有 layer.py 两处账：
``update_layer`` 的布局/初始化账与 ``Linear.forward`` 的双 einsum 链。

机制：把权重矩阵 W (out, in) 按输出/输入各划 k 段；LoRA 的秩 r 拆成
gralora_rank = r - hybrid_r（可再留一段标准全秩 LoRA 旁路），每段秩
sub = gralora_rank // k。A 按输入分段存放 (k, in//k, gralora_rank)、
B 按输出分段存放 (k, gralora_rank, out//k)。前向是一个双 einsum 链：
输入第 n 段过 A_n 升到 gralora_rank 维，rank 维切 k 段重排（permute
交换"段"与"输入块"两个 N 轴后复合 reshape），再与 B 的第 j 个输出段
的对应行段内积。等价的分块公式：ΔW 的 (j, n) 块 =
B[j][n*sub:(n+1)*sub, :].T @ A[n][:, j*sub:(j+1)*sub]——块秩只有
sub，参数量却与同秩普通 LoRA 完全相同（rank*(in+out)），这是论文
"granular bottleneck" 的卖点。scaling = alpha / r 用**总秩 r**（含
hybrid 部分），与普通 LoRA 口径一致。

五入口复刻（dropout=0 的 Identity 分支不设参）:
- split_dims: update_layer 的尺寸/校验账——r>0、gralora_rank % k == 0、
  子块尺寸 sub_in/sub_out/sub_rank 与 scaling=alpha/r
- param_layout: 参数布局账——A/B（及 hybrid 旁路）的 shape 与零初始化
  集合（B 与 B_general 全零，A 与 A_general kaiming 非零）
- forward_delta: 官方 einsum 链直译（含 scaling），x 为 (B, L, in)
- block_delta: 等价分块公式显式组装 ΔW (out, in)
- full_forward: gralora 分支 + hybrid 旁路（scaling * Bg(Ag(x))）
"""

import torch


def split_dims(in_features, out_features, r, k, hybrid_r, alpha):
    """update_layer 的尺寸账与校验（dropout 账不设参，恒为 Identity）。"""
    if r <= 0:
        raise ValueError(
            "`r` should be a positive integer value but the value passed is %d" % r)
    gralora_rank = r - hybrid_r  # 分配给 granular 方法的秩
    if gralora_rank % k != 0:
        raise ValueError(
            "r should be divisible by gralora_k, but got %d and %d" % (r, k))
    return {
        "gralora_rank": gralora_rank,
        "sub_rank": gralora_rank // k,
        "sub_in": in_features // k,
        "sub_out": out_features // k,
        "scaling": alpha / r,  # 总秩 r（含 hybrid），与普通 LoRA 同口径
    }


def param_layout(in_features, out_features, r, k, hybrid_r):
    """update_layer 的参数布局账：shapes 与零初始化集合。"""
    dims = split_dims(in_features, out_features, r, k, hybrid_r, alpha=1)
    gralora_rank = dims["gralora_rank"]
    layout = {
        "gralora_A": (k, dims["sub_in"], gralora_rank),
        "gralora_B": (k, gralora_rank, dims["sub_out"]),
    }
    zero_init = ["gralora_B"]  # B 全零 => 训练起点 ΔW = 0；A kaiming 非零
    if hybrid_r > 0:
        layout["gralora_A_general"] = (hybrid_r, in_features)
        layout["gralora_B_general"] = (out_features, hybrid_r)
        zero_init.append("gralora_B_general")
    layout["zero_init"] = zero_init
    return layout


def forward_delta(x, gralora_A, gralora_B, scaling, k):
    """官方 Linear.forward 的 einsum 链直译（dropout=0、不含 hybrid 旁路）。

    x: (B, L, in)；gralora_A: (k, in//k, rank)；gralora_B: (k, rank, out//k)
    返回 scaling * delta，形状 (B, L, k * (out//k))。
    """
    B, L, in_features = x.shape
    N = k
    rank = gralora_A.shape[2]
    sub = rank // N
    h = torch.einsum(
        "blni,nir->blnr",
        x.reshape(B, L, N, in_features // N),
        gralora_A,
    )
    h = (h.reshape(B, L, N, N, sub)
         .permute(0, 1, 3, 2, 4)
         .reshape(B, L, N, N * sub))
    o = torch.einsum("bljr,jro->bljo", h, gralora_B)
    return scaling * o.reshape(B, L, -1)


def block_delta(gralora_A, gralora_B, k):
    """等价分块公式显式组装 ΔW (out, in)。

    ΔW 的 (j, n) 块 = B[j][n*sub:(n+1)*sub, :].T @ A[n][:, j*sub:(j+1)*sub]
    （B 的第 n 行段转置 x A 的第 j 列段，块秩 = sub）。
    """
    N = k
    rank = gralora_A.shape[2]
    sub = rank // N
    sub_in = gralora_A.shape[1]
    sub_out = gralora_B.shape[2]
    dW = torch.zeros(N * sub_out, N * sub_in, dtype=torch.float64)
    for j in range(N):
        for n in range(N):
            brow = gralora_B[j][n * sub:(n + 1) * sub, :].T  # (sub_out, sub)
            aseg = gralora_A[n][:, j * sub:(j + 1) * sub]    # (sub_in, sub)
            dW[j * sub_out:(j + 1) * sub_out,
               n * sub_in:(n + 1) * sub_in] = brow @ aseg.T
    return dW


def full_forward(x, gralora_A, gralora_B, scaling, k, hybrid_r,
                 gralora_A_general=None, gralora_B_general=None):
    """gralora 分支 + hybrid 全秩 LoRA 旁路。

    hybrid_r > 0 时必须提供 A_general (hybrid_r, in) 与
    B_general (out, hybrid_r)，旁路输出同乘 scaling；
    hybrid_r == 0 时两个 general 必须为 None（官方为 nn.Identity 占位）。
    """
    if hybrid_r > 0:
        if gralora_A_general is None or gralora_B_general is None:
            raise ValueError("hybrid_r > 0 需要提供 A_general 与 B_general")
    elif gralora_A_general is not None or gralora_B_general is not None:
        raise ValueError("hybrid_r == 0 时不接受 general 旁路参数")
    result = forward_delta(x, gralora_A, gralora_B, scaling, k)
    if hybrid_r > 0:
        result = result + scaling * (
            x @ gralora_A_general.T @ gralora_B_general.T)
    return result


def solve(mode=None, **kw):
    if mode == "split_dims":
        return split_dims(kw["in_features"], kw["out_features"], kw["r"],
                          kw["k"], kw["hybrid_r"], kw["alpha"])
    if mode == "param_layout":
        return param_layout(kw["in_features"], kw["out_features"], kw["r"],
                            kw["k"], kw["hybrid_r"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["gralora_A"], kw["gralora_B"],
                             kw["scaling"], kw["k"])
    if mode == "block_delta":
        return block_delta(kw["gralora_A"], kw["gralora_B"], kw["k"])
    if mode == "full_forward":
        return full_forward(kw["x"], kw["gralora_A"], kw["gralora_B"],
                            kw["scaling"], kw["k"], kw["hybrid_r"],
                            gralora_A_general=kw.get("gralora_A_general"),
                            gralora_B_general=kw.get("gralora_B_general"))
    raise ValueError("unknown mode: %s" % mode)
