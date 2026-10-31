import torch


def solve(q, k, v, beta, g, scale=None, initial_state=None):
    """逐行转写 fla-org/flash-linear-attention 的 naive_recurrent_gated_delta_rule。

    参数
    ----
    q : [B, T, H, K]   查询
    k : [B, T, H, K]   键
    v : [B, T, H, V]   值
    beta : [B, T, H]   delta rule 学习率
    g : [B, T, H]      log 域门控衰减（每步乘 exp(g)）
    scale : None 或 float；None 时取 1/sqrt(K)，先乘 q
    initial_state : None 或 [B, H, K, V]

    返回
    ----
    dict：
        o            [B, T, H, V]   逐步输出
        final_state  [B, H, K, V]   终态

    递推（输入全部 transpose(1,2) 成 [B,H,T,·] 并 float32 化后计算）：
        h = initial_state（缺省 0）
        q = q * scale
        for i in range(T):
            h = h * exp(g[:, :, i])[..., None, None]
            b_v = v[:, :, i] - (h * k[:, :, i][..., None]).sum(-2)   # hᵀk
            b_v = b_v * beta[:, :, i][..., None]
            h = h + k[:, :, i].unsqueeze(-1) * b_v.unsqueeze(-2)     # k ⊗ b_v
            o[:, :, i] = (q[:, :, i] · h)  逐头内积
    """
    raise NotImplementedError
