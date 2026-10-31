import torch


def solve(
    aq,
    ak,
    av,
    bq,
    bk,
    bv,
    is_causal=False,
    attn_mask=None,
    scale=None,
    scale_q=None,
    scale_k=None,
    scale_v=None,
):
    """TPA 张量积注意力（2501.06425，CPU 参考口径）。
    aq:(B,N,H,R)、ak:(B,M,H,S)、av:(B,M,H,S_V) 为逐头因子；
    bq:(B,N,R,D)、bk:(B,M,D,S)、bv:(B,M,E,S_V) 为逐位置、跨 head 共享的因子。
    先对 aq/ak/av 取 sigmoid，再按 R/S/S_V 收缩成 q/k/v；
    默认 scale_q=1/R、scale_k=1/S、scale_v=1/S_V、scale=D^-0.5。
    is_causal=True 用左上对齐的 tril(N,M) 掩码；attn_mask 为 (N,M) 的 bool（True=保留）
    或同 dtype 加法掩码；两者不可同时给出。softmax 沿 M 轴。
    返回 dict：out:(B,N,H,E)、prob:(B,H,N,M)，dtype 同输入。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
