import torch


def solve(P, draft, eps, u_new):
    """Typical acceptance（arXiv:2211.17192 §3.2）：逐 draft token，若 P[t][draft[t]]>=eps 则接受；
    否则从 P[t] 按 CDF 反查 u_new 采样一个 token 替换并停止。全部接受则从 P[gamma] 采样 bonus。
    P:(gamma+1,V) 目标分布, draft:(gamma,) 投机 token。返回 torch.long 一维 token 序列。"""
    # TODO
    raise NotImplementedError
