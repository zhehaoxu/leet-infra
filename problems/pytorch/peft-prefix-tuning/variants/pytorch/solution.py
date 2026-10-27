import torch
import math


def solve(P_k, P_v, K, V, Q):
    """Prefix-tuning：虚拟 KV 前缀拼在真实 KV 前面，softmax(Q[P_k;K]^T/√d)[P_v;V]。
    P_k,P_v:(L,d) L>=0；K,V:(S,d)；Q:(Nq,d)。
    返回 (attn:(Nq,d) float32, n_direct:int=2Ld, n_reparam:int=Ld+(d*d+d)+d*2Ld+2Ld)。
    禁止现成 PEFT/transformers 库。"""
    d = K.shape[-1]
    K2 = torch.cat([P_k.to(torch.float64), K.to(torch.float64)], dim=0)
    V2 = torch.cat([P_v.to(torch.float64), V.to(torch.float64)], dim=0)
    scores = Q.to(torch.float64) @ K2.T / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1) @ V2
    L = P_k.shape[0]
    h = d
    n_direct = 2 * L * d
    n_reparam = L * d + (d * h + h) + (h * (2 * L * d) + 2 * L * d)
    return attn.to(torch.float32), int(n_direct), int(n_reparam)
