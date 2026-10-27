import torch


def solve(P_k, P_v, K, V, Q):
    """Prefix-tuning：虚拟 KV 前缀拼在真实 KV 前面，softmax(Q[P_k;K]^T/√d)[P_v;V]。
    P_k,P_v:(L,d) L>=0；K,V:(S,d)；Q:(Nq,d)。
    返回 (attn:(Nq,d) float32, n_direct:int=2Ld, n_reparam:int=Ld+(d*d+d)+d*2Ld+2Ld)。
    禁止现成 PEFT/transformers 库。"""
    # TODO
    raise NotImplementedError
