import torch


def solve(q, k, v, num_heads):
    # TODO: 多查询注意力 (MQA)。q:(B,L_q,hidden)，k/v:(B,L_k,head_dim) 单个共享 KV 头。
    # head_dim = hidden // num_heads；把单 KV 头 expand 到 num_heads 份，逐头做
    #   softmax(q_h @ k^T / sqrt(head_dim)) @ v，再按头拼回 (B, L_q, hidden)。
    # 结果须与 expand 单 KV 头后的 F.scaled_dot_product_attention 一致。
    raise NotImplementedError
