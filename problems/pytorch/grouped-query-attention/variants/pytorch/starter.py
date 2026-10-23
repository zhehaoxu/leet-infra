import torch


def solve(q, k, v, num_heads, num_kv_heads):
    # TODO: GQA/MQA —— q 有 num_heads 个头、k/v 只有 num_kv_heads 个头，
    # 每 (num_heads // num_kv_heads) 个「连续」query 头共享同一组 kv 头
    # （repeat_interleave 语义，非循环取模）；缩放 1/sqrt(dh)，dh = q末维 // num_heads。
    raise NotImplementedError
