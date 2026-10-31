import torch


def solve(q, k, request_mask, window=16, tau=1.0, recent=16, budget=32):
    """Q-Trust 两跳注意力传播打分与恒保尾部选 key（CPU 参考口径，纯 PyTorch）。
    q:(N,H,D)、k:(N,KV,D) 同 dtype（float32/float64），NHD，H = KV * G；request_mask:(N,) bool 请求段。
    first_hop:(KV,N)=最后 window 行因果注意力按 GQA 组求和；query_weight:(KV,N)：请求段原始权重恒为
    1、其余取 s(q)，归一化到 Σw=1 后把 w < tau/N 的 query 置 0（请求段永不剔除；tau=0 全保留）；
    kept_query:(M,) int64 升序=各头保留 query 的并集；score:(KV,N)=Σ_{q>=i} w(q) Σ_{h∈group} A_h(q,i)；
    selected:(KV,budget) int64 升序=[0,N-recent) 按分数降序取 budget-recent 个（同分取小下标）+ 末尾
    recent 个恒保 key。内部 float64，返回输入 dtype。返回 dict：first_hop/query_weight/kept_query/
    score/selected。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
