import torch


def solve(ranks, k):
    """RRF 融合：ranks (L,N) int64，ranks[l,d]=1-based 名次或 0 缺席。
    score(d)=sum_{r>0} 1/(k+r)，按 score 降序返回文档下标 (N,)，并列按idx升序。"""
    # TODO
    raise NotImplementedError
