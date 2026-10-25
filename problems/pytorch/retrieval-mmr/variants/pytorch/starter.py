import torch


def solve(query, docs, lam, k):
    """MMR 贪心重排：首个取 sim(d,q) 最大；其后每步取
    argmax_{d∉S} lam*cos(d,q)-(1-lam)*max_{s∈S} cos(d,s)。
    余弦相似度，零范数记 0，并列取小下标。返回 int64 形状 (k,)。"""
    # TODO
    raise NotImplementedError
