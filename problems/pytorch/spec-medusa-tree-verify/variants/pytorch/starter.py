import torch


def solve(cands, P, eps, u_bonus):
    """Medusa 候选树验证：把所有候选路径的前缀去重成树；一条路径被接受 iff 父链每层
    P[i-1][path[i-1]]>=eps；取被接受的最深路径（同深取字典序最小），再从 P[D] 按
    CDF 反查 u_bonus 追加 bonus token。返回 torch.long 一维张量。"""
    # TODO
    raise NotImplementedError
