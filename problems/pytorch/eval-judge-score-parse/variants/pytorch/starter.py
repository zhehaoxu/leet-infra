import torch


def solve(judgments, mode="pairwise", tie_delta=0.1, baseline=None):
    # TODO: 解析 judge 输出里的 "[[a, b]]" / "[[r]]" 分数，
    #       折成 g1 / g2 两侧判分，再做一致性判定与胜率聚合
    raise NotImplementedError
