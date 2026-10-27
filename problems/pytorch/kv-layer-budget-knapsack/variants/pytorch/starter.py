import torch


def solve(gain, budget):
    """层间名额背包：总名额 budget 分给各层，最大化收益之和。
    gain:(Lc,M) float32，gain[l][x] 为第 l 层保留 x 个位置的收益（x 取 0..M-1）；
    约束 sum_l x_l <= budget、0 <= x_l <= M-1。
    收益最大者胜；收益相同时取**字典序最小**的分配向量（先比 x_0，再比 x_1 ...）。
    返回 (alloc:(Lc,) int64, total:float32)。
    budget<0、gain 为空或含负值抛 ValueError。"""
    # TODO
    raise NotImplementedError
