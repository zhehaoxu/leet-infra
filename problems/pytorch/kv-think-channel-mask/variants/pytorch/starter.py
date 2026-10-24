import torch


def solve(queries, keys, compression_ratio):
    """ThinK：通道 energy = mean(q^2) * mean(k^2)，最小 n_pruned 通道置 0。"""
    # TODO
    raise NotImplementedError
