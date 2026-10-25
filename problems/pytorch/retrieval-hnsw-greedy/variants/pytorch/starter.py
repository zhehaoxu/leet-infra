import torch


def solve(points, adj, entry, query):
    """HNSW 单层贪心搜索：从 entry 出发，每步跳到「比当前更近」的
    最近邻居，无则停。平方欧氏，邻居等距取小下标，-1 为占位。
    返回 0 维 int64（停止时节点下标）。"""
    # TODO
    raise NotImplementedError
