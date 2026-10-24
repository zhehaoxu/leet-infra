import torch


def solve(a, b):
    # TODO: 逐对 IoU，box = [x1,y1,x2,y2]。交集宽高 clamp_min(0)，
    # IoU = 交集 / (面积a + 面积b - 交集)，并集为 0 时返回 0。支持广播 (...,4)。
    raise NotImplementedError
