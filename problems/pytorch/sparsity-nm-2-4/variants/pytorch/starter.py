import torch


def solve(weight):
    """2:4 结构化剪枝：沿最后一维每 4 个连续元素一组，组内保留 |w| 最大的 2 个、其余置零（并列保留更靠前索引）。
    weight float32，最后一维是 4 的倍数。返回同形状。禁止 torch.sparse/cuSPARSELt。"""
    # TODO
    raise NotImplementedError
