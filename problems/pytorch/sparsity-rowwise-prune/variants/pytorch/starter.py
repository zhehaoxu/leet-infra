import torch


def solve(weight, sparsity):
    """逐行(局部)幅值剪枝：对每行独立把该行 |w| 最小的 round(sparsity*in) 个置零（并列按列索引升序先剪）。
    weight float32 2-D (out,in)。返回同形状。禁止 torch.nn.utils.prune。"""
    # TODO
    raise NotImplementedError
