import torch


def solve(weight, sparsity):
    """全局幅值剪枝：把 |w| 最小的 round(sparsity*N) 个元素置零（并列按展平索引升序先剪）。
    weight 任意形状 float32，sparsity∈[0,1]。返回同形状 float32。禁止 torch.nn.utils.prune。"""
    # TODO
    raise NotImplementedError
