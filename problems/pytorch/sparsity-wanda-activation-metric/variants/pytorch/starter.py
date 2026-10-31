import torch


def solve(weight, activations, sparsity, prune_n=0, prune_m=0):
    """Wanda 激活加权幅值剪枝（2306.11695，locuslab/wanda 口径）。
    weight:(out,in)、activations:(tokens,in) 同 dtype（float32/float64）。
    metric = |W| * sqrt(mean_tokens(X**2))，按**输出行**独立剪枝：
    非结构化剪该行 metric 最小的 int(in*sparsity) 个（截断取整，非 round）；
    prune_n>0 时按每 prune_m 个连续列分块、块内剪最小的 prune_n 个，且 sparsity 必须为 0.5。
    并列时列索引小者先剪。内部 float64，返回按输入 dtype。
    返回 dict：metric / mask（True=被剪）/ pruned_weight / sparsity（实际稀疏率）。
    禁止 torch.nn.utils.prune；非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
