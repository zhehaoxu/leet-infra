import torch


def solve(x: torch.Tensor, weight=None, bias=None, eps: float = 1e-5) -> torch.Tensor:
    """手写 LayerNorm 前向（沿最后一维，有偏方差）。

    x:     (..., H) 浮点张量（2D/3D 均可），沿最后一维归一化
    weight: (H,) 或 None（等价于全 1）
    bias:  (H,) 或 None（等价于全 0）
    eps:   防除零
    """
    # TODO: 1) mean / 有偏方差（除以 H），keepdim=True 以便广播
    #       2) (x - mu) / sqrt(var + eps)
    #       3) * weight + bias（None 时跳过对应仿射项）
    # 注意：不得 in-place 修改 x，不得使用 F.layer_norm / nn.LayerNorm
    raise NotImplementedError("请实现 solve()")
