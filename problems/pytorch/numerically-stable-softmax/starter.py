import torch


def solve(logits: torch.Tensor, mask=None, dim: int = -1) -> torch.Tensor:
    """数值稳定（可选 masked）的 softmax。

    logits: 任意形状浮点张量
    mask:   与 logits 同形状的 bool 张量（True = 有效位置），或 None
    dim:    归约维度
    """
    # TODO: 1) 沿 dim 求 masked max（无效位置应视为 -inf）
    #       2) 计算 exp(x - max)，无效位置置 0
    #       3) 沿 dim 归一化；整组全部被 mask 时输出全 0（不允许 nan）
    raise NotImplementedError("请实现 solve()")
