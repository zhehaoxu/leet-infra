"""参考答案：张量形状与索引 —— flip/narrow/concat/reshape。"""

import torch


def solve(x: torch.Tensor, op: str) -> torch.Tensor:
    if op == "flip":
        # x[:, ::-1] 在该 torch 版本不支持负步长；即便支持也是负 stride 视图，
        # torch.flip 直接产出连续内存的新张量
        return torch.flip(x, dims=[-1])
    if op == "narrow":
        return x[:, : x.shape[1] // 2]
    if op == "concat":
        return torch.cat([x, x + 1], dim=1)
    if op == "reshape":
        return x.reshape(x.shape[1], x.shape[0])
    raise ValueError(f"unknown op: {op!r}")
