"""参考答案：全局池化 —— 对空间维整体归约，keepdim 保持 (N,C,1,1)。"""

import torch


def solve(x: torch.Tensor, mode: str = "avg") -> torch.Tensor:
    dims = tuple(range(2, x.dim()))
    if mode == "avg":
        return x.mean(dim=dims, keepdim=True)
    if mode == "max":
        return x.amax(dim=dims, keepdim=True)
    raise ValueError(f"未知 mode: {mode}（应为 'avg' 或 'max'）")
