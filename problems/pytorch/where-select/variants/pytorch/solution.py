"""参考答案：Where 条件选择 —— condition 为真取 x 否则取 y，三者按广播规则对齐。"""

import torch


def solve(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    yt = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    m, a, b = torch.broadcast_tensors(condition.bool(), xt, yt)
    return a * m.to(a.dtype) + b * (~m).to(a.dtype)
