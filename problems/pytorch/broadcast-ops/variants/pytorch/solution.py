"""参考答案：广播实战 —— unsqueeze 对齐维度做逐元素运算。"""

import torch


def solve(a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor:
    if op == "row_scale":
        return a * b[:, None]  # (B,) -> (B,1) 广播到每列
    if op == "col_bias":
        return a + b[None, :]  # (D,) -> (1,D) 广播到每行
    if op == "outer":
        return a[:, None] * b[None, :]
    raise ValueError(f"unknown op: {op!r}")
