import torch


def solve(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU MLP：down( SiLU(gate(x)) * up(x) )，三条线性均无 bias。

    x           : (..., hidden)
    gate_weight : (inter, hidden)   up_weight : (inter, hidden)   down_weight : (hidden, inter)
    权重按 nn.Linear.weight 的 [out, in] 布局（F.linear(x, W) = x @ W.T）。
    返回 (..., hidden)。SiLU(z) = z * sigmoid(z)。
    """
    # TODO: g=linear(x,gate_w); u=linear(x,up_w); return linear(silu(g)*u, down_w)
    raise NotImplementedError
