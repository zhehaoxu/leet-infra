import torch

CODES = {
    4: [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    3: [-4.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 4.0],
}
MAXQ = {4: 12.0, 3: 8.0}


def solve(W, bits):
    """AFPQ：正负两侧各用一个浮点 scale，在 FP4/FP3 码表上做最近码取整。"""
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (3, 4):
        raise ValueError("bits 只能是 3 或 4")
    if not isinstance(W, torch.Tensor) or W.dim() != 2 or not W.is_floating_point():
        raise ValueError("W 必须是二维浮点张量")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 不能为空")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("W 必须有限")

    x = W.to(torch.float64)
    code = torch.tensor(CODES[bits], dtype=torch.float64)
    half = MAXQ[bits] / 2

    zero = torch.zeros((), dtype=torch.float64)
    xmin = torch.minimum(x.amin(dim=1), zero)
    xmax = torch.maximum(x.amax(dim=1), zero)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, -torch.ones_like(xmin), xmin)
    xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)

    scale_pos = xmax / half
    scale_neg = xmin.abs() / half

    x_pos = x.clamp(min=0)
    x_neg = x.clamp(max=0)
    safe_pos = torch.where(scale_pos > 0, scale_pos, torch.ones_like(scale_pos))
    safe_neg = torch.where(scale_neg > 0, scale_neg, torch.ones_like(scale_neg))
    qp = x_pos / safe_pos.unsqueeze(1)
    qn = x_neg / safe_neg.unsqueeze(1)

    q_pos = code[(qp.unsqueeze(-1) - code).abs().argmin(dim=-1)]
    q_neg = code[(qn.unsqueeze(-1) - code).abs().argmin(dim=-1)]
    q_pos = torch.where((scale_pos > 0).unsqueeze(1), q_pos, torch.zeros_like(q_pos))
    q_neg = torch.where((scale_neg > 0).unsqueeze(1), q_neg, torch.zeros_like(q_neg))

    dequant = q_pos * scale_pos.unsqueeze(1) + q_neg * scale_neg.unsqueeze(1)
    return {
        "scale_pos": scale_pos.to(torch.float32),
        "scale_neg": scale_neg.to(torch.float32),
        "q_pos": q_pos.to(torch.float32),
        "q_neg": q_neg.to(torch.float32),
        "dequant": dequant.to(torch.float32),
    }
