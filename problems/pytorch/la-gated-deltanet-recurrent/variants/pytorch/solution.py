import torch


def solve(q, k, v, beta, g, scale=None, initial_state=None):
    """逐行转写 naive_recurrent_gated_delta_rule（见 starter 注释）。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    qf, kf, vf, betaf, gf = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)
    ]
    h = torch.zeros(B, H, K, V).to(torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    qf = qf * scale

    o = torch.zeros(B, H, T, V).to(torch.float32)
    for i in range(T):
        b_q = qf[:, :, i]
        b_k = kf[:, :, i]
        b_v = vf[:, :, i].clone()
        h = h.clone() * gf[:, :, i].exp()[..., None, None]
        b_beta = betaf[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    return {
        "o": o.transpose(1, 2).contiguous(),
        "final_state": h,
    }
