import torch
import torch.nn.functional as F


def solve(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """逐行转写 selective_scan_ref（实数 A；B/C 支持静态 [D,N] 与变体 [B,N,L]）。"""
    if A.is_complex():
        raise ValueError("complex A is not supported in this task")
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()

    x = torch.zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    return {"out": out, "last_state": last_state.to(dtype=dtype_in)}
