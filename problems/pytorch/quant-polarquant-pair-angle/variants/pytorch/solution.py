"""参考解：PolarQuant 极坐标成对量化。

PolarQuant/models/modeling_llama_polar.py::LlamaPolarGroupAttention.quantize_and_pack_nbit L135-157
PolarQuant/models/kernel4group.py L113-158（作者自备的 PyTorch 参考核，含 bin 中心重构）
"""

import math

import torch


def _is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def solve(keys, group_size=128, rbits=4, tbits=4):
    if not torch.is_tensor(keys) or keys.dim() != 4:
        raise ValueError("keys 必须是四维 (B, H, L, D)")
    if keys.dtype not in (torch.float32, torch.float64):
        raise ValueError("keys 必须是 float32 或 float64")
    batch, head, seq, dim = keys.shape
    if dim % 2:
        raise ValueError("D 必须是偶数：head_dim 按前后两半配对")
    if not _is_pos_int(group_size) or not _is_pos_int(rbits) or not _is_pos_int(tbits):
        raise ValueError("group_size / rbits / tbits 必须是正整数")
    if seq <= 0 or seq % group_size:
        raise ValueError("L 必须是 group_size 的正整数倍")

    x = keys.double()
    g, dh = group_size, dim // 2
    lb = seq // g
    pair = x.reshape(batch, head, lb, g, 2, dh)
    lo, hi = pair[:, :, :, :, 0, :], pair[:, :, :, :, 1, :]
    phi = torch.atan2(hi, lo)
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
    radii = torch.sqrt(lo * lo + hi * hi)

    tmax = phi.amax(-2, keepdim=True)
    tmin = phi.amin(-2, keepdim=True)
    tscale = (tmax - tmin) / 2 ** tbits
    rmax = radii.amax(-2, keepdim=True)
    rmin = radii.amin(-2, keepdim=True)
    rscale = (rmax - rmin) / 2 ** rbits

    # 常数组（span==0）上游会 0/0 变 NaN，本口径直接给码字 0。
    theta = torch.where(tscale > 0, torch.floor((phi - tmin) / tscale), torch.zeros_like(phi))
    rho = torch.where(rscale > 0, torch.floor((radii - rmin) / rscale), torch.zeros_like(radii))
    theta = theta.clamp(0, 2 ** tbits - 1)
    rho = rho.clamp(0, 2 ** rbits - 1)

    phi_hat = (theta + 0.5) * tscale + tmin
    radii_hat = (rho + 0.5) * rscale + rmin
    recon = torch.stack([radii_hat * torch.cos(phi_hat), radii_hat * torch.sin(phi_hat)], dim=-2)

    indices = (rho.long() << tbits) + theta.long()
    out = {"phi": phi, "radii": radii, "theta": theta.long(), "rho": rho.long(),
           "indices": indices, "recon": recon.reshape(batch, head, seq, dim)}
    return {k: v.to(keys.dtype) if v.is_floating_point() else v for k, v in out.items()}
