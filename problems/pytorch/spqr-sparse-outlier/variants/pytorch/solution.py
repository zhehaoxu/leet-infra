import torch


def solve(W, tau, bits=8):
    """SpQR：稀疏离群逐元素保留 FP、稠密主体逐行对称 RTN，重构回原形状。"""
    O, C = W.shape
    out_mask = W.abs() > tau                      # (O, C) 逐元素离群
    qmax = 2 ** (bits - 1) - 1
    recon = W.clone()
    for o in range(O):
        non = [j for j in range(C) if not bool(out_mask[o, j])]
        if not non:
            continue
        scale = float(W[o, non].abs().max()) or 1.0
        vals = W[o, non]
        q = torch.clamp(torch.round(vals / scale * qmax), -qmax - 1, qmax)
        recon[o, non] = q / qmax * scale
    return recon, out_mask
