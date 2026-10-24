import torch


def solve(W, tau, bits=8):
    """LLM.int8()：离群列保留 FP、普通列 INT8 对称量化，重构回原形状。"""
    out_mask = W.abs().amax(dim=0) > tau          # (C,) 离群列
    recon = W.clone()
    non = ~out_mask
    if bool(non.any()):
        sub = W[:, non]
        qmax = 2 ** (bits - 1) - 1
        scale = float(sub.abs().max()) or 1.0
        q = torch.clamp(torch.round(sub / scale * qmax), -qmax - 1, qmax)
        recon[:, non] = q / qmax * scale
    return recon, out_mask
