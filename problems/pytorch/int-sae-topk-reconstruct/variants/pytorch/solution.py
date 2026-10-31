import torch


def solve(x, W_enc, b_enc, W_dec, b_dec, k):
    """TopK SAE encode/decode 账（见 starter 注释）。"""
    if not (1 <= k <= W_enc.shape[0]):
        raise ValueError("k must satisfy 1 <= k <= F")
    x = x.float()
    pre_acts = (x - b_dec.float()) @ W_enc.float().T + b_enc.float()

    B, F = pre_acts.shape
    topk = torch.topk(pre_acts, k, dim=-1)
    feature_acts = torch.zeros_like(pre_acts)
    feature_acts.scatter_(-1, topk.indices, topk.values)

    recon = feature_acts @ W_dec.float() + b_dec.float()
    recon_error = x - recon
    l0 = (feature_acts != 0).sum(-1).to(feature_acts.dtype)
    mse = (recon_error ** 2).mean(-1)
    max_feat = feature_acts.abs().argmax(-1)

    return {
        "feature_acts": feature_acts,
        "recon": recon,
        "recon_error": recon_error,
        "l0": l0,
        "mse": mse,
        "max_feat": max_feat,
    }
