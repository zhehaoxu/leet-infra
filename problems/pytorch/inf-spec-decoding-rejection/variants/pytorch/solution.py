import torch


def _sample_from(pi, u):
    c = torch.cumsum(pi, dim=0)
    return int((c < u).sum().item())


def solve(draft_probs, target_probs, draft_tokens, u_accept, u_resample):
    """Speculative decoding 拒绝采样。"""
    gamma = int(draft_tokens.shape[0])
    u_resample = float(u_resample)
    out = []
    rejected = False
    for i in range(gamma):
        xt = int(draft_tokens[i])
        p = float(target_probs[i][xt])
        q = float(draft_probs[i][xt])
        ratio = p / q if q > 0 else float("inf")
        if float(u_accept[i]) < min(1.0, ratio):
            out.append(xt)
        else:
            modified = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
            modified = modified / modified.sum()
            out.append(_sample_from(modified, u_resample))
            rejected = True
            break
    if not rejected:
        out.append(_sample_from(target_probs[gamma], u_resample))
    return torch.tensor(out, dtype=torch.long)
