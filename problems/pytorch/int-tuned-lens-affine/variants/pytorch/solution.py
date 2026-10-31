import torch
import torch.nn.functional as F


def solve(hidden_states, probe_w, probe_b, final_logits, temperature=1.0):
    """tuned lens 仿射探针账（见 starter 注释）。"""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    L, B, D = hidden_states.shape
    V = final_logits.shape[-1]

    hs = hidden_states.to(torch.float64)
    pw = probe_w.to(torch.float64)
    pb = probe_b.to(torch.float64)
    fl = final_logits.to(torch.float64)

    # logits_l = W_l @ h_l + b_l
    layer_logits = torch.einsum('lbd,lvd->lbv', hs, pw) + pb[:, None, :]

    target = fl.argmax(-1)  # [B]
    logp = F.log_softmax(layer_logits / temperature, dim=-1)
    layer_ce = -logp.gather(-1, target[None, :, None].expand(L, B, 1)).squeeze(-1)

    p_final = F.log_softmax(fl / temperature, dim=-1).exp()[None, :, :]  # [1,B,V]
    logp_layer = F.log_softmax(layer_logits / temperature, dim=-1)       # [L,B,V]
    layer_kl = (p_final * (p_final.log() - logp_layer)).sum(-1)

    argmax_match = layer_logits.argmax(-1) == target[None, :]

    probe_rank = torch.tensor([torch.linalg.matrix_rank(pw[l]).item() for l in range(L)],
                              dtype=torch.int64)

    return {
        "layer_logits": layer_logits,
        "layer_ce": layer_ce,
        "layer_kl": layer_kl,
        "argmax_match": argmax_match,
        "probe_rank": probe_rank,
    }
