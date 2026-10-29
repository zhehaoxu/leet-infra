import torch


def solve(x, gate_w, noise_w, noise, k, training=True):
    experts = gate_w.shape[0]
    if type(k) is not int or not 1 <= k <= experts:
        raise ValueError("k must be an integer between 1 and the expert count")
    logits = x @ gate_w.T
    if training:
        raw_scale = x @ noise_w.T
        scale = torch.logaddexp(torch.zeros_like(raw_scale), raw_scale)
        logits = logits + noise * scale
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :k]
    selected = logits.gather(1, indices)
    weights = torch.softmax(selected, dim=-1)
    gates = torch.zeros_like(logits).scatter(1, indices, weights)
    return gates, indices, logits
