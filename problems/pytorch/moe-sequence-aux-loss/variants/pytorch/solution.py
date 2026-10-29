import torch


def solve(scores, selected_ids, alpha=1.0):
    b, t, e = scores.shape
    k = selected_ids.shape[-1]
    probabilities = scores / scores.sum(dim=-1, keepdim=True)
    p = probabilities.mean(dim=1)
    counts = torch.zeros((b, e), dtype=scores.dtype, device=scores.device)
    counts.scatter_add_(1, selected_ids.reshape(b, -1),
                        torch.ones((b, t * k), dtype=scores.dtype, device=scores.device))
    f = counts * (e / (t * k))
    per_sequence = alpha * (f * p).sum(dim=-1)
    sequence_loss = per_sequence.mean()
    batch_loss = alpha * (f.mean(dim=0) * p.mean(dim=0)).sum()
    return f, p, per_sequence, sequence_loss, batch_loss
