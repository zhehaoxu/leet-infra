"""Reference solution: FastV visual token pruning at layer K (arXiv:2403.06764).

Keeps the official code's rule -- mean over heads of the LAST query row of the
layer-K attention, only the visual span competes, prefix/suffix survive -- and
pins three upstream ambiguities to deterministic orders: ascending index
re-emission, (score desc, index asc) tie-breaking, and a per-row batch axis.
"""

import torch


def _check_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a plain Python int")
    return value


def solve(hidden_states, attention, image_start, image_len, r):
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(attention, torch.Tensor):
        raise ValueError("hidden_states and attention must be tensors")
    if hidden_states.dtype not in (torch.float32, torch.float64):
        raise ValueError("hidden_states must be float32 or float64")
    if attention.dtype is not hidden_states.dtype:
        raise ValueError("attention must share hidden_states' floating dtype")
    if hidden_states.dim() != 3 or attention.dim() != 4:
        raise ValueError("expected hidden_states [B,L,D] and attention [B,H,L,L]")

    batch, seq_length, feature = hidden_states.shape
    heads = attention.shape[1]
    if batch < 1 or seq_length < 1 or feature < 1 or heads < 1:
        raise ValueError("empty batch, sequence, feature or head dimension")
    if attention.shape[0] != batch or attention.shape[2] != seq_length:
        raise ValueError("attention batch/sequence size mismatch")
    if attention.shape[3] != seq_length:
        raise ValueError("attention must be square in its last two dimensions")

    _check_int(image_start, "image_start")
    _check_int(image_len, "image_len")
    if image_start < 0 or image_len < 1 or image_start + image_len > seq_length:
        raise ValueError("the visual span must lie inside the sequence")
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        raise ValueError("r must be a plain Python real number")
    if not 0.0 <= float(r) < 1.0:
        raise ValueError("r is a drop fraction and must satisfy 0 <= r < 1")
    if not torch.isfinite(hidden_states).all() or not torch.isfinite(attention).all():
        raise ValueError("hidden_states and attention must be finite")

    # score[b, i] = mean over heads of the last query row's column inside the span.
    score = attention[:, :, -1, image_start:image_start + image_len].mean(dim=1)
    keep_count = min(max(round(image_len * (1.0 - float(r))), 0), image_len)

    # Stable descending sort keeps equal scores in ascending index order, which
    # pins the undefined torch.topk tie-break; sorting again restores causality.
    ranked = torch.sort(score, dim=-1, descending=True, stable=True).indices
    kept_visual = torch.sort(ranked[:, :keep_count], dim=-1).values + image_start

    prefix = torch.arange(image_start, dtype=torch.long, device=hidden_states.device)
    suffix = torch.arange(
        image_start + image_len, seq_length, dtype=torch.long, device=hidden_states.device
    )
    keep_indexs = torch.cat(
        (
            prefix.expand(batch, image_start),
            kept_visual,
            suffix.expand(batch, seq_length - image_start - image_len),
        ),
        dim=1,
    )
    gather_index = keep_indexs.unsqueeze(-1).expand(batch, keep_indexs.shape[1], feature)
    pruned_hidden_states = hidden_states.gather(1, gather_index)
    position_ids = keep_indexs.clone()
    return pruned_hidden_states, keep_indexs, position_ids, score
