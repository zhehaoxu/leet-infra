import torch


def solve(input_ids, beam_scores, next_log_probs, num_beams):
    """Beam search 单步扩展 + 回退。"""
    V = int(next_log_probs.shape[1])
    scores = beam_scores.unsqueeze(-1) + next_log_probs
    flat = scores.reshape(-1)
    topk = torch.topk(flat, int(num_beams), dim=-1)
    idx = topk.indices
    new_scores = topk.values
    prev_beam = torch.div(idx, V, rounding_mode="floor")
    new_tok = idx % V
    seq = input_ids[prev_beam]
    new_ids = torch.cat([seq, new_tok.unsqueeze(-1)], dim=-1)
    return new_ids.long(), new_scores
