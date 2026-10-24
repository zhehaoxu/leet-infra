import torch
import torch.nn.functional as F


def solve(attn, kernel_size):
    """对齐 kvpress SnapKVPress.score（kv-group 视为 1）。"""
    H, W, P = attn.shape
    scores = attn.mean(dim=-2)  # (H, P)
    scores = F.avg_pool1d(
        scores.unsqueeze(1),
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    ).squeeze(1)  # (H, P)
    m = scores.max().item() + 1.0
    return F.pad(scores, (0, W), value=m)
