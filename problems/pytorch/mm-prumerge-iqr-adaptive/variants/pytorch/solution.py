"""Reference solution: LLaVA-PruMerge IQR-adaptive token count."""

import numpy as np
import torch


def solve(cls_attn):
    if not isinstance(cls_attn, torch.Tensor):
        raise ValueError("cls_attn must be a torch tensor")
    if cls_attn.dtype not in (torch.float32, torch.float64):
        raise ValueError("cls_attn must be float32 or float64")
    if cls_attn.dim() != 2:
        raise ValueError("cls_attn must be 2-D [B,N]")
    batch, tokens = cls_attn.shape
    if batch < 1 or tokens < 1:
        raise ValueError("empty batch or token dimension")
    if not torch.isfinite(cls_attn).all():
        raise ValueError("cls_attn must be finite")

    # Upstream `outlier_dectection`: flatten the WHOLE batch, cast to float32
    # BEFORE the percentile pass, Tukey 1.5 upper tail, strictly-greater count.
    flat = cls_attn.to(dtype=torch.float32).cpu().numpy().reshape(-1)
    q1 = float(np.percentile(flat, 25))
    q3 = float(np.percentile(flat, 75))
    upper = q3 + 1.5 * (q3 - q1)
    outlier_indices = np.where(flat > upper)[0]
    ratio = len(outlier_indices) / len(flat)

    keep = int(tokens * ratio)
    if keep == 0:
        pruned = (
            torch.arange(tokens, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )
        return ratio, torch.empty((batch, 0), dtype=torch.int64), pruned

    # Same selection conventions as order 536: stable descending sort resolves
    # ties as "value descending, index ascending", then re-sort ascending.
    order = torch.argsort(cls_attn, dim=1, descending=True, stable=True)
    kept = torch.sort(order[:, :keep], dim=1).values
    mask = torch.ones((batch, tokens), dtype=torch.bool)
    mask.scatter_(1, kept, False)
    # stable argsort puts the 0-masked (kept, ascending) first, so the pruned
    # ascending complement occupies the LAST tokens-keep columns
    pruned = torch.argsort(mask.to(torch.int64), dim=1, stable=True)[:, keep:]
    return ratio, kept.contiguous(), pruned.contiguous()
