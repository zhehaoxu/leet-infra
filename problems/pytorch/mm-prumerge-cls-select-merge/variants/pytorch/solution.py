"""Reference solution: LLaVA-PruMerge CLS selection plus cosine residual merge."""

import torch
import torch.nn.functional as F


def _as_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a plain Python real number")
    return float(value)


def solve(image_features, cls_attn, reduction_ratio, eps=1e-12):
    if not isinstance(image_features, torch.Tensor) or not isinstance(cls_attn, torch.Tensor):
        raise ValueError("image_features and cls_attn must be tensors")
    if image_features.dtype not in (torch.float32, torch.float64):
        raise ValueError("image_features must be float32 or float64")
    if cls_attn.dtype is not image_features.dtype:
        raise ValueError("cls_attn must share image_features' floating dtype")
    if image_features.dim() != 3 or cls_attn.dim() != 2:
        raise ValueError("expected image_features [B,N,C] and cls_attn [B,N]")

    batch, tokens, channels = image_features.shape
    if batch < 1 or tokens < 1 or channels < 1:
        raise ValueError("empty batch, patch or feature dimension")
    if cls_attn.shape != (batch, tokens):
        raise ValueError("cls_attn must have shape [B,N]")

    ratio = _as_real(reduction_ratio, "reduction_ratio")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("reduction_ratio must satisfy 0 < reduction_ratio <= 1")
    epsilon = _as_real(eps, "eps")
    if not epsilon > 0.0:
        raise ValueError("eps must be positive")
    keep = int(tokens * ratio)
    if keep == 0:
        raise ValueError("int(N * reduction_ratio) must keep at least one patch")
    if not torch.isfinite(image_features).all() or not torch.isfinite(cls_attn).all():
        raise ValueError("image_features and cls_attn must be finite")

    # A stable descending sort makes ties resolve to the lower index; sorting the
    # winners again restores ascending patch order.
    ranked = torch.sort(cls_attn, dim=-1, descending=True, stable=True).indices[:, :keep]
    kept_indexs = torch.sort(ranked, dim=-1).values
    selected = torch.zeros((batch, tokens), dtype=torch.bool, device=image_features.device)
    selected.scatter_(1, kept_indexs, True)
    # ascending stable sort puts the unselected (0) patches first, in index order
    pruned_indexs = torch.argsort(selected.to(torch.int8), dim=-1, stable=True)[:, :tokens - keep]

    kept_features = image_features.gather(
        1, kept_indexs.unsqueeze(-1).expand(batch, keep, channels)
    )
    pruned_features = image_features.gather(
        1, pruned_indexs.unsqueeze(-1).expand(batch, tokens - keep, channels)
    )
    # F.normalize's denominator is max(||x||, eps), so zero features give cosine 0.
    kept_unit = F.normalize(kept_features, p=2, dim=-1, eps=epsilon)
    pruned_unit = F.normalize(pruned_features, p=2, dim=-1, eps=epsilon)
    similarity = torch.bmm(pruned_unit, kept_unit.transpose(1, 2))
    cluster = torch.argmax(similarity, dim=-1)
    probabilities = torch.softmax(similarity, dim=-1)
    column = cluster.unsqueeze(-1)
    weights = torch.gather(probabilities, 2, column).squeeze(-1) * (channels ** -0.5)
    scores = torch.gather(similarity, 2, column).squeeze(-1)
    contributions = pruned_features * weights.unsqueeze(-1)
    merged = kept_features.scatter_add(
        1, column.expand(batch, tokens - keep, channels), contributions
    )
    return merged, kept_indexs, pruned_indexs, cluster, weights, scores
