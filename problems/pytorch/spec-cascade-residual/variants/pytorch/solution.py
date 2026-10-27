import torch


def solve(p, qs):
    """逐级 min(q, rem) 填坑；填不满的残余质量由 target 重新采（内部 float64）。"""
    if p.ndim != 1:
        raise ValueError("p 必须是一维分布")
    if qs.ndim != 2:
        raise ValueError("qs 必须是 (L, T) 二维")
    vocab = int(p.numel())
    levels = int(qs.shape[0])
    if vocab < 1:
        raise ValueError("词表大小必须 >= 1")
    if int(qs.shape[1]) != vocab:
        raise ValueError("p 与 qs 的词表大小必须一致")

    target = p.to(torch.float64)
    drafts = qs.to(torch.float64)
    if bool((target < 0).any()) or bool((drafts < 0).any()):
        raise ValueError("概率不能为负")
    if abs(float(target.sum()) - 1.0) > 1e-5:
        raise ValueError("p 必须归一化")
    if levels > 0 and bool(((drafts.sum(dim=1) - 1.0).abs() > 1e-5).any()):
        raise ValueError("qs 的每一行都必须归一化")

    rem = target.clone()
    hit = torch.zeros_like(target)
    accepted = torch.zeros(levels, dtype=torch.float64)
    for level in range(levels):
        take = torch.minimum(drafts[level], rem)      # 本级只能填「还没解释」的那部分
        hit += take
        rem -= take
        accepted[level] = take.sum()
    fallback = float(rem.sum())                       # 填不满的质量退回 target 重采
    out = hit + fallback * target

    return (out.to(torch.float32), accepted.to(torch.float32),
            torch.tensor(fallback, dtype=torch.float32))
