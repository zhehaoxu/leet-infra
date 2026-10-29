import math

import torch


def solve(eps_cond, eps_uncond, guidance_scale):
    if not torch.is_tensor(eps_cond) or not torch.is_tensor(eps_uncond):
        raise ValueError("eps_cond 与 eps_uncond 必须是 torch.Tensor")
    if eps_cond.shape != eps_uncond.shape:
        raise ValueError(f"形状不一致：{tuple(eps_cond.shape)} vs {tuple(eps_uncond.shape)}")
    if eps_cond.dim() < 2:
        raise ValueError("至少需要二维输入（batch × 特征）")

    scale = float(guidance_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("guidance_scale 必须是非负有限实数")

    cond = eps_cond.to(torch.float64)
    uncond = eps_uncond.to(torch.float64)
    batch = cond.shape[0]
    lead = (batch,) + (1,) * (cond.dim() - 1)

    flat_cond = cond.flatten(1)
    flat_uncond = uncond.flatten(1)
    pp = torch.sum(flat_cond * flat_cond, dim=1, keepdim=True)
    qq = torch.sum(flat_uncond * flat_uncond, dim=1, keepdim=True)
    pq = torch.sum(flat_cond * flat_uncond, dim=1, keepdim=True)

    # 2×D 矩阵 [cond; uncond] 的 Gram 矩阵 [[pp, pq], [pq, qq]]：最大特征值
    # 对应的特征向量给出第一右奇异方向在 {cond, uncond} 这组（非正交）基下的系数
    half_gap = 0.5 * (pp - qq)
    radius = torch.sqrt(half_gap * half_gap + pq * pq)
    top = 0.5 * (pp + qq) + radius

    first_norm2 = pq * pq + (top - pp) ** 2
    second_norm2 = (top - qq) ** 2 + pq * pq
    use_first = first_norm2 >= second_norm2
    alpha = torch.where(use_first, pq, top - qq)
    beta = torch.where(use_first, top - pp, pq)
    # 两个候选系数同时为零只发生在 pp == qq 且 pq == 0（主方向不唯一），退化为 cond 方向
    tied = (first_norm2 <= 0.0) & (second_norm2 <= 0.0)
    alpha = torch.where(tied, torch.ones_like(alpha), alpha)
    beta = torch.where(tied, torch.zeros_like(beta), beta)

    direction = alpha.view(lead) * cond + beta.view(lead) * uncond
    flat_dir = direction.flatten(1)
    dir_norm2 = torch.sum(flat_dir * flat_dir, dim=1, keepdim=True)
    dir_dot = torch.sum(flat_uncond * flat_dir, dim=1, keepdim=True)
    safe_norm2 = torch.where(dir_norm2 > 0.0, dir_norm2, torch.ones_like(dir_norm2))
    ratio = torch.where(dir_norm2 > 0.0, dir_dot / safe_norm2, torch.zeros_like(dir_dot))

    projected = ratio.view(lead) * direction
    pred = projected + scale * (cond - projected)
    return pred.to(torch.float32)
