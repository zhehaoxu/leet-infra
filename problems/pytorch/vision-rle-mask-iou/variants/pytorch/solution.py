"""参考答案：COCO RLE 编解码与游程归并 IoU（列优先、不展开稠密掩码）。"""

import numbers

import torch


def _canonical_counts(flat):
    """把列优先压平后的 0/1 序列压成规范 RLE（首项为背景，可为 0）。"""
    vals, cnts = torch.unique_consecutive(flat, return_counts=True)
    cnts = cnts.to(torch.int64)
    if int(vals[0]) == 1:
        cnts = torch.cat([torch.zeros(1, dtype=torch.int64, device=cnts.device), cnts])
    return cnts


def _merge_intersection(counts_a, counts_b):
    """双指针归并两份交替游程，累加同为前景的长度。"""
    la = [int(v) for v in counts_a.tolist()]
    lb = [int(v) for v in counts_b.tolist()]
    ia = ib = 0
    rem_a, rem_b = la[0], lb[0]
    va = vb = 0  # 游程值：偶数下标为 0（背景），奇数下标为 1（前景）
    inter = 0
    while ia < len(la) and ib < len(lb):
        take = rem_a if rem_a < rem_b else rem_b
        if va == 1 and vb == 1:
            inter += take
        rem_a -= take
        rem_b -= take
        if rem_a == 0:
            ia += 1
            if ia < len(la):
                rem_a = la[ia]
                va = ia & 1
        if rem_b == 0:
            ib += 1
            if ib < len(lb):
                rem_b = lb[ib]
                vb = ib & 1
    return inter


def solve(mask, counts, size):
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask 必须是 torch.Tensor")
    if mask.dim() != 2:
        raise ValueError(f"mask 必须是 (H,W) 二维，got {tuple(mask.shape)}")
    if mask.dtype not in (torch.bool, torch.uint8):
        raise ValueError(f"mask dtype 必须是 bool 或 uint8，got {mask.dtype}")
    if bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError("mask 的元素只能是 0/1")
    h, w = int(mask.shape[0]), int(mask.shape[1])
    if h < 1 or w < 1:
        raise ValueError(f"mask 的 H/W 必须 >= 1，got {(h, w)}")

    if not isinstance(counts, torch.Tensor):
        raise ValueError("counts 必须是 torch.Tensor")
    if counts.dim() != 1 or counts.numel() == 0:
        raise ValueError(f"counts 必须是非空一维张量，got {tuple(counts.shape)}")
    if counts.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"counts dtype 必须是 int64/int32，got {counts.dtype}")
    if bool((counts < 0).any()):
        raise ValueError("counts 不能含负值")
    if counts.numel() > 1 and bool((counts[1:] == 0).any()):
        raise ValueError("counts 除首项外不能出现 0（非规范游程）")
    cb = counts.to(torch.int64)
    if int(cb.sum()) != h * w:
        raise ValueError(f"counts 元素和必须等于 H*W={h * w}，got {int(cb.sum())}")

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("size 必须是长度 2 的整数序列 (H, W)")
    sh, sw = size[0], size[1]
    if not all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in (sh, sw)):
        raise ValueError("size 的两项必须是整数")
    sh, sw = int(sh), int(sw)
    if sh < 1 or sw < 1:
        raise ValueError(f"size 的 H/W 必须 >= 1，got {(sh, sw)}")
    if (sh, sw) != (h, w):
        raise ValueError(f"size {(sh, sw)} 与 mask.shape {(h, w)} 不一致")

    flat = mask.permute(1, 0).reshape(-1)  # 列优先：先走完整列
    counts_a = _canonical_counts(flat)
    inter = _merge_intersection(counts_a, cb)
    area_a = int(mask.sum())
    area_b = int(cb[1::2].sum())
    union = area_a + area_b - inter
    iou = torch.tensor(inter / union if union > 0 else 0.0, dtype=torch.float64)
    return (iou, torch.tensor(inter, dtype=torch.int64),
            torch.tensor(union, dtype=torch.int64), counts_a)
