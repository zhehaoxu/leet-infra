"""SAM stability scores and the ONNX single-mask selection rule."""

import math

import torch

_FLOAT_DTYPES = (torch.float32, torch.float64)


def _as_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite Python real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def solve(masks, iou_preds, num_points, mask_threshold=0.0, offset=1.0):
    if not isinstance(masks, torch.Tensor) or not isinstance(iou_preds, torch.Tensor):
        raise ValueError("masks and iou_preds must be torch.Tensor")
    if masks.ndim != 4:
        raise ValueError("masks must be 4-D with shape (B, C, H, W)")
    if any(dim <= 0 for dim in masks.shape):
        raise ValueError("every masks dimension must be positive")
    if masks.dtype not in _FLOAT_DTYPES or iou_preds.dtype not in _FLOAT_DTYPES:
        raise ValueError("masks and iou_preds must be float32 or float64")
    if masks.dtype != iou_preds.dtype:
        raise ValueError("masks and iou_preds must share one dtype")
    if tuple(iou_preds.shape) != (masks.shape[0], masks.shape[1]):
        raise ValueError("iou_preds must have shape (B, C) matching masks")
    if type(num_points) is not int or num_points <= 0:
        raise ValueError("num_points must be a positive Python int")
    threshold = _as_real(mask_threshold, "mask_threshold")
    step = _as_real(offset, "offset")
    if step <= 0:
        raise ValueError("offset must be positive")
    if not bool(torch.isfinite(masks).all()) or not bool(torch.isfinite(iou_preds).all()):
        raise ValueError("masks and iou_preds must contain only finite values")

    batch, tokens = masks.shape[:2]
    dtype = masks.dtype
    rows = torch.arange(batch, device=masks.device)

    keep = masks > threshold
    intersections = (masks > (threshold + step)).flatten(2).sum(dim=-1)
    unions = (masks > (threshold - step)).flatten(2).sum(dim=-1)
    areas = keep.flatten(2).sum(dim=-1)

    empty = unions == 0
    safe_unions = torch.where(empty, torch.ones_like(unions), unions)
    stability = intersections.to(dtype) / safe_unions.to(dtype)
    stability = torch.where(empty, torch.zeros_like(stability), stability)

    weights = torch.zeros((tokens,), dtype=dtype, device=masks.device)
    weights[0] = 1000.0
    biased = stability + (float(num_points) - 2.5) * weights
    best_idx = torch.argmax(biased, dim=1)

    return (
        keep[rows, best_idx],
        stability[rows, best_idx],
        stability,
        biased,
        best_idx,
        areas,
        iou_preds[rows, best_idx],
    )
