import torch

FP32_MAX = float(torch.finfo(torch.float32).max)


def _new_scale(reduced, prev_scale, fp8_max, margin):
    """复刻 _default_sf_compute：算 sf → 不可用 amax 回退旧 scale → 最后夹到 FP32 上限。"""
    sf = (torch.tensor(fp8_max, dtype=torch.float32) / reduced) / float(2 ** margin)
    usable = bool(reduced > 0.0) and bool(torch.isfinite(reduced))
    if not usable:
        sf = prev_scale.clone()
    clamped = bool(torch.isinf(sf))
    if clamped:
        sf = torch.tensor(FP32_MAX, dtype=torch.float32)
    return sf, (not usable), clamped


def solve(amax_seq, history_len=3, fp8_max=448.0, margin=0, algo="max", init_scale=1.0):
    if not isinstance(history_len, int) or isinstance(history_len, bool) or history_len < 1:
        raise ValueError("history_len 必须是 >=1 的整数")
    if algo not in ("max", "most_recent"):
        raise ValueError("algo 必须是 max 或 most_recent")
    if not isinstance(margin, int) or isinstance(margin, bool) or margin < 0:
        raise ValueError("margin 必须是 >=0 的整数")
    if not (fp8_max > 0.0) or not torch.isfinite(torch.tensor(fp8_max)):
        raise ValueError("fp8_max 必须是有限正数")
    if not (init_scale > 0.0) or not torch.isfinite(torch.tensor(init_scale)):
        raise ValueError("init_scale 必须是有限正数")

    hist = torch.zeros(history_len, dtype=torch.float32)
    scale = torch.tensor(float(init_scale), dtype=torch.float32)

    scales, next_scales, amax_used, snaps = [], [], [], []
    kept, clamped = [], []
    for raw in amax_seq.tolist():
        # 1) 本步 cast 用旧 scale，观测到的 amax 写进槽 0
        hist[0] = float(raw)
        snaps.append(hist.clone())
        # 2) 按 algo 归约历史
        reduced = hist.max() if algo == "max" else hist[0].clone()
        # 3) 算新 scale（可能回退、可能被夹）
        new_scale, was_kept, was_clamped = _new_scale(reduced, scale, fp8_max, margin)
        scales.append(scale.clone())
        next_scales.append(new_scale.clone())
        amax_used.append(reduced.clone())
        kept.append(was_kept)
        clamped.append(was_clamped)
        scale = new_scale
        # 4) 历史左移一格并清零槽 0
        if history_len > 1:
            hist = torch.roll(hist, -1, 0)
        hist[0] = 0.0

    t = len(scales)
    return {
        "scales": torch.stack(scales).double() if t else torch.zeros(0, dtype=torch.float64),
        "next_scales": torch.stack(next_scales).double() if t else torch.zeros(0, dtype=torch.float64),
        "amax_used": torch.stack(amax_used).double() if t else torch.zeros(0, dtype=torch.float64),
        "histories": torch.stack(snaps).double() if t else torch.zeros((0, history_len), dtype=torch.float64),
        "kept": torch.tensor(kept, dtype=torch.bool),
        "clamped": torch.tensor(clamped, dtype=torch.bool),
        "final_scale": float(scale.item()),
        "final_history": hist.double(),
    }
