import torch


def _geometric_offsets(max_frames):
    """上游 _build_geometric_offsets：1, 2, 4, ... <= max_frames。"""
    values, step = [], 1.0
    while step <= max_frames:
        values.append(step)
        step *= 2
    return values or [1.0]


def solve(q_mean, q_abs, k, rope_cos, rope_sin, omega, pos, keep,
          current_end_frame, frame_seq_length=1, temporal_count=None,
          offset_max_frames=128, score_aggregation="mean", normalize_scores=True):
    """TriAttention 相位打分 → 逐 (layer,head) z-normalise → 逐 head topk（float64 内部）。"""
    if q_mean.dim() != 4 or q_mean.shape[-1] != 2 or q_abs.dim() != 3 or k.dim() != 4:
        raise ValueError("q_mean:(L,H,F,2)、q_abs:(L,H,F)、k:(L,T,H,2F)")
    layers, heads, freq = q_mean.shape[0], q_mean.shape[1], q_mean.shape[2]
    tokens = k.shape[1]
    if k.shape[0] != layers or k.shape[2] != heads or k.shape[3] != 2 * freq:
        raise ValueError("k 必须是 (L,T,H,2F) 且 L/H/F 与 q_mean 一致")
    if tuple(q_abs.shape) != (layers, heads, freq) or tuple(rope_cos.shape) != (tokens, freq) \
            or tuple(rope_sin.shape) != (tokens, freq) or tuple(omega.shape) != (freq,):
        raise ValueError("q_abs / rope_cos / rope_sin / omega 的形状与 F、T 不匹配")
    if pos.dim() != 1 or pos.shape[0] != tokens or pos.is_floating_point():
        raise ValueError("pos 必须是长度 T 的整数张量")
    floats = (q_mean, q_abs, k, rope_cos, rope_sin, omega)
    if any(t.dtype != q_mean.dtype for t in floats) or q_mean.dtype not in (torch.float32, torch.float64):
        raise ValueError("全部浮点入参必须同 dtype 且为 float32/float64")
    for name, value in (("keep", keep), ("current_end_frame", current_end_frame),
                        ("frame_seq_length", frame_seq_length), ("offset_max_frames", offset_max_frames)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是 int")
    if not 0 <= keep <= tokens:
        raise ValueError("keep 必须在 [0, T]")
    if frame_seq_length < 1 or offset_max_frames < 0:
        raise ValueError("frame_seq_length 必须 >= 1，offset_max_frames 必须 >= 0")
    if temporal_count is None:
        temporal_count = freq - 2 * (freq // 3)
    if not 0 <= temporal_count <= freq:
        raise ValueError("temporal_count 必须在 [0, F]")
    if score_aggregation not in ("mean", "max"):
        raise ValueError("score_aggregation 只能是 mean / max")
    if not isinstance(normalize_scores, bool):
        raise ValueError("normalize_scores 必须是 bool")

    out_dtype = q_mean.dtype
    pairs = k.to(torch.float64).reshape(layers, tokens, heads, freq, 2)
    qr, qi = q_mean[..., 0].to(torch.float64), q_mean[..., 1].to(torch.float64)
    cos = rope_cos.to(torch.float64).view(1, tokens, 1, freq)            # (T,F) → 对齐 T 轴
    sin = rope_sin.to(torch.float64).view(1, tokens, 1, freq)

    # 反解 RoPE：k_pre = k_post * exp(-iθ)
    pre_re = pairs[..., 0] * cos + pairs[..., 1] * sin
    pre_im = pairs[..., 1] * cos - pairs[..., 0] * sin
    q_mag = torch.sqrt(qr * qr + qi * qi)                                # |mean q|
    k_mag = torch.sqrt(pre_re * pre_re + pre_im * pre_im)
    rel_re = qr.unsqueeze(1) * pre_re + qi.unsqueeze(1) * pre_im          # q_mean * conj(k)
    rel_im = qi.unsqueeze(1) * pre_re - qr.unsqueeze(1) * pre_im
    phi = torch.atan2(rel_im, rel_re)                                     # (L,T,H,F)
    amp = q_mag.unsqueeze(1) * k_mag
    extra = (q_abs.to(torch.float64) - q_mag).unsqueeze(1) * k_mag

    frames = torch.div(pos, frame_seq_length, rounding_mode="floor").to(torch.float64)
    offsets = torch.tensor(_geometric_offsets(offset_max_frames), dtype=torch.float64)
    delta_t = (float(current_end_frame) - frames).unsqueeze(1) + offsets   # (T,O)
    mask = torch.zeros(freq, dtype=torch.float64)
    mask[:temporal_count] = 1.0
    shift = delta_t.unsqueeze(-1) * (mask * omega.to(torch.float64))       # (T,O,F)
    phase = shift.view(1, tokens, 1, -1, freq) + phi.unsqueeze(3)          # (L,T,H,O,F)
    base = (amp.unsqueeze(3) * torch.cos(phase)).sum(-1)                   # (L,T,H,O)
    combined = base + extra.sum(-1).unsqueeze(-1)
    agg = combined.max(dim=-1).values if score_aggregation == "max" else combined.mean(dim=-1)
    score = agg.permute(0, 2, 1).contiguous()                              # (L,H,T)

    if normalize_scores:
        mean = score.mean(dim=-1, keepdim=True)
        std = score.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
        normalized = (score - mean) / std
    else:
        normalized = score
    head_score = normalized.mean(dim=0)                                    # (H,T)
    order = torch.argsort(-head_score, dim=-1, stable=True)[:, :keep]       # 同分取小下标
    kept = order.sort(dim=-1).values.to(torch.int64)
    return {
        "phi": phi.permute(0, 2, 1, 3).to(out_dtype),
        "delta_t": delta_t.to(out_dtype),
        "score": score.to(out_dtype),
        "normalized": normalized.to(out_dtype),
        "head_score": head_score.to(out_dtype),
        "kept": kept.contiguous(),
    }
