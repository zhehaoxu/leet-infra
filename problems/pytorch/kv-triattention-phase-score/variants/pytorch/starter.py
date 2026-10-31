import torch


def solve(q_mean, q_abs, k, rope_cos, rope_sin, omega, pos, keep,
          current_end_frame, frame_seq_length=1, temporal_count=None,
          offset_max_frames=128, score_aggregation="mean", normalize_scores=True):
    """TriAttention：逆转 RoPE 取相位 → 几何 Δt 的 cos 加权打分 → 逐 (layer,head) z-normalise → topk。"""
    # TODO
    raise NotImplementedError
