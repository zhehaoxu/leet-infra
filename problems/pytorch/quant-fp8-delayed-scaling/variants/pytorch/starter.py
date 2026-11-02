import torch


def solve(amax_seq, history_len=3, fp8_max=448.0, margin=0, algo="max", init_scale=1.0):
    # TODO: 复刻 Transformer Engine 的 DelayedScaling per-tensor scale 递推。
    #   每步顺序固定：① 本步 amax 写进 history 槽 0（cast 用的是旧 scale）；② 按 algo 归约
    #   （max = 窗口最大值，most_recent = 槽 0）；③ 算新 scale = (fp8_max / amax) / 2^margin，
    #   但 amax <= 0 或非有限时**保留旧 scale**，算出的 sf 为 inf 时夹到 FP32 上限；
    #   ④ 历史左移一格（history_len == 1 不滚）并把槽 0 清零。
    #   所有 scale 运算在 float32 下进行（amax 入 history 时也会被窄化成 float32）。
    #   返回 {"scales", "next_scales", "amax_used", "histories", "kept", "clamped",
    #         "final_scale", "final_history"}。
    raise NotImplementedError
