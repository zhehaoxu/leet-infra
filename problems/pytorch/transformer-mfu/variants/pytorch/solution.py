"""参考答案：Transformer 训练 FLOPs 与 MFU 估算。

按标准口径拆每层 matmul 参数量 + 注意力非参数 FLOPs，得前向 FLOPs；step 取 3×（反向≈2×前向）；
再由实测耗时算 achieved、除以峰值即 MFU。整数中间量用 Python int，MFU 为浮点。
"""

import torch


def solve(b, s, layers, hidden, ffn, peak_tflops, step_ms):
    p_layer = 4 * hidden * hidden + 2 * hidden * ffn          # attn qkvo + FFN 上下投影
    c = 2 * p_layer + 4 * s * hidden                          # 每 token 每层前向 FLOPs（含注意力）
    forward_flops = b * s * layers * c                        # 一个 step 前向
    step_flops = 3 * forward_flops                            # 前向 + 反向（反向≈2×前向）
    achieved_tflops = step_flops / (step_ms / 1000.0) / 1e12  # FLOP/s → TFLOP/s
    mfu = achieved_tflops / peak_tflops
    return int(forward_flops), int(step_flops), mfu
