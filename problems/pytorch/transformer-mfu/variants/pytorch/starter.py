import torch


def solve(b, s, layers, hidden, ffn, peak_tflops, step_ms):
    # TODO: 估算 Transformer 训练 step 的 FLOPs 与 MFU，返回 (forward_flops, step_flops, mfu)。
    #   P_layer = 4*hidden*hidden + 2*hidden*ffn            （每层 matmul 参数量）
    #   c       = 2*P_layer + 4*s*hidden                    （每 token 每层前向 FLOPs）
    #   forward_flops = b*s*layers*c                        （整数）
    #   step_flops    = 3*forward_flops                     （前向+反向，整数）
    #   achieved = step_flops/(step_ms/1000); mfu = (achieved/1e12)/peak_tflops
    raise NotImplementedError
