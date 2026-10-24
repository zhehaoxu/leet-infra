import torch


def solve(n_elems, dtype_bytes, num_ops):
    # TODO: 量化算子融合的 HBM 带宽收益，返回 (non_fused_bytes, fused_bytes, saved_bytes, speedup)。
    #   non_fused_bytes = 2*num_ops*n_elems*dtype_bytes   （m 个算子各一读一写）
    #   fused_bytes     = 2*n_elems*dtype_bytes            （整条并一 kernel，只读入+写出各一次）
    #   saved_bytes     = non_fused_bytes - fused_bytes
    #   speedup         = non_fused_bytes / fused_bytes    （memory-bound 下 = num_ops，浮点）
    # 前三项返回整数，speedup 返回浮点。
    raise NotImplementedError
