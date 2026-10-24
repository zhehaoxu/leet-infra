"""参考答案：算子融合的显存带宽收益（纯访存 roofline 模型）。

m 个逐元素算子各一读一写 → 非融合流量 2m·N·bytes；融合后只读入一次写出一次 → 2·N·bytes；
在 memory-bound 假设下加速比等于算子数 m。整数用 Python int，speedup 为浮点。
"""

import torch


def solve(n_elems, dtype_bytes, num_ops):
    non_fused = 2 * num_ops * n_elems * dtype_bytes
    fused = 2 * n_elems * dtype_bytes
    saved = non_fused - fused
    speedup = non_fused / fused
    return int(non_fused), int(fused), int(saved), float(speedup)
