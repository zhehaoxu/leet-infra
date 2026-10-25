def solve(total_activation, shardable_fraction, cp_degree):
    """序列/上下文并行显存-通信账（简化模型）。
    per_rank = (1-f)*A + f*A/C ; comm = 2 * f*A * (C-1)/C，其中 A=total_activation, f=shardable_fraction, C=cp_degree。
    返回 (per_rank, comm) 两个 float 字节数。C==1 时 comm=0.0 且 per_rank==A。"""
    # TODO
    raise NotImplementedError
