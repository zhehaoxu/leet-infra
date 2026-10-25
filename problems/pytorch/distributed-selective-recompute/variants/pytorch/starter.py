def solve(mem, cost, budget, min_keep):
    """选择性激活重算：在 Σ_{i∈K} mem[i] ≤ budget 且 |K| ≥ min_keep 下，最小化重算代价
    Σ_{i∉K} cost[i]（等价于最大化 Σ_{i∈K} cost[i]）。返回最小重算代价（int）。
    若不存在可行保留集合（连 mem 最小的 min_keep 个之和都 > budget），抛 ValueError。
    建议 DP：dp[j][m] = 恰好保留 j 块、占用显存 m 时的最大保留 cost，最后取 j≥min_keep、m≤budget 的最大值。"""
    # TODO
    raise NotImplementedError
