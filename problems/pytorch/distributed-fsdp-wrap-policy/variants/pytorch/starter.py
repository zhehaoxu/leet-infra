def solve(param_sizes, wrap_size, prefetch_distance):
    """Toy adjacent-layer packing / 玩具相邻层分块与集合通信计数。

    Inputs / 输入:
        param_sizes: list[int], possibly empty; each size is in [1, wrap_size].
            可为空列表；每层参数量为正整数，不超过 wrap_size。
        wrap_size: positive int, maximum parameters per block / 每块参数量上限。
        prefetch_distance: positive int, resident consecutive blocks INCLUDING
            the current block, clipped to remaining blocks in each traversal.
            含当前块的连续驻留块数，按剩余块数截断；前向和反向使用同一窗口。
        All integers are Python ints, excluding bool / 整数均为 Python int，不含 bool。

    Returns / 返回:
        (n_blocks, block_sizes, peak_params, comm_rounds):
        n_blocks: int, number of blocks / 块数。
        block_sizes: list[int], block parameter counts in layer order / 按层顺序的块大小。
        peak_params: int, transient full parameters only; excludes permanent
            shards, gradients and activations; no forward/backward overlap.
            仅临时完整参数的峰值，不含常驻分片、梯度、激活；前反向不重叠。
        comm_rounds: int, legacy name for COLLECTIVE CALLS: forward all-gather,
            backward all-gather and gradient reduce-scatter, three per block.
            沿用旧字段名，实际为集合通信调用次数，每块三次，不是网络轮次。
        Empty input returns (0, [], 0, 0) / 空列表返回 (0, [], 0, 0)。

    Raises / 异常:
        ValueError for any invalid input type or range above, even when the
        layer list is empty / 类型或取值不满足上述约束时抛出，包括空列表时的非法参数。
    """
    raise NotImplementedError
