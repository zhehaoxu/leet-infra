def solve(param_sizes, wrap_size, prefetch_distance):
    """Toy adjacent-layer packing and FSDP collective-call accounting."""
    if type(wrap_size) is not int or wrap_size <= 0:
        raise ValueError("wrap_size must be a positive int")
    if type(prefetch_distance) is not int or prefetch_distance <= 0:
        raise ValueError("prefetch_distance must be a positive int")
    if not isinstance(param_sizes, list) or any(
        type(s) is not int or not 0 < s <= wrap_size for s in param_sizes
    ):
        raise ValueError("param_sizes must be a list of positive ints <= wrap_size")
    if not param_sizes:
        return 0, [], 0, 0
    block_sizes = []
    cur = 0
    for s in param_sizes:
        if cur + s <= wrap_size:
            cur += s
        else:
            block_sizes.append(cur)
            cur = s
    block_sizes.append(cur)
    n_blocks = len(block_sizes)
    peak = 0
    for i in range(n_blocks):
        window = block_sizes[i:min(i + prefetch_distance, n_blocks)]
        peak = max(peak, sum(window))
    collective_calls = n_blocks * 3
    return n_blocks, block_sizes, peak, collective_calls
