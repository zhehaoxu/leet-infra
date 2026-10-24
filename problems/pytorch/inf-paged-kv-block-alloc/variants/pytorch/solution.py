import math


def solve(free_blocks, block_size, seq_lens):
    """PagedAttention block 准入分配。"""
    free = [int(x) for x in free_blocks]
    block_size = int(block_size)
    tables = []
    for s in seq_lens.tolist():
        s = int(s)
        need = math.ceil(s / block_size) if s > 0 else 0
        if need <= len(free):
            tables.append(free[:need])
            free = free[need:]
        else:
            tables.append([])
    return tables, free
