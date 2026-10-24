import torch


def _oracle(free_blocks, block_size, seq_lens):
    """独立：整数向上取整 -(-a//b)，用游标指针消费空闲池而非切片。"""
    free = [int(x) for x in free_blocks]
    bs = int(block_size)
    tables = []
    cursor = 0
    for s in seq_lens.tolist():
        s = int(s)
        need = 0 if s == 0 else -(-s // bs)
        if need <= (len(free) - cursor):
            tables.append(free[cursor:cursor + need])
            cursor += need
        else:
            tables.append([])
    return tables, free[cursor:]


def check_example(solve):
    tables, rem = solve([0, 1, 2, 3, 4, 5], 4, torch.tensor([10, 4, 6, 5]))
    assert tables == [[0, 1, 2], [3], [4, 5], []], tables
    assert rem == [], rem


def check_exact_and_zero(solve):
    # block_size=2, lens=[4,0,2] → need=[2,0,1]
    tables, rem = solve([7, 8, 9, 10], 2, torch.tensor([4, 0, 2]))
    assert tables == [[7, 8], [], [9]], tables
    assert rem == [10], rem


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for _ in range(30):
        nb = int(torch.randint(1, 15, (1,)))
        free = list(range(nb))
        torch.manual_seed(100 + _)
        perm = torch.randperm(nb).tolist()
        free = [perm[i] for i in range(nb)]  # 乱序物理块 id
        bs = int(torch.randint(1, 6, (1,)))
        N = int(torch.randint(1, 7, (1,)))
        seq_lens = torch.randint(0, 13, (N,))
        got = solve(free, bs, seq_lens)
        exp = _oracle(free, bs, seq_lens)
        assert got[0] == exp[0], (free, bs, seq_lens.tolist(), got[0])
        assert got[1] == exp[1], (free, bs, seq_lens.tolist(), got[1])


check_example.example = True
