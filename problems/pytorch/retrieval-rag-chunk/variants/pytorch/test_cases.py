import torch


def _oracle(num_tokens, chunk_size, overlap):
    step = chunk_size - overlap
    rows = []
    s = 0
    while s < num_tokens:
        rows.append([s + j if s + j < num_tokens else -1 for j in range(chunk_size)])
        s += step
    return torch.tensor(rows, dtype=torch.long) if rows else torch.zeros(0, chunk_size, dtype=torch.long)


def check_example(solve):
    y = solve(7, 3, 1)
    exp = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, -1, -1]]
    assert y.tolist() == exp, f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    for (n, cs, ov) in [(7, 3, 1), (10, 4, 2), (12, 5, 0), (9, 2, 1), (6, 6, 3), (13, 4, 2)]:
        y = solve(n, cs, ov)
        assert torch.equal(y, _oracle(n, cs, ov)), (n, cs, ov, y.tolist())


def check_overlap_zero_hard_split(solve):
    # overlap=0 -> step=chunk_size 无重叠硬切，尾巴块 -1 补
    y = solve(7, 3, 0)
    assert y.tolist() == [[0, 1, 2], [3, 4, 5], [6, -1, -1]], y.tolist()


def check_coverage_no_lost_tokens(solve):
    # 带重叠切块必须覆盖每个 token 至少一次（不丢 token）
    n = 10
    y = solve(n, 4, 2)
    valid = set(int(v) for v in y.flatten().tolist() if v >= 0)
    assert valid == set(range(n)), sorted(valid)


def check_shape_dtype_values(solve):
    n, cs, ov = 13, 5, 1
    y = solve(n, cs, ov)
    step = cs - ov
    exp_nc = len(range(0, n, step))
    assert y.dtype == torch.int64 and y.shape == (exp_nc, cs), (y.dtype, y.shape)
    flat = set(int(v) for v in y.flatten().tolist())
    assert all(v == -1 or 0 <= v < n for v in flat), sorted(flat)
