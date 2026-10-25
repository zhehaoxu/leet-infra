def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算参考：通信量用「环上逐步模拟」——reduce-scatter 与 all-gather 各 C-1 步、每步搬 S/C，
# 累加得到 2*(C-1)*S/C；与参考解直接乘 (C-1)/C 的闭合式结构不同。
def _oracle(total_activation, shardable_fraction, cp_degree):
    A = float(total_activation)
    f = float(shardable_fraction)
    C = int(cp_degree)
    S = A * f                       # 可切分总量
    per_rank = A * (1.0 - f) + S / C
    step = S / C
    comm = 0.0
    for _ in range(C - 1):
        comm += step                # reduce-scatter
    for _ in range(C - 1):
        comm += step                # all-gather
    return per_rank, comm


def check_example(solve):
    pr, comm = solve(1000, 0.8, 4)
    assert _close(pr, 400.0) and _close(comm, 1200.0), (pr, comm)
    pr, comm = solve(8000, 0.75, 8)
    assert _close(pr, 2750.0) and _close(comm, 10500.0), (pr, comm)


def check_matches_oracle(solve):
    cases = [
        (1000, 0.8, 4), (8000, 0.75, 8), (1e9, 0.6, 32),
        (500, 1.0, 5), (500, 0.0, 7), (12345, 0.37, 3),
        (2e6, 0.9, 128), (0, 0.5, 16), (999, 0.5, 1),
    ]
    for A, f, C in cases:
        pr, comm = solve(A, f, C)
        rpr, rcomm = _oracle(A, f, C)
        assert _close(pr, rpr), (A, f, C, pr, rpr)
        assert _close(comm, rcomm), (A, f, C, comm, rcomm)


def check_cp_degree_one_degenerate(solve):
    for A, f in [(1000, 0.8), (7e8, 0.3), (500, 1.0), (500, 0.0)]:
        pr, comm = solve(A, f, 1)
        assert _close(comm, 0.0), (A, f, comm)
        assert _close(pr, A), (A, f, pr, A)


def check_no_shardable_means_no_comm(solve):
    # f=0：没有可切分激活 → 每 rank 都是全量 A、无通信
    pr, comm = solve(1000, 0.0, 8)
    assert _close(pr, 1000.0) and _close(comm, 0.0), (pr, comm)
    # f=1：全部可切分 → per_rank=A/C，comm=2A(C-1)/C
    pr, comm = solve(1000, 1.0, 5)
    assert _close(pr, 200.0), pr
    assert _close(comm, 2 * 1000 * 4 / 5), comm


def check_monotonic_in_cp_degree(solve):
    A, f = 1e6, 0.8
    prev_pr = None
    prev_comm = None
    for C in (1, 2, 4, 8, 16, 64, 256):
        pr, comm = solve(A, f, C)
        if prev_pr is not None:
            assert pr < prev_pr, (C, pr, prev_pr)      # 并行度越高每卡激活越少
            assert comm > prev_comm, (C, comm, prev_comm)  # 但通信越多
        prev_pr, prev_comm = pr, comm


check_example.example = True
