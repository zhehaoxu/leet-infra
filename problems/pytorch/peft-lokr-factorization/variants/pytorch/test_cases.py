CASES = [(256, -1), (127, -1), (128, -1), (48, -1), (36, -1),
         (97, -1), (210, -1), (128, 4), (12, 6), (100, 10)]


def _oracle(dimension, factor):
    """独立重算：暴力枚举全部因子对（factor=-1 时取 (m+n, m) 字典序最小且 m<=n）。
    结构异于参考解的单调扫描。"""
    dimension, factor = int(dimension), int(factor)
    if factor > 0:
        if dimension % factor != 0:
            raise ValueError("nope")
        return (factor, dimension // factor)
    cands = [(m, dimension // m) for m in range(1, dimension + 1)
             if dimension % m == 0 and m <= dimension // m]
    return min(cands, key=lambda p: (p[0] + p[1], p[0]))


def check_example(solve):
    assert solve(256, -1) == (16, 16), f"got {solve(256, -1)}"
    assert solve(128, -1) == (8, 16)


check_example.example = True


def check_matches_oracle(solve):
    for d, f in CASES:
        got, exp = solve(d, f), _oracle(d, f)
        assert got == exp, f"factorization({d},{f}): got {got}, expected {exp}"
        assert isinstance(got[0], int) and isinstance(got[1], int), "应返回两个 int"


def check_prime_degenerates(solve):
    assert solve(97, -1) == (1, 97), "质数只能 (1, d)"
    assert solve(2, -1) == (1, 2)


def check_factor_divides_direct(solve):
    """显式分支按原序返回，不做大小交换。"""
    assert solve(128, 4) == (4, 32)
    assert solve(12, 6) == (6, 2), "12=6*2，原序 (6,2) 而非 (2,6)"
    assert solve(100, 10) == (10, 10)


def check_infeasible_raises(solve):
    for d, f in [(128, 5), (12, 0), (12, -2), (0, -1), (-6, -1)]:
        try:
            solve(d, f)
        except ValueError:
            continue
        raise AssertionError(f"factorization({d},{f}) 应抛 ValueError")
