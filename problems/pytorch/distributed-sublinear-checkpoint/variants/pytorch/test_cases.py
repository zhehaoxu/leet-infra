# 独立重算参考：把每个 k 的 peak 单位数先全列出来取最小值，再回头找达到该最小值的最小 k，
# 与参考解「升序扫描 + 严格小于才更新」的结构不同（一个两趟、一个个贪心更新）。
def _oracle(n, a):
    units = [(-(-n // k) + k) for k in range(1, n + 1)]
    mn = min(units)
    best_k = 1 + units.index(mn)
    return (best_k, mn * a, n * a)


def check_example(solve):
    assert solve(9, 1) == (3, 6, 9), solve(9, 1)
    assert solve(100, 1_000_000) == (10, 20_000_000, 100_000_000), solve(100, 1_000_000)


def check_matches_oracle(solve):
    for n in (1, 2, 3, 4, 5, 7, 9, 10, 16, 17, 50, 100, 999, 4096):
        for a in (1, 3, 4_000_000):
            assert solve(n, a) == _oracle(n, a), (n, a, solve(n, a), _oracle(n, a))


def check_perfect_square_root(solve):
    # n=m^2 时最优间隔应为 m=√n，峰值=2m
    for m in (2, 3, 4, 5, 8, 10, 32):
        best_k, min_peak, _ = solve(m * m, 1)
        assert best_k == m, (m, best_k)
        assert min_peak == 2 * m, (m, min_peak)


def check_sublinear_beats_storeall(solve):
    # 深网络：检查点峰值严格优于全存基线，且不大于全存
    for n in (4, 9, 16, 100, 1000, 40000):
        _, min_peak, store_all = solve(n, 1)
        assert min_peak <= store_all, (n, min_peak, store_all)
    for n in (16, 100, 1000, 40000):
        _, min_peak, store_all = solve(n, 7)
        assert min_peak < store_all, (n, min_peak, store_all)


def check_scale_linear(solve):
    # act_bytes 线性缩放：best_k 不变，两个字节量都按 a 等比放大
    for n in (9, 100, 37):
        k1, p1, s1 = solve(n, 1)
        k2, p2, s2 = solve(n, 12345)
        assert k2 == k1, (n, k1, k2)
        assert p2 == p1 * 12345 and s2 == s1 * 12345, (n, p1, p2, s1, s2)


check_example.example = True
