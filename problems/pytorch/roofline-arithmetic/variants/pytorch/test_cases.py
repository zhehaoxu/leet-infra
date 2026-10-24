def _ref(flops, nbytes, peak_flops, bandwidth):
    """独立重算：直接用 min 包络式，判定用 ridge 比较，避免复用 solve 的分支。"""
    ai = float(flops) / float(nbytes)
    ridge = float(peak_flops) / float(bandwidth)
    attainable = min(float(peak_flops), ai * float(bandwidth))
    mem = ai < ridge
    return ai, ridge, attainable, mem


def _close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    got = solve(1e9, 1e7, 2e14, 1e12)
    want = (100.0, 200.0, 100000000000000.0, True)
    assert len(got) == 4
    for g, w in zip(got[:3], want[:3]):
        assert _close(g, w), (got, want)
    assert got[3] is True, got


def check_compute_bound(solve):
    ai, ridge, attainable, mem = solve(1e12, 1e7, 2e14, 1e12)
    assert _close(ai, 1e5) and _close(ridge, 200.0)
    assert _close(attainable, 2e14), attainable
    assert mem is False, (ai, ridge, mem)


def check_ridge_boundary_is_compute(solve):
    # AI == ridge 归 compute-bound
    ai, ridge, attainable, mem = solve(1e9, 5e6, 2e14, 1e12)
    assert _close(ai, 200.0) and _close(ridge, 200.0)
    assert mem is False, mem
    assert _close(attainable, 2e14), attainable


def check_matches_ref(solve):
    cases = [(1e9, 1e7, 2e14, 1e12), (1e12, 1e7, 2e14, 1e12),
             (3e6, 2e6, 1e15, 5e11), (1e15, 1e10, 3e14, 2e12),
             (1000.0, 10.0, 1e6, 1e3)]
    for c in cases:
        got = solve(*c)
        want = _ref(*c)
        for g, w in zip(got[:3], want[:3]):
            assert _close(g, w), (c, got, want)
        assert got[3] == want[3], (c, got, want)


def check_attainable_le_peak(solve):
    for c in [(1e9, 1e7, 2e14, 1e12), (1e18, 1e6, 2e14, 1e12)]:
        _, _, attainable, _ = solve(*c)
        assert attainable <= c[2] + 1e-6, (c, attainable)


check_example.example = True
