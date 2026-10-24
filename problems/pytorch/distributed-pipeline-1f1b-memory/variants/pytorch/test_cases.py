def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立参考：逐段算 1F1B 在飞数 min(m, p-i)，取 max/min；GPipe 每段 m
def _ref(L, p, m, act):
    lps = L // p
    per_mb = float(lps) * float(act)
    profile = [min(m, p - i) for i in range(p)]
    return per_mb * m, per_mb * max(profile), per_mb * min(profile)


# 样例用例
def check_example(solve):
    g, w, b = solve(32, 4, 8, 1e6)
    assert _close(g, 6.4e7), g
    assert _close(w, 3.2e7), w
    assert _close(b, 8.0e6), b


def check_m_le_p_no_worst_saving(solve):
    # m<=p：最重段与 GPipe 相同（无省），但最轻段仍只存 1 份
    for m, p in [(2, 4), (4, 4), (3, 8)]:
        g, w, b = solve(16, p, m, 1000.0)
        assert _close(w, g), (m, p, g, w)
        assert _close(b, g / m), (m, p, b, g)   # best = lps*act*1, g=lps*act*m


def check_reduction_ratio(solve):
    # m>p：最重段 / GPipe == p/m
    for L, p, m, act in [(32, 4, 8, 1e6), (24, 6, 30, 2e5), (48, 8, 64, 1e3)]:
        g, w, b = solve(L, p, m, act)
        assert _close(w / g, p / m), (L, p, m, w, g)
        assert w <= g + 1e-9, (w, g)


def check_worst_ge_best(solve):
    for L, p, m, act in [(32, 4, 8, 1e6), (16, 2, 2, 100.0), (12, 3, 20, 50.0)]:
        g, w, b = solve(L, p, m, act)
        assert w >= b - 1e-9, (w, b)


def check_single_stage(solve):
    # p=1：只有一个段，1F1B 令其在飞数=1（最重=最轻），GPipe 仍存 m 份
    g, w, b = solve(8, 1, 5, 1000.0)   # lps=8, per_mb=8000
    assert _close(g, 40000.0), g        # 8000*5
    assert _close(w, 8000.0), w         # 8000*min(5,1)
    assert _close(b, 8000.0), b


def check_matches_ref(solve):
    for L, p, m, act in [(32, 4, 8, 1e6), (32, 8, 4, 1e6), (12, 3, 1, 7.0),
                         (48, 6, 12, 3e5), (16, 4, 40, 1.0)]:
        got = solve(L, p, m, act)
        ref = _ref(L, p, m, act)
        assert len(got) == 3
        for x, y in zip(got, ref):
            assert _close(x, y), (L, p, m, act, got, ref)


check_example.example = True
