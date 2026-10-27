import torch


def _oracle(w, sens, k, iters):
    """纯 python 标量重算：排序初始化 + 逐点分配 + 加权均值，结构与参考解不同。"""
    n = len(w)
    pairs = sorted((float(w[i]), float(sens[i])) for i in range(n))
    ws = [p[0] for p in pairs]
    cent = [ws[min(int((j + 0.5) * n / k), n - 1)] for j in range(k)]
    for _ in range(iters):
        a = []
        for x in w:
            best, bj = None, 0
            for j, c in enumerate(cent):
                d = abs(float(x) - c)
                if best is None or d < best:
                    best, bj = d, j
            a.append(bj)
        new = list(cent)
        for j in range(k):
            tot, cnt = 0.0, 0.0
            for i in range(n):
                if a[i] == j:
                    tot += float(w[i]) * float(sens[i])
                    cnt += float(sens[i])
            if cnt > 0:
                new[j] = tot / cnt
        cent = sorted(new)
    a = []
    for x in w:
        best, bj = None, 0
        for j, c in enumerate(cent):
            d = abs(float(x) - c)
            if best is None or d < best:
                best, bj = d, j
        a.append(bj)
    return (torch.tensor(cent, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor([cent[j] for j in a], dtype=torch.float32))


def check_example(solve):
    w = torch.tensor([0.0, 1.0, 2.0, 3.0])
    sens = torch.ones(4)
    cent, assign, deq = solve(w, sens, 2, 1)
    # 初始中心 w[1]=1、w[3]=3；2 到两中心等距，并列取小簇 0，故左簇含 {0,1,2}
    assert torch.allclose(cent, torch.tensor([1.0, 3.0]), atol=1e-5), f"got {cent.tolist()}"
    assert assign.tolist() == [0, 0, 0, 1], f"got {assign.tolist()}"
    assert torch.allclose(deq, torch.tensor([1.0, 1.0, 1.0, 3.0]), atol=1e-5), f"got {deq.tolist()}"


check_example.example = True


def check_weighted_centroids(solve):
    """返回的每个中心必须等于其簇内加权均值（按返回的 assign 分组），且升序、assign 与 deq 自洽。"""
    torch.manual_seed(6)
    w = torch.randn(50) * 2
    sens = torch.rand(50) + 0.1
    cent, assign, deq = solve(w, sens, 5, 6)
    assert torch.all(cent[1:] >= cent[:-1]), "中心必须升序"
    assert torch.equal(deq, cent[assign]), "deq 必须等于 centroids[assign]"
    for j in range(5):
        m = assign == j
        if bool(m.any()):
            exp = float((w[m].to(torch.float64) * sens[m].to(torch.float64)).sum()
                        / sens[m].to(torch.float64).sum())
            assert abs(float(cent[j]) - exp) <= 1e-4, f"簇 {j} 中心 {float(cent[j])} != 加权均值 {exp}"


def check_beats_uniform_grid(solve):
    """同 k 下，学出的非均匀格点加权 MSE 应显著低于等距格点（linspace(min,max,k)）。"""
    torch.manual_seed(10)
    w = torch.randn(300) * 2.5
    sens = torch.rand(300) + 0.05
    cent, assign, deq = solve(w, sens, 8, 12)
    wmse = float((((w - deq) ** 2) * sens).sum())
    uni = torch.linspace(float(w.min()), float(w.max()), 8)
    a = (w[:, None] - uni[None, :]).abs().argmin(1)
    uni_mse = float((((w - uni[a]) ** 2) * sens).sum())
    assert wmse < 0.5 * uni_mse, f"加权 k-means {wmse:.3f} 应远小于均匀格点 {uni_mse:.3f}"


def check_uniform_sens_matches_lloyd(solve):
    """sens 全 1 时退化为普通 1D k-means：与标量重算逐值对拍。"""
    torch.manual_seed(15)
    w = torch.randn(40) * 1.5
    sens = torch.ones(40)
    got = solve(w, sens, 4, 8)
    exp = _oracle(w, sens, 4, 8)
    assert torch.allclose(got[0], exp[0], atol=1e-5), f"{got[0].tolist()} vs {exp[0].tolist()}"
    assert torch.equal(got[1], exp[1]), "assign 与重算不符"
    assert torch.allclose(got[2], exp[2], atol=1e-5)


def check_sensitivity_shifts_centroid(solve):
    """敏感度语义：把上半区权重的 sens 抬高，中心应向该区移动（k=2 时对比）。"""
    w = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    base = torch.ones(5)
    hi = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0])
    c0, _, _ = solve(w, base, 2, 3)
    c1, _, _ = solve(w, hi, 2, 3)
    assert float(c1[1]) > float(c0[1]) + 0.3, f"高敏感一侧中心应右移：{c0.tolist()} → {c1.tolist()}"
    assert abs(float(c1[0]) - float(c0[0])) < 1e-6, "低敏感一侧中心不应移动"
    # 非法入参：k 越界、iters<0、sens 负数/全零
    for args in ((0, 1), (9, 1), (2, -1)):
        try:
            solve(w, base, args[0], args[1])
            raise AssertionError(f"args={args} 应抛 ValueError")
        except ValueError:
            pass
    for bad_sens in (torch.full((5,), -1.0), torch.zeros(5)):
        try:
            solve(w, bad_sens, 2, 1)
            raise AssertionError("非法 sens 应抛 ValueError")
        except ValueError:
            pass
