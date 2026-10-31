import torch


def _nearest_in_coset(x, shift):
    """在 shift 陪集里枚举 y 与 y±e_i（y=round(x-shift)），返回最近点与 SSE。"""
    z = [v - shift for v in x]
    y = [round(v) for v in z]
    cands = [y[:]]
    for i in range(8):
        for d in (-1, 1):
            yy = y[:]
            yy[i] += d
            cands.append(yy)
    best, best_pt = None, None
    for c in cands:
        if sum(c) % 2 != 0:
            continue
        pt = [v + shift for v in c]
        sse = sum((x[i] - pt[i]) ** 2 for i in range(8))
        if best is None or sse < best - 1e-12:
            best, best_pt = sse, pt
    return best_pt, best


def _oracle(X):
    pts, cosets, sses = [], [], []
    for row in X:
        row = [float(v) for v in row]
        p0, e0 = _nearest_in_coset(row, 0.0)
        p1, e1 = _nearest_in_coset(row, 0.5)
        if e1 < e0:
            pts.append(p1)
            cosets.append(1)
            sses.append(e1)
        else:
            pts.append(p0)
            cosets.append(0)
            sses.append(e0)
    return pts, cosets, sses


def check_example(solve):
    X = torch.tensor([[0.9, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
    out = solve(X)
    expect = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out["rounded"], expect, atol=1e-6), f"got {out['rounded'].tolist()}"
    assert out["coset"].tolist() == [0], f"got {out['coset'].tolist()}"
    assert abs(float(out["sse"][0]) - 0.58) < 1e-6, f"got {out['sse'].tolist()}"


check_example.example = True


def check_lattice_membership(solve):
    """输出必须落在 E8：coset=0 是偶坐标和整数点，coset=1 是偶坐标和的半整数点。"""
    torch.manual_seed(6)
    X = torch.randn(50, 8) * 3
    out = solve(X)
    rounded = out["rounded"].to(torch.float64)
    coset = out["coset"].tolist()
    for i in range(X.shape[0]):
        if coset[i] == 0:
            vals = rounded[i]
            assert torch.allclose(vals, vals.round(), atol=1e-6), f"第 {i} 行不是整数点"
            assert int(vals.sum().item()) % 2 == 0, f"第 {i} 行整数坐标和不是偶数"
        else:
            vals = rounded[i] - 0.5
            assert torch.allclose(vals, vals.round(), atol=1e-6), f"第 {i} 行不是半整数点"
            assert int(vals.sum().item()) % 2 == 0, f"第 {i} 行半整数坐标和不是偶数"
        expect_sse = float(((X[i].to(torch.float64) - rounded[i]) ** 2).sum())
        assert abs(float(out["sse"][i]) - expect_sse) < 1e-5, f"第 {i} 行 SSE 不符"


def check_matches_oracle(solve):
    """40 行不同分布的随机点与纯 python 局部枚举 oracle 对齐。"""
    torch.manual_seed(18)
    blocks = [
        torch.randn(10, 8) * 0.5,
        torch.randn(10, 8) * 3.0,
        torch.rand(10, 8) * 6 - 3,
        torch.randn(10, 8) * 0.1,
    ]
    X = torch.cat(blocks, dim=0)
    out = solve(X)
    pts, cosets, sses = _oracle(X.tolist())
    assert out["coset"].tolist() == cosets, "陪集选择不符"
    assert torch.allclose(out["rounded"].to(torch.float64),
                          torch.tensor(pts, dtype=torch.float64), atol=1e-6), \
        "最近点不符"
    assert torch.allclose(out["sse"], torch.tensor(sses, dtype=torch.float32), atol=1e-5), \
        "SSE 不符"


def check_parity_fix(solve):
    """整数点坐标和为奇数时必须把一个坐标移动到次近整数（朝 x 的方向）。"""
    X = torch.tensor([[1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = solve(X)
    assert out["rounded"].tolist() == [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], \
        f"got {out['rounded'].tolist()}"
    assert out["coset"].tolist() == [0]
    assert abs(float(out["sse"][0]) - 0.64) < 1e-6, f"got {out['sse'].tolist()}"


def check_coset_selection(solve):
    """0.4 在 D8 中距离 1.28，在半整数格中距离 0.08，必须选 coset 1。"""
    X = torch.tensor([[0.4] * 8])
    out = solve(X)
    assert out["coset"].tolist() == [1], f"got {out['coset'].tolist()}"
    assert torch.allclose(out["rounded"], torch.full((1, 8), 0.5), atol=1e-6), \
        f"got {out['rounded'].tolist()}"
    assert abs(float(out["sse"][0]) - 0.08) < 1e-6, f"got {out['sse'].tolist()}"


def check_tie_prefers_d8(solve):
    """两个陪集等距时选 coset 0：0.25 到 0 与到 0.5 的距离都是 0.5。"""
    X = torch.tensor([[0.25] * 8])
    out = solve(X)
    assert out["coset"].tolist() == [0], f"got {out['coset'].tolist()}"
    assert torch.allclose(out["rounded"], torch.zeros(1, 8), atol=0.0), \
        f"got {out['rounded'].tolist()}"
    assert abs(float(out["sse"][0]) - 0.5) < 1e-6


def check_validation(solve):
    """形状、维度、浮点类型、非有限值非法时抛 ValueError。"""
    def expect_error(x):
        try:
            solve(x)
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError：shape={tuple(x.shape)}")

    expect_error(torch.zeros(8))
    expect_error(torch.zeros(2, 7))
    expect_error(torch.zeros(2, 9))
    expect_error(torch.zeros(2, 2, 8))
    expect_error(torch.zeros(0, 8))
    expect_error(torch.zeros(2, 8, dtype=torch.int64))
    expect_error(torch.tensor([[float("nan")] * 8]))
