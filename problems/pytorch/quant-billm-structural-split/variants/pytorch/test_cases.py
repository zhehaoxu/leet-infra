import math

import torch

_EXAMPLE_W = [
    [0.5, 3.0, 1.0, 2.0, 0.25, 4.0],
    [0.1, -2.5, 0.5, -1.0, 0.1, -3.5],
]


def _sign(v):
    if v > 0:
        return 1.0
    if v < 0:
        return -1.0
    return 0.0


def _hor(x, mask, order):
    rows, cols = len(x), len(x[0])
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        idxs = [c for c in range(cols) if mask[r][c]]
        if not idxs:
            continue
        for _ in range(order):
            residual = [x[r][c] - out[r][c] for c in idxs]
            mean = sum(residual) / len(residual)
            centered = [v - mean for v in residual]
            scale = sum(abs(v) for v in centered) / len(centered)
            for c, cv in zip(idxs, centered):
                out[r][c] += _sign(cv) * scale + mean
    return out


def _sum_groups(*groups):
    rows, cols = len(groups[0]), len(groups[0][0])
    return [
        [sum(g[r][c] for g in groups) for c in range(cols)]
        for r in range(rows)
    ]


def _mse(a, b):
    rows, cols = len(a), len(a[0])
    return sum(
        (a[r][c] - b[r][c]) ** 2 for r in range(rows) for c in range(cols)
    ) / (rows * cols)


def _quantile(sorted_vals, q):
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _quantile_grid():
    return [0.10 + 0.01 * k for k in range(81)]


def _oracle(W, up_lim):
    rows, cols = len(W), len(W[0])
    counts = [sum(abs(W[r][c]) for r in range(rows)) for c in range(cols)]
    order = sorted(range(cols), key=lambda c: (-counts[c], c))
    candidates = order[:up_lim]

    best_i, best_err = 1, float("inf")
    for i in range(1, up_lim):
        sel = set(candidates[:i])
        mask3 = [[c in sel for c in range(cols)] for _ in range(rows)]
        rest = [[not v for v in row] for row in mask3]
        rec = _sum_groups(_hor(W, rest, 2), _hor(W, mask3, 2))
        err = _mse(W, rec)
        if err < best_err:
            best_err = err
            best_i = i

    cols_sel = sorted(candidates[:best_i])
    sel = set(cols_sel)
    mask3 = [[c in sel for c in range(cols)] for _ in range(rows)]
    group3 = _hor(W, mask3, 2)
    flat = sorted(
        abs(W[r][c]) if not mask3[r][c] else 0.0
        for r in range(rows)
        for c in range(cols)
    )

    best_split, best_err = None, float("inf")
    for q in _quantile_grid():
        split = _quantile(flat, q)
        mask2 = [
            [(not mask3[r][c]) and abs(W[r][c]) >= split for c in range(cols)]
            for r in range(rows)
        ]
        mask1 = [
            [(not mask3[r][c]) and abs(W[r][c]) < split for c in range(cols)]
            for r in range(rows)
        ]
        rec = _sum_groups(_hor(W, mask1, 1), _hor(W, mask2, 1), group3)
        err = _mse(W, rec)
        if err < best_err:
            best_err = err
            best_split = split
            best_m1, best_m2, best_rec = mask1, mask2, rec

    return {
        "salient_cols": cols_sel,
        "split": best_split,
        "mask1": best_m1,
        "mask2": best_m2,
        "mask3": mask3,
        "reconstructed": best_rec,
    }


def _assert_close(actual, expect, atol):
    assert torch.allclose(
        actual.to(torch.float64), torch.tensor(expect, dtype=torch.float64), atol=atol, rtol=1e-5
    ), f"数值不符: {actual.tolist()}"


def check_example(solve):
    W = torch.tensor(_EXAMPLE_W, dtype=torch.float32)
    out = solve(W, 3)

    assert out["salient_cols"].tolist() == [1, 5], f"got {out['salient_cols'].tolist()}"
    assert abs(float(out["split"]) - 0.515) <= 1e-5, f"split = {float(out['split'])}"
    assert out["mask1"].tolist() == [[1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0]]
    assert out["mask2"].tolist() == [[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0]]
    assert out["mask3"].tolist() == [[0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
    expect = [
        [0.5, 3.0, 1.0, 2.0, 0.25, 4.0],
        [0.05555555555555555, -2.5, 0.4111111111111111, -1.0, 0.05555555555555555, -3.5],
    ]
    _assert_close(out["reconstructed"], expect, atol=1e-6)


check_example.example = True


def check_matches_oracle(solve):
    """多组形状 / up_lim 与纯 Python oracle 对齐（掩码、显著列、切分、重建）。"""
    torch.manual_seed(11)
    cases = [
        (torch.tensor(_EXAMPLE_W), 2),
        (torch.tensor(_EXAMPLE_W), 6),
        (torch.randn(3, 5) * 2.0, 4),
        (torch.randn(6, 9) * 3.0, 7),
        (torch.randn(4, 4) * 0.5, 2),
    ]
    for W, up_lim in cases:
        out = solve(W, up_lim)
        ref = _oracle(W.tolist(), up_lim)
        assert out["salient_cols"].tolist() == ref["salient_cols"], (up_lim, out["salient_cols"].tolist(), ref["salient_cols"])
        assert out["mask3"].tolist() == ref["mask3"], f"mask3 不符 (up_lim={up_lim})"
        assert out["mask1"].tolist() == ref["mask1"], f"mask1 不符 (up_lim={up_lim})"
        assert out["mask2"].tolist() == ref["mask2"], f"mask2 不符 (up_lim={up_lim})"
        assert abs(float(out["split"]) - ref["split"]) <= 1e-5, (up_lim, float(out["split"]), ref["split"])
        _assert_close(out["reconstructed"], ref["reconstructed"], atol=1e-5)


def check_salient_columns_by_norm(solve):
    """显著列必须按列绝对值和挑选；up_lim=2 时只能选出第 1 名，且输出升序。"""
    W = torch.tensor(
        [[0.1, -4.0, 0.2, 1.0],
         [0.1, 2.0, -0.2, 0.5],
         [0.1, 1.0, 0.2, -0.5]]
    )
    out = solve(W, 2)
    assert out["salient_cols"].tolist() == [1], f"got {out['salient_cols'].tolist()}"

    out = solve(W, 4)
    cols = out["salient_cols"].tolist()
    assert cols == sorted(cols), "salient_cols 必须升序"
    counts = W.abs().sum(dim=0)
    chosen = sum(float(counts[c]) for c in cols)
    ordered = sorted(counts.tolist(), reverse=True)[: len(cols)]
    assert abs(chosen - sum(ordered)) <= 1e-6, "选中的不是列范数最大的若干列"


def check_masks_partition(solve):
    """三个掩码互斥、并集覆盖全部元素；mask3 恰好是显著的整列。"""
    torch.manual_seed(23)
    W = torch.randn(5, 7)
    out = solve(W, 4)
    m1, m2, m3 = out["mask1"], out["mask2"], out["mask3"]
    assert not bool((m1 & m2).any())
    assert not bool((m1 & m3).any())
    assert not bool((m2 & m3).any())
    assert bool((m1 | m2 | m3).all()), "掩码未覆盖全部元素"
    cols = out["salient_cols"].tolist()
    assert m3.sum(dim=0).tolist() == [W.shape[0] if c in cols else 0 for c in range(W.shape[1])]
    assert not bool(m1[:, cols].any()) and not bool(m2[:, cols].any())
    assert bool(m1.any()) and bool(m2.any()), "两段都应为非空"


def check_split_is_masked_quantile(solve):
    """切分点必须是「显著列置零后的 |W|」在 0.10~0.90、步长 0.01 的 81 个分位点之一。"""
    torch.manual_seed(31)
    W = torch.randn(4, 6) * 2.5
    out = solve(W, 3)
    mask3 = out["mask3"]
    flat = torch.where(mask3, torch.zeros_like(W), W).abs().flatten()
    values = torch.sort(flat).values
    n = values.numel()
    grid = torch.linspace(0.10, 0.90, 81, dtype=torch.float64)
    pos = grid * (n - 1)
    lo = torch.floor(pos).to(torch.int64)
    hi = torch.clamp(lo + 1, max=n - 1)
    qs = values[lo] + (values[hi] - values[lo]) * (pos - lo.to(torch.float64))
    split = float(out["split"])
    assert min(abs(split - float(v)) for v in qs) <= 1e-5, f"split={split} 不在分位网格上"


def check_ties_stable(solve):
    """列绝对值和相等时按原列号升序取候选，显著列必须是 [0, 1, ...] 的前缀。"""
    W = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0]])
    out = solve(W, 3)
    cols = out["salient_cols"].tolist()
    assert cols == list(range(len(cols))), f"并列时未保持原列号顺序: {cols}"
    assert 1 <= len(cols) <= 2


def check_boundary_up_lim(solve):
    """up_lim=2 只选 1 列；up_lim=列数时最多选 C-1 列；单行矩阵同样可用。"""
    torch.manual_seed(41)
    W = torch.randn(5, 6) * 2.0
    out = solve(W, 2)
    assert len(out["salient_cols"]) == 1
    out = solve(W, 6)
    assert 1 <= len(out["salient_cols"]) <= 5
    single = torch.tensor([[0.5, -2.0, 1.0, 0.25]])
    out = solve(single, 4)
    assert 1 <= len(out["salient_cols"]) <= 3
    assert out["mask1"].shape == single.shape
    assert bool((out["mask1"] | out["mask2"] | out["mask3"]).all())


def check_validation(solve):
    W = torch.randn(3, 4)
    bad_calls = [
        lambda: solve(W, 1),
        lambda: solve(W, 5),
        lambda: solve(W, 2.0),
        lambda: solve(W, True),
        lambda: solve(W.flatten(), 2),
        lambda: solve(torch.empty(0, 4), 2),
        lambda: solve(torch.tensor([[1.0, float("nan"), 2.0, 3.0]]), 2),
        lambda: solve(torch.tensor([[1, 2, 3, 4]]), 2),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("非法输入应抛 ValueError")
