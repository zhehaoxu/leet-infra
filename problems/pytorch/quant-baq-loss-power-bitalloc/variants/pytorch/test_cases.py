import math

import torch

_TH = 1.95
_EPS = 1e-8


def _oracle(W, loss, R_ref, ratio):
    nrows, ncols = len(W), len(W[0])
    loss_mean = float(ratio) * (sum(float(v) for v in loss) / len(loss))
    fac = 2.0 ** (_TH * R_ref)

    levels = []
    for v in loss:
        t = max((float(v) / loss_mean) * fac, 1.0)
        levels.append(int(round(t ** (1.0 / _TH))))
    maxq = [v - 1 for v in levels]
    bits = [math.log2(v) for v in levels]

    xmin, xmax = [], []
    for r in range(nrows):
        lo = min(min(float(v) for v in W[r]), 0.0)
        hi = max(max(float(v) for v in W[r]), 0.0)
        if lo == 0.0 and hi == 0.0:
            lo, hi = -1.0, 1.0
        xmin.append(lo)
        xmax.append(hi)

    scale, zero, dequant = [], [], []
    for r in range(nrows):
        diff = xmax[r] - xmin[r]
        if diff == 0.0:
            diff += _EPS
        sr, zr, dr = [], [], []
        for c in range(ncols):
            s = diff / (maxq[c] + _EPS)
            z = round(-xmin[r] / s)
            q = round(float(W[r][c]) / s) + z
            q = min(max(q, 0), maxq[c])
            sr.append(s)
            zr.append(z)
            dr.append(s * (q - z))
        scale.append(sr)
        zero.append(zr)
        dequant.append(dr)

    return {"bits": bits, "maxq": maxq, "scale": scale, "zero": zero, "dequant": dequant}


def _assert_matches(out, ref, tag=""):
    assert out["maxq"].tolist() == ref["maxq"], f"{tag} maxq: {out['maxq'].tolist()} != {ref['maxq']}"
    assert out["zero"].tolist() == ref["zero"], f"{tag} zero: {out['zero'].tolist()} != {ref['zero']}"
    assert torch.allclose(out["bits"].to(torch.float64),
                          torch.tensor(ref["bits"], dtype=torch.float64), atol=1e-6), \
        f"{tag} bits: {out['bits'].tolist()} != {ref['bits']}"
    assert torch.allclose(out["scale"].to(torch.float64),
                          torch.tensor(ref["scale"], dtype=torch.float64), atol=1e-6), \
        f"{tag} scale: {out['scale'].tolist()} != {ref['scale']}"
    assert torch.allclose(out["dequant"].to(torch.float64),
                          torch.tensor(ref["dequant"], dtype=torch.float64), atol=1e-6), \
        f"{tag} dequant: {out['dequant'].tolist()} != {ref['dequant']}"


def check_example(solve):
    W = torch.tensor([[0.8, 0.6, 0.9], [0.2, 0.6, 0.3]], dtype=torch.float32)
    loss = torch.tensor([0.3, 1.0, 1.7], dtype=torch.float32)
    out = solve(W, loss, 2, 1.0)

    assert out["maxq"].tolist() == [1, 3, 4], f"got {out['maxq'].tolist()}"
    assert out["zero"].tolist() == [[0, 0, 0], [0, 0, 0]], f"got {out['zero'].tolist()}"
    expect_bits = torch.tensor([1.0, 2.0, math.log2(5)])
    assert torch.allclose(out["bits"], expect_bits, atol=1e-6), f"got {out['bits'].tolist()}"

    expect_scale = torch.tensor([[0.9, 0.3, 0.225], [0.6, 0.2, 0.15]])
    assert torch.allclose(out["scale"], expect_scale, atol=1e-6), f"got {out['scale'].tolist()}"

    expect_deq = torch.tensor([[0.9, 0.6, 0.9], [0.0, 0.6, 0.3]])
    assert torch.allclose(out["dequant"], expect_deq, atol=1e-6), f"got {out['dequant'].tolist()}"


check_example.example = True


def check_allocation(solve):
    """位宽只由每列损失经幂律公式决定，且随损失单调不减。"""
    W = torch.arange(-12, 13, dtype=torch.float32).reshape(5, 5) / 8
    loss = torch.tensor([0.1, 0.5, 1.0, 2.0, 8.0])
    ref = _oracle(W.tolist(), loss.tolist(), 3, 1.0)
    out = solve(W, loss, 3, 1.0)
    _assert_matches(out, ref, "allocation")
    bits = out["bits"].tolist()
    assert all(bits[i] <= bits[i + 1] for i in range(len(bits) - 1)), f"bits 不单调: {bits}"

    # 损失恰好等于均值时取参考位宽
    one = solve(torch.tensor([[1.0, -1.0]]), torch.tensor([1.0, 1.0]), 4, 1.0)
    assert torch.allclose(one["bits"], torch.tensor([4.0, 4.0]), atol=1e-6), \
        f"got {one['bits'].tolist()}"
    assert one["maxq"].tolist() == [15, 15], f"got {one['maxq'].tolist()}"
    assert torch.allclose(one["scale"][0], torch.tensor([2.0 / 15.0, 2.0 / 15.0]), atol=1e-6), \
        f"got {one['scale'].tolist()}"


def check_row_stats(solve):
    """scale 用每行的动态范围、除以每列 maxq；zero 由该行 xmin 决定。"""
    torch.manual_seed(11)
    W = torch.randn(4, 6) * 2
    loss = torch.tensor([0.2, 0.4, 0.9, 1.5, 3.0, 6.0])
    out = solve(W, loss, 2, 1.0)
    maxq = out["maxq"].to(torch.float64)
    for r in range(W.shape[0]):
        row = W[r].to(torch.float64)
        lo = min(float(row.min()), 0.0)
        hi = max(float(row.max()), 0.0)
        diff = hi - lo
        for c in range(W.shape[1]):
            s = diff / (float(maxq[c]) + _EPS)
            assert abs(float(out["scale"][r, c]) - s) <= 1e-6 * max(1.0, abs(s)), \
                f"scale[{r},{c}] 不符: {float(out['scale'][r, c])} vs {s}"
            z = round(-lo / s)
            assert int(out["zero"][r, c]) == z, \
                f"zero[{r},{c}] 不符: {int(out['zero'][r, c])} vs {z}"


def check_matches_oracle(solve):
    """多组随机输入与纯 Python oracle 全字段对齐。"""
    torch.manual_seed(23)
    cases = [
        (torch.randn(6, 5) * 0.5, torch.rand(5) * 2),
        (torch.randn(4, 7) * 3.0, torch.rand(7) * 10),
        (torch.rand(5, 4) * 8 - 4, torch.rand(4) * 0.2),
    ]
    for idx, (W, loss) in enumerate(cases):
        ref = _oracle(W.tolist(), loss.tolist(), 2, 1.0)
        out = solve(W, loss, 2, 1.0)
        _assert_matches(out, ref, f"case{idx}")


def check_quant_grid(solve):
    """反量化值必须落在 scale 的整数格上，且整数码在 [0, maxq] 内。"""
    torch.manual_seed(31)
    W = torch.randn(5, 6) * 2
    loss = torch.tensor([0.3, 0.6, 1.0, 1.8, 4.0, 9.0])
    out = solve(W, loss, 2, 1.0)
    scale = out["scale"].to(torch.float64)
    zero = out["zero"].to(torch.float64)
    maxq = out["maxq"].to(torch.float64)
    grid = out["dequant"].to(torch.float64) / scale + zero
    rounded = grid.round()
    assert torch.allclose(grid, rounded, atol=1e-4), f"不在格点上: {grid.tolist()}"
    assert bool((rounded >= 0).all()), f"码值小于 0: {rounded.tolist()}"
    assert bool((rounded <= maxq.unsqueeze(0)).all()), "码值超过 maxq"
    assert torch.allclose(out["dequant"].to(torch.float64),
                          scale * (rounded - zero), atol=1e-5), "反量化与码值不一致"


def check_boundary_allocation(solve):
    """极小损失被 clamp 到 0 bit（整列反量化为 0）；大损失得到更多码字。"""
    W = torch.tensor([[1.0, 0.5], [-0.5, -1.0], [0.25, 0.75]])
    out = solve(W, torch.tensor([1e-12, 1e6]), 2, 1.0)
    assert out["maxq"].tolist() == [0, 5], f"got {out['maxq'].tolist()}"
    assert abs(float(out["bits"][0])) < 1e-6, f"got {out['bits'].tolist()}"
    assert torch.allclose(out["dequant"][:, 0], torch.zeros(3), atol=1e-6), \
        f"0 bit 列：{out['dequant'][:, 0].tolist()}"
    assert float(out["bits"][1]) > 2.0, f"got {out['bits'].tolist()}"

    # 均值放大 16 倍相当于损失缩小 16 倍：R_ref=4 时位宽降为 2
    W2 = torch.tensor([[1.0, -1.0]])
    loss2 = torch.tensor([1.0, 1.0])
    b1 = solve(W2, loss2, 4, 1.0)["bits"]
    b16 = solve(W2, loss2, 4, 16.0)["bits"]
    assert torch.allclose(b1, torch.tensor([4.0, 4.0]), atol=1e-6), f"got {b1.tolist()}"
    assert torch.allclose(b16, torch.tensor([2.0, 2.0]), atol=1e-6), f"got {b16.tolist()}"


def check_zero_rows(solve):
    """全零行按 (-1, +1) 处理，但零权重仍必须反量化回 0。"""
    W = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    out = solve(W, torch.tensor([1.0, 1.0]), 2, 1.0)
    assert out["maxq"].tolist() == [3, 3], f"got {out['maxq'].tolist()}"
    assert torch.allclose(out["dequant"][0], torch.zeros(2), atol=1e-6), \
        f"got {out['dequant'][0].tolist()}"
    assert out["zero"][0].tolist() == [2, 2], f"got {out['zero'].tolist()}"


def check_validation(solve):
    """形状、类型、损失取值、R_ref 与 ratio 非法时抛 ValueError。"""
    good_W = torch.zeros(2, 3)
    good_loss = torch.ones(3)

    def expect_error(fn, tag):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError：{tag}")

    expect_error(lambda: solve(torch.zeros(3), good_loss, 2, 1.0), "W 一维")
    expect_error(lambda: solve(torch.zeros(2, 3, 1), good_loss, 2, 1.0), "W 三维")
    expect_error(lambda: solve(torch.zeros(0, 3), good_loss, 2, 1.0), "W 空行")
    expect_error(lambda: solve(torch.zeros(2, 3, dtype=torch.int64), good_loss, 2, 1.0), "W 非浮点")
    expect_error(lambda: solve(torch.full((2, 3), float("nan")), good_loss, 2, 1.0), "W 含 nan")
    expect_error(lambda: solve(good_W, torch.ones(2), 2, 1.0), "loss 长度不符")
    expect_error(lambda: solve(good_W, torch.ones(2, 3), 2, 1.0), "loss 二维")
    expect_error(lambda: solve(good_W, torch.tensor([1.0, -1.0, 1.0]), 2, 1.0), "loss 负值")
    expect_error(lambda: solve(good_W, torch.tensor([0.0, 0.0, 0.0]), 2, 1.0), "loss 全零")
    expect_error(lambda: solve(good_W, torch.tensor([1.0, float("inf"), 1.0]), 2, 1.0), "loss 非有限")
    expect_error(lambda: solve(good_W, good_loss, 0, 1.0), "R_ref 过小")
    expect_error(lambda: solve(good_W, good_loss, 9, 1.0), "R_ref 过大")
    expect_error(lambda: solve(good_W, good_loss, 2.0, 1.0), "R_ref 非整数")
    expect_error(lambda: solve(good_W, good_loss, True, 1.0), "R_ref 为 bool")
    expect_error(lambda: solve(good_W, good_loss, 2, 0.0), "ratio 为零")
    expect_error(lambda: solve(good_W, good_loss, 2, -1.0), "ratio 为负")
    expect_error(lambda: solve(good_W, good_loss, 2, float("inf")), "ratio 非有限")
