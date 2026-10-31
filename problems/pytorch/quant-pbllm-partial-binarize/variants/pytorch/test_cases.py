import torch


def _oracle(W, low_frac, high_bits):
    """纯 python 逐行重算：全局阈值掩码 + xnor low + 逐行非对称 high。"""
    W = [[float(v) for v in row] for row in W]
    rows, cols = len(W), len(W[0])
    flat = sorted(abs(v) for row in W for v in row)
    thresh = flat[int(rows * cols * low_frac)]
    maxq = 2 ** high_bits - 1

    mask_all, means, lscales, hscales, hzeros, deq_all = [], [], [], [], [], []
    for row in W:
        mrow = [abs(v) <= thresh for v in row]
        masked = [v if m else 0.0 for v, m in zip(row, mrow)]
        mean = sum(masked) / cols
        lscale = sum(abs(v - mean) for v in masked) / cols

        xmin = min(min(row), 0.0)
        xmax = max(max(row), 0.0)
        if xmin == 0.0 and xmax == 0.0:
            xmin, xmax = -1.0, 1.0
        hscale = (xmax - xmin) / maxq
        hzero = round(-xmin / hscale)

        deq = []
        for v, m in zip(row, mrow):
            q = min(max(round(v / hscale) + hzero, 0), maxq)
            high = (q - hzero) * hscale
            d = v - mean
            sign = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
            low = sign * lscale + mean
            deq.append(low if m else high)

        mask_all.append(mrow)
        means.append(mean)
        lscales.append(lscale)
        hscales.append(hscale)
        hzeros.append(hzero)
        deq_all.append(deq)
    return mask_all, means, lscales, hscales, hzeros, deq_all


def check_example(solve):
    W = torch.tensor([[1.0, -0.5, 0.25, -0.25, 0.1, -0.1]], dtype=torch.float32)
    out = solve(W, 0.5, 4)
    assert out["mask"].tolist() == [[False, False, True, True, True, True]], \
        f"got {out['mask'].tolist()}"
    assert torch.allclose(out["low_mean"], torch.tensor([0.0]), atol=1e-6), \
        f"got {out['low_mean'].tolist()}"
    assert torch.allclose(out["low_scale"], torch.tensor([0.7 / 6]), atol=1e-6), \
        f"got {out['low_scale'].tolist()}"
    assert torch.allclose(out["high_scale"], torch.tensor([0.1]), atol=1e-6), \
        f"got {out['high_scale'].tolist()}"
    assert out["high_zero"].tolist() == [5], f"got {out['high_zero'].tolist()}"
    expect = torch.tensor([[1.0, -0.5, 0.7 / 6, -0.7 / 6, 0.7 / 6, -0.7 / 6]])
    assert torch.allclose(out["dequant"], expect, atol=1e-6), \
        f"got {out['dequant'].tolist()}"


check_example.example = True


def check_mask_threshold(solve):
    """掩码必须按全局 |W| 排序的第 int(numel*low_frac) 个值为阈值，含等号。"""
    torch.manual_seed(5)
    W = torch.randn(4, 7)
    for frac in (0.0, 0.3, 0.5, 0.9):
        out = solve(W, frac, 4)
        vals = sorted(W.abs().flatten().tolist())
        thresh = vals[int(W.numel() * frac)]
        expect = (W.abs() <= thresh)
        assert torch.equal(out["mask"], expect), f"frac={frac} 掩码不符"

    same = torch.full((2, 5), 5.0)
    out = solve(same, 0.2, 4)
    assert bool(out["mask"].all()), "全等权重时阈值等于该值，所有位置都应入选"


def check_low_stats(solve):
    """low_mean/low_scale 必须在 masked=W*mask 上按行统计，未掩码位置以 0 计入。"""
    torch.manual_seed(9)
    W = torch.randn(3, 8) * 2
    out = solve(W, 0.4, 4)
    mask = out["mask"]
    masked = W.to(torch.float64) * mask
    mean = masked.mean(dim=1)
    scale = (masked - mean.unsqueeze(1)).abs().mean(dim=1)
    assert torch.allclose(out["low_mean"], mean.to(torch.float32), atol=1e-6), \
        f"{out['low_mean'].tolist()} vs {mean.tolist()}"
    assert torch.allclose(out["low_scale"], scale.to(torch.float32), atol=1e-6), \
        f"{out['low_scale'].tolist()} vs {scale.tolist()}"


def check_high_stats(solve):
    """high 的 scale/zero 必须逐行由 min/max 决定，全零行退化为 (-1,+1)。"""
    W = torch.tensor([
        [1.0, -0.5, 0.25],
        [0.3, 0.9, 0.6],
        [-0.9, -0.3, -0.6],
        [0.0, 0.0, 0.0],
    ])
    out = solve(W, 0.5, 4)
    assert torch.allclose(out["high_scale"], torch.tensor([0.1, 0.06, 0.06, 2.0 / 15]), atol=1e-6), \
        f"got {out['high_scale'].tolist()}"
    assert out["high_zero"].tolist() == [5, 0, 15, 8], f"got {out['high_zero'].tolist()}"


def check_matches_oracle(solve):
    """30 组随机配置与纯 python oracle 逐项对齐（mask/统计量/重建）。"""
    torch.manual_seed(23)
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    for t in range(30):
        G = int(torch.randint(1, 7, (1,)))
        C = int(torch.randint(1, 33, (1,)))
        bits = int(torch.randint(2, 9, (1,)))
        frac = fracs[t % len(fracs)]
        W = torch.randn(G, C) * (1 + t % 3)
        if t % 7 == 0:
            W = W.abs()
        elif t % 7 == 1:
            W = -W.abs()
        elif t % 7 == 2:
            W[0] = W[0, 0]
        out = solve(W, frac, bits)
        mask, means, lscales, hscales, hzeros, deq = _oracle(W.tolist(), frac, bits)
        assert out["mask"].tolist() == mask, f"case {t} mask 不符"
        assert torch.allclose(out["low_mean"], torch.tensor(means), atol=1e-6), f"case {t} low_mean"
        assert torch.allclose(out["low_scale"], torch.tensor(lscales), atol=1e-6), f"case {t} low_scale"
        assert torch.allclose(out["high_scale"], torch.tensor(hscales), atol=1e-6), f"case {t} high_scale"
        assert out["high_zero"].tolist() == hzeros, f"case {t} high_zero"
        assert torch.allclose(out["dequant"], torch.tensor(deq), atol=1e-6), f"case {t} dequant"


def check_piecewise_selection(solve):
    """dequant 必须逐元素按 mask 选择 low/high 两条通路的重建值。"""
    torch.manual_seed(41)
    W = torch.randn(3, 16)
    out = solve(W, 0.3, 4)
    mask = out["mask"]
    hs = out["high_scale"].to(torch.float64).unsqueeze(1)
    hz = out["high_zero"].to(torch.float64).unsqueeze(1)
    lm = out["low_mean"].to(torch.float64).unsqueeze(1)
    ls = out["low_scale"].to(torch.float64).unsqueeze(1)
    x = W.to(torch.float64)
    q = (x / hs).round().add(hz).clamp(0, 15)
    high = (q - hz) * hs
    low = torch.sign(x - lm) * ls + lm
    expect = torch.where(mask, low, high)
    assert torch.allclose(out["dequant"], expect.to(torch.float32), atol=1e-6)


def check_all_equal_and_zero_rows(solve):
    """全等行：掩码全 True，low_scale=0，重建恒等于该常量；全零行重建为 0。"""
    same = torch.full((2, 5), 0.7)
    out = solve(same, 0.2, 4)
    assert bool(out["mask"].all())
    assert torch.allclose(out["low_mean"], torch.full((2,), 0.7), atol=1e-6)
    assert torch.allclose(out["low_scale"], torch.zeros(2), atol=1e-9)
    assert torch.allclose(out["dequant"], same, atol=1e-6)

    zeros = torch.zeros(2, 5)
    out = solve(zeros, 0.2, 4)
    assert bool(out["mask"].all())
    assert torch.allclose(out["dequant"], zeros, atol=0.0)
    assert torch.allclose(out["high_scale"], torch.full((2,), 2.0 / 15), atol=1e-6)
    assert out["high_zero"].tolist() == [8, 8]


def check_validation(solve):
    """high_bits、low_frac、W 的非法入参必须抛 ValueError。"""
    W = torch.zeros(2, 3)

    def expect_error(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError：{args}")

    expect_error(W, 0.2, 1)
    expect_error(W, 0.2, 9)
    expect_error(W, 0.2, True)
    expect_error(W, 0.2, 4.0)
    expect_error(W, -0.1, 4)
    expect_error(W, 1.0, 4)
    expect_error(W, True, 4)
    expect_error(W, float("nan"), 4)
    expect_error(torch.zeros(3), 0.2, 4)
    expect_error(torch.zeros(2, 2, 2), 0.2, 4)
    expect_error(torch.zeros(0, 3), 0.2, 4)
    expect_error(torch.zeros(2, 3, dtype=torch.int64), 0.2, 4)
    expect_error(torch.tensor([[float("nan"), 0.0]]), 0.2, 4)
