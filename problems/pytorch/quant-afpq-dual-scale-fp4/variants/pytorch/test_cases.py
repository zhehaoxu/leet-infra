import torch

CODES = {
    4: [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    3: [-4.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 4.0],
}
MAXQ = {4: 12.0, 3: 8.0}


def _nearest(q, codes):
    """最近码，距离相同时取码表里更靠前的码（严格小于才替换）。"""
    best, best_d = codes[0], abs(q - codes[0])
    for c in codes[1:]:
        d = abs(q - c)
        if d < best_d:
            best, best_d = c, d
    return best


def _oracle(W, bits):
    """纯 python 逐行重算：双侧 scale + 最近码。"""
    W = [[float(v) for v in row] for row in W]
    codes = CODES[bits]
    half = MAXQ[bits] / 2
    scale_pos, scale_neg = [], []
    q_pos_all, q_neg_all, deq_all = [], [], []
    for row in W:
        xmin = min(min(row), 0.0)
        xmax = max(max(row), 0.0)
        if xmin == 0.0 and xmax == 0.0:
            xmin, xmax = -1.0, 1.0
        sp = xmax / half
        sn = abs(xmin) / half
        qp_row, qn_row, deq_row = [], [], []
        for w in row:
            xp = w if w >= 0 else 0.0
            xn = w if w < 0 else 0.0
            qp = xp / sp if sp != 0.0 else 0.0
            qn = xn / sn if sn != 0.0 else 0.0
            cp = _nearest(qp, codes) if sp != 0.0 else 0.0
            cn = _nearest(qn, codes) if sn != 0.0 else 0.0
            qp_row.append(cp)
            qn_row.append(cn)
            deq_row.append(cp * sp + cn * sn)
        scale_pos.append(sp)
        scale_neg.append(sn)
        q_pos_all.append(qp_row)
        q_neg_all.append(qn_row)
        deq_all.append(deq_row)
    return scale_pos, scale_neg, q_pos_all, q_neg_all, deq_all


def check_example(solve):
    W = torch.tensor([[3.0, -0.3, 0.2]], dtype=torch.float32)
    out = solve(W, 4)
    assert torch.allclose(out["scale_pos"], torch.tensor([0.5]), atol=1e-6), \
        f"got {out['scale_pos'].tolist()}"
    assert torch.allclose(out["scale_neg"], torch.tensor([0.05]), atol=1e-6), \
        f"got {out['scale_neg'].tolist()}"
    assert torch.allclose(out["q_pos"], torch.tensor([[6.0, 0.0, 0.5]]), atol=1e-6), \
        f"got {out['q_pos'].tolist()}"
    assert torch.allclose(out["q_neg"], torch.tensor([[0.0, -6.0, 0.0]]), atol=1e-6), \
        f"got {out['q_neg'].tolist()}"
    assert torch.allclose(out["dequant"], torch.tensor([[3.0, -0.3, 0.25]]), atol=1e-6), \
        f"got {out['dequant'].tolist()}"


check_example.example = True


def check_codebook(solve):
    """返回的 q_pos/q_neg 必须只取对应 bit 宽的码表值。"""
    torch.manual_seed(3)
    for bits in (3, 4):
        W = torch.randn(6, 64) * 2
        out = solve(W, bits)
        allowed = set(CODES[bits])
        for key in ("q_pos", "q_neg"):
            vals = set(out[key].flatten().tolist())
            assert vals <= allowed, f"bits={bits} 的 {key} 出现码表外取值：{sorted(vals - allowed)}"
        assert float(out["q_pos"].abs().max()) <= MAXQ[bits] / 2
        assert float(out["q_neg"].abs().max()) <= MAXQ[bits] / 2


def check_scales(solve):
    """scale_pos=xmax/half、scale_neg=|xmin|/half，且全零行退化为 (-1,+1)。"""
    W = torch.tensor([
        [3.0, -0.3, 0.2],
        [0.0, 0.0, 0.0],
        [-1.8, -0.6, 0.0],
        [0.4, 0.8, 1.2],
    ])
    out = solve(W, 4)
    expect_pos = torch.tensor([0.5, 1.0 / 6, 0.0, 0.2])
    expect_neg = torch.tensor([0.05, 1.0 / 6, 0.3, 0.0])
    assert torch.allclose(out["scale_pos"], expect_pos, atol=1e-6), \
        f"got {out['scale_pos'].tolist()}"
    assert torch.allclose(out["scale_neg"], expect_neg, atol=1e-6), \
        f"got {out['scale_neg'].tolist()}"


def check_matches_oracle(solve):
    """30 组随机配置（含单侧行）与纯 python oracle 逐项对齐。"""
    torch.manual_seed(17)
    for t in range(30):
        G = int(torch.randint(1, 9, (1,)))
        N = int(torch.randint(1, 33, (1,)))
        bits = 4 if t % 2 == 0 else 3
        W = torch.randn(G, N) * (2 + t % 4)
        if t % 5 == 0:
            W = W.abs()
        elif t % 5 == 1:
            W = -W.abs()
        out = solve(W, bits)
        sp, sn, qp, qn, deq = _oracle(W.tolist(), bits)
        assert torch.allclose(out["scale_pos"], torch.tensor(sp, dtype=torch.float32), atol=1e-6)
        assert torch.allclose(out["scale_neg"], torch.tensor(sn, dtype=torch.float32), atol=1e-6)
        assert torch.allclose(out["q_pos"], torch.tensor(qp), atol=1e-6)
        assert torch.allclose(out["q_neg"], torch.tensor(qn), atol=1e-6)
        assert torch.allclose(out["dequant"], torch.tensor(deq), atol=1e-6), \
            f"case {t} 重建不符"


def check_benefit_vs_single_scale(solve):
    """正负极值悬殊时，双侧 scale 能把小的一侧量化得很准，单 scale 做不到。"""
    W = torch.tensor([[3.0, -0.3, 0.25]])
    out = solve(W, 4)
    dual_err = float(((W - out["dequant"]) ** 2).sum())
    assert dual_err < 1e-12, f"双侧 scale 应精确重建该行，err={dual_err}"
    scale = 3.0 / 6
    codes = torch.tensor(CODES[4])
    q = (W[0] / scale).unsqueeze(-1)
    single = codes[(q - codes).abs().argmin(dim=-1)] * scale
    single_err = float(((W[0] - single) ** 2).sum())
    assert single_err > 1e-3, f"单 scale 应存在明显误差，err={single_err}"
    assert dual_err < single_err


def check_tie_picks_first_code(solve):
    """最近码距离相同时必须取码表里更靠前的码：q=1.75 应取 1.5 而非 2。"""
    W = torch.tensor([[3.0, 0.875]])
    out = solve(W, 4)
    assert float(out["scale_pos"][0]) == 0.5
    assert float(out["q_pos"][0, 1]) == 1.5, f"got {out['q_pos'].tolist()}"
    assert abs(float(out["dequant"][0, 1]) - 0.75) < 1e-6, f"got {out['dequant'].tolist()}"


def check_one_sided_rows(solve):
    """整行无正数/无负数时，空的一侧 scale=0 且贡献为 0，不做 0/0。"""
    pos = solve(torch.tensor([[0.6, 0.3, 1.2]]), 4)
    assert float(pos["scale_neg"][0]) == 0.0
    assert torch.allclose(pos["q_neg"], torch.zeros(1, 3), atol=0.0)
    assert torch.allclose(pos["dequant"], torch.tensor([[0.6, 0.3, 1.2]]), atol=1e-6)

    neg = solve(torch.tensor([[-1.8, -0.6]]), 4)
    assert float(neg["scale_pos"][0]) == 0.0
    assert torch.allclose(neg["q_pos"], torch.zeros(1, 2), atol=0.0)
    assert torch.allclose(neg["dequant"], torch.tensor([[-1.8, -0.6]]), atol=1e-6)


def check_all_zero_row(solve):
    """全零行按 xmin=-1、xmax=1 处理：两侧 scale 都是 1/half，重建为 0。"""
    out = solve(torch.zeros(2, 5), 4)
    assert torch.allclose(out["scale_pos"], torch.full((2,), 1.0 / 6), atol=1e-6)
    assert torch.allclose(out["scale_neg"], torch.full((2,), 1.0 / 6), atol=1e-6)
    assert torch.allclose(out["dequant"], torch.zeros(2, 5), atol=0.0)
    assert torch.allclose(out["q_pos"], torch.zeros(2, 5), atol=0.0)
    assert torch.allclose(out["q_neg"], torch.zeros(2, 5), atol=0.0)


def check_validation(solve):
    """bits、维度、空张量、非浮点、非有限值非法时抛 ValueError。"""
    W = torch.zeros(1, 3)

    def expect_error(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError：{args}")

    expect_error(W, 2)
    expect_error(W, 5)
    expect_error(W, True)
    expect_error(W, 4.0)
    expect_error(torch.zeros(3), 4)
    expect_error(torch.zeros(2, 2, 2), 4)
    expect_error(torch.zeros(0, 3), 4)
    expect_error(torch.zeros(2, 0), 4)
    expect_error(torch.zeros(1, 3, dtype=torch.int64), 4)
    expect_error(torch.tensor([[float("inf"), 0.0]]), 4)
