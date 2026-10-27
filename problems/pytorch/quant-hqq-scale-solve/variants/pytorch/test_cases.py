import torch


def _oracle(W, bits, iters):
    """逐组标量重算（纯 python 循环，与参考解的向量化路径无关）。"""
    qmax = 2 ** bits - 1
    scales, zeros, outs = [], [], []
    for g in range(W.shape[0]):
        w = [float(v) for v in W[g]]
        lo, hi = min(w), max(w)
        if hi <= lo:
            scales.append(1.0)
            zeros.append(0)
            outs.append(torch.full((len(w),), lo))
            continue
        sc = (hi - lo) / qmax
        zp = min(max(round(-lo / sc), 0), qmax)
        q = [min(max(round(v / sc) + zp, 0), qmax) - zp for v in w]
        for _ in range(iters):
            den = sum(t * t for t in q)
            if den > 1e-12:
                sc = sum(v * t for v, t in zip(w, q)) / den
            q = [min(max(round(v / sc) + zp, 0), qmax) - zp for v in w]
        den = sum(t * t for t in q)
        if den > 1e-12:
            sc = sum(v * t for v, t in zip(w, q)) / den
        scales.append(sc)
        zeros.append(zp)
        outs.append(torch.tensor([t * sc for t in q]))
    return (torch.tensor(scales, dtype=torch.float32),
            torch.tensor(zeros, dtype=torch.int64), torch.stack(outs))


def check_example(solve):
    W = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    scale, zero, deq = solve(W, 4, 2)
    assert torch.allclose(scale, torch.tensor([1.0]), atol=1e-5), f"got {scale.tolist()}"
    assert zero.tolist() == [0], f"got {zero.tolist()}"
    assert torch.allclose(deq, W, atol=1e-5), f"got {deq.tolist()}"


check_example.example = True


def check_closed_form_optimality(solve):
    """固定整数 q 时 scale 必须是 <W,q>/<q,q>：把 scale 扰动 ±10% 用同一 q 重建会更差。"""
    torch.manual_seed(4)
    W = torch.randn(3, 64) * 1.5 + 0.3
    scale, zero, deq = solve(W, 4, 3)
    for g in range(3):
        w = W[g].to(torch.float64)
        s = float(scale[g])
        z = int(zero[g])
        q = ((deq[g].to(torch.float64) / s).round()).clamp(-(2 ** 4), 2 ** 4)
        num = float((w * q).sum())
        den = float((q * q).sum())
        assert abs(s - num / den) <= 1e-4 * max(1.0, abs(s)), (
            f"第 {g} 组 scale 不是闭式解：{s} vs {num / den}")
        base = float(((w - q * s) ** 2).sum())
        for f in (0.9, 1.1):
            assert float(((w - q * (s * f)) ** 2).sum()) >= base - 1e-9, (
                "闭式 scale 应为固定 q 下的最小二乘最优")
        assert z >= 0


def check_beats_rtn(solve):
    """同一数据下 HQQ 的总 SSE 应不高于对称 RTN（absmax/(2^(b-1)-1)，zero=0）。"""
    torch.manual_seed(12)
    W = torch.randn(4, 128) * 2.0 + 0.5
    _, _, deq = solve(W, 4, 3)
    hqq_err = float(((W - deq) ** 2).sum())
    sc = W.abs().amax(1, keepdim=True) / 7
    q = (W / sc).round().clamp(-7, 7)
    rtn_err = float(((W - q * sc) ** 2).sum())
    assert hqq_err <= rtn_err, f"HQQ {hqq_err:.4f} 应 <= RTN {rtn_err:.4f}"


def check_matches_oracle(solve):
    torch.manual_seed(23)
    W = torch.randn(3, 32) * 3 - 1.0
    got_s, got_z, got_d = solve(W, 5, 2)
    exp_s, exp_z, exp_d = _oracle(W, 5, 2)
    assert torch.allclose(got_s, exp_s, atol=1e-5, rtol=1e-5), f"{got_s.tolist()} vs {exp_s.tolist()}"
    assert torch.equal(got_z, exp_z), f"{got_z.tolist()} vs {exp_z.tolist()}"
    assert torch.allclose(got_d, exp_d, atol=1e-5, rtol=1e-5), "重建与重算不符"


def check_degenerate_and_validation(solve):
    """常量组：scale=1、zero=0、deq 精确等于常量；bits/iters 非法抛 ValueError。"""
    W = torch.tensor([[2.5, 2.5, 2.5], [-1.0, 0.0, 1.0]])
    scale, zero, deq = solve(W, 4, 2)
    assert abs(float(scale[0]) - 1.0) < 1e-6 and int(zero[0]) == 0
    assert torch.allclose(deq[0], torch.full((3,), 2.5), atol=1e-6), f"got {deq[0].tolist()}"
    for bad_b, bad_i in ((1, 1), (9, 1), (4, -1)):
        try:
            solve(W, bad_b, bad_i)
            raise AssertionError(f"bits={bad_b}, iters={bad_i} 应抛 ValueError")
        except ValueError:
            pass
