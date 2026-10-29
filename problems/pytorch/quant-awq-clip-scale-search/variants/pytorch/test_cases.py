import torch


def _norm_s(X, W, beta, max_shrink):
    """测试侧独立的 scale 计算：mean|X| -> s -> 几何归一 -> 收缩下界。"""
    xm = X.to(torch.float64).abs().mean(dim=0)
    wa = W.to(torch.float64).abs().amax(dim=0)
    s = torch.ones_like(xm)
    for j in range(len(xm)):
        if float(xm[j]) > 0 and float(wa[j]) > 0:
            s[j] = float(xm[j]) ** float(beta) / float(wa[j]) ** (1.0 - float(beta))
    gm = float(torch.sqrt(s.max() * s.min()))
    if gm > 0:
        s = s / gm
    return torch.clamp(s, min=1.0 - float(max_shrink))


def _oracle(W, X, ratios, clips, max_shrink, group_size, bits):
    """独立 oracle：纯 Python 嵌套循环，逐 (beta, clip) 重算码与逻辑权重误差。"""
    Wl = W.to(torch.float64).tolist()
    Xl = X.to(torch.float64).tolist()
    O, C = len(Wl), len(Wl[0])
    T = len(Xl)
    qmax = 2 ** (bits - 1) - 1
    xm = [sum(abs(Xl[i][j]) for i in range(T)) / T for j in range(C)]
    wa = [max(abs(Wl[o][j]) for o in range(O)) for j in range(C)]
    wmax = max(abs(v) for row in Wl for v in row)
    losses = torch.zeros(len(ratios), len(clips), dtype=torch.float64)
    best = None
    for bi, beta in enumerate(ratios):
        b = float(beta)
        s = [1.0 if xm[j] == 0.0 or wa[j] == 0.0 else xm[j] ** b / wa[j] ** (1.0 - b) for j in range(C)]
        gm = (max(s) * min(s)) ** 0.5
        if gm > 0:
            s = [v / gm for v in s]
        s = [max(v, 1.0 - float(max_shrink)) for v in s]
        for ci, clip in enumerate(clips):
            thr = float(clip) * wmax
            Wc = [[max(-thr, min(thr, v)) for v in row] for row in Wl]
            codes = [[0] * C for _ in range(O)]
            err = 0.0
            for o in range(O):
                ws = [Wc[o][j] * s[j] for j in range(C)]
                for g in range(C // group_size):
                    seg = ws[g * group_size:(g + 1) * group_size]
                    amax = max(abs(v) for v in seg)
                    sc = amax / qmax
                    for idx, v in enumerate(seg):
                        j = g * group_size + idx
                        if sc == 0.0:
                            q, log = 0, 0.0
                        else:
                            q = max(-qmax, min(qmax, round(v / sc)))
                            log = q * sc / s[j]
                        codes[o][j] = q
                        err += xm[j] * (Wc[o][j] - log) ** 2
            loss = err / (O * C)
            losses[bi, ci] = loss
            if best is None or loss < best[0]:
                best = (loss, b, float(clip), list(s), [row[:] for row in codes])
    return (
        best[1],
        best[2],
        torch.tensor(best[3], dtype=torch.float64),
        torch.tensor(best[4], dtype=torch.int64),
        losses,
    )


def check_example(solve):
    W = torch.tensor([[10.5, -6.0, 2.0], [-1.8, -2.8, 5.8], [-2.4, -1.9, -2.6]])
    X = torch.tensor(
        [[0.02, 0.95, 0.48], [0.02, 0.06, 0.20], [0.01, 1.36, 1.02], [0.01, 0.42, 0.94]]
    )
    ratios = [0.0, 0.5, 1.0]
    clips = [1.0, 0.75, 0.5, 0.25, 0.1]
    beta, clip, scales, codes, losses = solve(W, X, ratios, clips, 0.5, 3, 4)
    assert abs(beta - 0.5) < 1e-9, f"best_beta 应为 0.5，got {beta}"
    assert abs(clip - 0.1) < 1e-9, f"best_clip 应为 0.1，got {clip}"
    assert torch.allclose(
        scales, torch.tensor([0.5, 3.003466, 2.971558]), atol=1e-5
    ), f"got {scales.tolist()}"
    assert codes.tolist() == [[1, -7, 7], [-1, -7, 7], [-1, -7, -7]], f"got {codes.tolist()}"
    exp = torch.tensor(
        [
            [1.29940e-2, 1.32284e-2, 9.88267e-3, 3.14876e-3, 3.11347e-4],
            [2.00691e-2, 2.97825e-2, 1.43926e-2, 1.80263e-3, 1.38912e-4],
            [3.02391e-2, 5.21349e-2, 6.84213e-2, 2.85086e-2, 5.74039e-3],
        ]
    )
    assert torch.allclose(losses, exp, rtol=1e-4, atol=1e-9), f"got {losses.tolist()}"


check_example.example = True


def check_clip_beats_no_clip(solve):
    """离群权重所在通道激活很小：裁掉它换来同组其它列的精度，loss 大幅下降。"""
    W = torch.tensor([[10.5, -6.0, 2.0], [-1.8, -2.8, 5.8], [-2.4, -1.9, -2.6]])
    X = torch.tensor(
        [[0.02, 0.95, 0.48], [0.02, 0.06, 0.20], [0.01, 1.36, 1.02], [0.01, 0.42, 0.94]]
    )
    clips = [1.0, 0.75, 0.5, 0.25, 0.1]
    beta, clip, _, _, losses = solve(W, X, [0.0, 0.5, 1.0], clips, 0.5, 3, 4)
    assert clip == 0.1, f"got {clip}"
    assert float(losses[1, 4]) < float(losses[1, 0]), f"got {losses[1].tolist()}"
    assert float(losses[1, 4]) < float(losses[1, 3]) < float(losses[1, 2]), "裁剪应单调改善"


def check_beta_beats_zero(solve):
    """beta>0 把精度向高激活通道倾斜，优于 beta=0。"""
    W = torch.tensor([[10.5, -6.0, 2.0], [-1.8, -2.8, 5.8], [-2.4, -1.9, -2.6]])
    X = torch.tensor(
        [[0.02, 0.95, 0.48], [0.02, 0.06, 0.20], [0.01, 1.36, 1.02], [0.01, 0.42, 0.94]]
    )
    beta, _, _, _, losses = solve(W, X, [0.0, 0.5, 1.0], [1.0, 0.1], 0.5, 3, 4)
    assert abs(beta - 0.5) < 1e-9, f"got {beta}"
    assert float(losses[1, 1]) < float(losses[0, 1]), f"got {losses[:, 1].tolist()}"
    assert float(losses[1, 1]) < float(losses[2, 1]), f"got {losses[:, 1].tolist()}"


def check_shrink_bound(solve):
    """s 的下界严格是 1 - max_shrink；返回的 scale 不高于/低于任何其它约定值。"""
    W = torch.tensor([[10.5, -6.0, 2.0], [-1.8, -2.8, 5.8], [-2.4, -1.9, -2.6]])
    X = torch.tensor(
        [[0.02, 0.95, 0.48], [0.02, 0.06, 0.20], [0.01, 1.36, 1.02], [0.01, 0.42, 0.94]]
    )
    ratios = [0.0, 0.5, 1.0]
    for shrink in (0.0, 0.3, 0.5, 0.9):
        beta, clip, scales, _, losses = solve(W, X, ratios, [1.0, 0.1], shrink, 3, 4)
        assert float(scales.min()) >= 1.0 - shrink - 1e-6, f"shrink={shrink}: {scales.tolist()}"
        assert torch.allclose(scales, _norm_s(X, W, beta, shrink).to(torch.float32), atol=1e-5), (
            f"shrink={shrink}: 返回的 s 与公式不符，got {scales.tolist()}"
        )
        assert torch.isfinite(losses).all()
    beta0, _, s0, _, _ = solve(W, X, ratios, [1.0], 0.0, 3, 4)
    beta5, _, s5, _, _ = solve(W, X, ratios, [1.0], 0.5, 3, 4)
    assert beta0 == beta5 and not torch.allclose(s0, s5), "下界不同时 s 应不同"


def check_group_quant(solve):
    """分组量化：组内最大绝对值精确落在 ±qmax 上，全零行码全为 0。"""
    W = torch.tensor([[3.0, -1.0, 0.5, -0.25], [0.0, 0.0, 0.0, 0.0]])
    X = torch.ones(2, 4)
    beta, clip, scales, codes, _ = solve(W, X, [0.0], [1.0], 0.0, 2, 4)
    assert codes.shape == (2, 4), f"got {tuple(codes.shape)}"
    assert int(codes.abs().max()) <= 7
    for g in range(2):
        seg = W[0, 2 * g:2 * g + 2] * scales[2 * g:2 * g + 2]
        assert int(codes[0, 2 * g:2 * g + 2].abs().max()) == 7, f"组 {g} 的最大值应为 ±7"
    assert torch.equal(codes[1], torch.zeros(4, dtype=torch.int64)), "全零行应全为 0 码"


def check_bits_eight(solve):
    W = torch.tensor([[5.0, -2.0, 1.0], [-0.5, 3.0, 0.25]])
    X = torch.ones(2, 3)
    _, _, _, codes, losses = solve(W, X, [0.5], [1.0], 0.0, 3, 8)
    assert int(codes.abs().max()) <= 127, f"got {int(codes.abs().max())}"
    assert torch.isfinite(losses).all()


def check_all_zero_weight(solve):
    """全零权重：所有组合 loss=0，取网格先出现者，码全 0。"""
    W = torch.zeros(2, 4)
    X = torch.rand(3, 4) + 0.1
    beta, clip, scales, codes, losses = solve(W, X, [0.0, 0.5], [1.0, 0.5], 0.5, 2, 4)
    assert beta == 0.0 and clip == 1.0, f"got ({beta}, {clip})"
    assert float(losses.max()) == 0.0, f"got {losses.tolist()}"
    assert torch.equal(codes, torch.zeros(2, 4, dtype=torch.int64))
    assert torch.isfinite(scales).all()


def check_bad_bits(solve):
    W = torch.ones(2, 2)
    X = torch.ones(2, 2)
    for bits in (1, 9):
        try:
            solve(W, X, [0.5], [1.0], 0.5, 2, bits)
        except ValueError:
            continue
        raise AssertionError(f"bits={bits} 应抛 ValueError")


def check_loss_grid_matches_best(solve):
    torch.manual_seed(17)
    W = torch.randn(3, 4, dtype=torch.float32)
    X = torch.rand(4, 4, dtype=torch.float32) + 0.1
    ratios = [0.0, 0.5, 1.0]
    clips = [1.0, 0.6, 0.2]
    beta, clip, _, _, losses = solve(W, X, ratios, clips, 0.4, 2, 4)
    assert losses.shape == (3, 3), f"got {tuple(losses.shape)}"
    i, j = ratios.index(beta), clips.index(clip)
    assert torch.isclose(losses[i, j], losses.min()), f"最优组合不是最小 loss：{losses.tolist()}"
    flat = losses.reshape(-1)
    first = int(torch.argmin(flat))
    assert (i, j) == divmod(first, len(clips)), "并列时应取先出现的组合"


def check_random_against_oracle(solve):
    """随机权重 / 校准集 / 网格上与独立 oracle 一致。"""
    g = torch.Generator().manual_seed(20261030)
    for _ in range(60):
        O = int(torch.randint(1, 4, (1,), generator=g))
        C = int(torch.randint(1, 3, (1,), generator=g)) * 2
        T = int(torch.randint(1, 5, (1,), generator=g))
        W = torch.randn(O, C, generator=g, dtype=torch.float32) * 2
        X = torch.rand(T, C, generator=g, dtype=torch.float32) + 0.05
        if C > 1:
            X[:, int(torch.randint(0, C, (1,), generator=g))] *= 5.0
        ratios = [0.0, 0.5, 1.0]
        clips = [1.0, 0.5, 0.2]
        shrink = float(torch.randint(0, 6, (1,), generator=g)) / 10.0
        group = int(torch.randint(1, C // 2 + 1, (1,), generator=g))
        while C % group:
            group -= 1
        bits = int(torch.randint(4, 9, (1,), generator=g))
        got = solve(W, X, ratios, clips, shrink, group, bits)
        want = _oracle(W, X, ratios, clips, shrink, group, bits)
        tag = (O, C, T, shrink, group, bits)
        assert got[0] == want[0], f"{tag}: beta {got[0]} vs {want[0]}"
        assert got[1] == want[1], f"{tag}: clip {got[1]} vs {want[1]}"
        assert torch.allclose(got[2], want[2].to(torch.float32), atol=1e-5), f"{tag}: scales 不符"
        assert torch.equal(got[3], want[3]), f"{tag}: codes 不符"
        assert torch.allclose(got[4], want[4].to(torch.float32), rtol=1e-4, atol=1e-10), f"{tag}: losses 不符"
