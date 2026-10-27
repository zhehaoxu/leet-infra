import torch

EXACT = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
         -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
         0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
         0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
         0.7229568362236023, 1.0]


def _oracle(w, offset):
    """逐标量重算：python 循环构造码本 + 最近邻编码（结构异于参考解的向量化路径）。"""
    off = float(offset)

    def ppf(p):
        return float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)) * 2 ** 0.5)

    vals = [ppf(off + (0.5 - off) * i / 8.0) for i in range(8)]
    vals += [-ppf(off + (0.5 - off) * i / 7.0) for i in range(7)]
    vals.append(0.0)
    vals = sorted(vals)
    mx = max(abs(v) for v in vals)
    table = [v / mx for v in vals]
    m, n = w.shape
    codes = torch.zeros(m, n, dtype=torch.long)
    deq = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = max(abs(float(w[i, t])) for t in range((j // 64) * 64, (j // 64) * 64 + 64))
            if s == 0:
                codes[i, j] = 7
                continue
            u = float(w[i, j]) / s
            best, bc = None, 0
            for k, t in enumerate(table):
                d = abs(u - t)
                if best is None or d < best:
                    best, bc = d, k
            codes[i, j] = bc
            deq[i][j] = table[bc] * s
    return (torch.tensor(table, dtype=torch.float32), codes,
            torch.tensor(deq, dtype=torch.float32))


def check_example(solve):
    torch.manual_seed(0)
    w = torch.full((1, 64), 0.5)
    w[0, 0], w[0, 1], w[0, 2] = 1.0, -1.0, 0.95
    table, codes, deq = solve(w, 0.9677083)
    assert codes[0, :4].tolist() == [15, 0, 15, 12], f"got {codes[0, :4].tolist()}"
    exp = torch.tensor([1.0, -1.0, 1.0, 0.44070983])
    assert torch.allclose(deq[0, :4], exp, atol=1e-5, rtol=1e-5), f"got {deq[0, :4].tolist()}"


check_example.example = True


def check_codebook_matches_reference(solve):
    """offset=0.9677083 的码本必须复现 bitsandbytes 的 16 值精确表（1e-5）。"""
    torch.manual_seed(1)
    w = torch.randn(2, 64)
    table, _, _ = solve(w, 0.9677083)
    exp = torch.tensor(EXACT)
    assert torch.allclose(table, exp.to(torch.float32), atol=1e-5, rtol=1e-5), (
        f"码本偏差 {float((table - exp.to(torch.float32)).abs().max())}")


def check_codebook_anchors(solve):
    """结构锚点：严格递增、T[0]=-1、T[7]=0、T[15]=1、正侧 8 负侧 7（不对称）。"""
    table, _, _ = solve(torch.zeros(1, 64), 0.9677083)
    assert table.numel() == 16, f"码本长度应为 16，got {table.numel()}"
    assert torch.all(table[1:] > table[:-1]), "码本必须严格递增"
    assert abs(float(table[0]) + 1.0) < 1e-6 and abs(float(table[-1]) - 1.0) < 1e-6
    assert abs(float(table[7])) < 1e-6, "码 7 必须是精确 0"
    assert abs(float(table[7]) - float(table[6])) > abs(float(table[8]) - float(table[7])), (
        "正侧间距小于负侧（8+7 的不对称口径）")


def check_matches_oracle(solve):
    """随机 (4,128) 与逐标量重算对拍，包含全零行（块 absmax=0 → 码 7、重建 0）。"""
    torch.manual_seed(21)
    w = torch.randn(4, 128) * 1.7
    w[1, :] = 0.0
    table, codes, deq = solve(w, 0.9677083)
    exp_t, exp_c, exp_d = _oracle(w, 0.9677083)
    assert torch.allclose(table, exp_t, atol=1e-5, rtol=1e-5), "码本与重算不符"
    assert torch.equal(codes, exp_c), "码与重算不符"
    assert torch.allclose(deq, exp_d, atol=1e-5, rtol=1e-5), "重建与重算不符"
    assert int((codes[1] == 7).sum()) == 128 and float(deq[1].abs().max()) == 0.0


def check_block_scaling_linearity(solve):
    """块缩放语义：同一块内 w 整体乘 2 → 码不变、重建值精确乘 2；offset 不同则码本不同。"""
    torch.manual_seed(5)
    w = torch.randn(2, 64)
    _, c1, d1 = solve(w, 0.9677083)
    _, c2, d2 = solve(w * 2.0, 0.9677083)
    assert torch.equal(c1, c2), "整体缩放不应改变码"
    assert torch.allclose(d2, d1 * 2.0, atol=1e-5, rtol=1e-5), "重建值应线性放大"
    t1, _, _ = solve(w, 0.95)
    t2, _, _ = solve(w, 0.99)
    assert float((t1 - t2).abs().max()) > 1e-3, "offset 改变应改变码本"
    try:
        solve(w, 1.0)
        raise AssertionError("offset=1.0 应抛 ValueError")
    except ValueError:
        pass
