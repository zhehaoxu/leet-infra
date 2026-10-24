import torch


def _oracle(W, tau, bits=8):
    """独立纯 Python：逐元素离群保留；其余逐行 per-row 对称 RTN。"""
    O, C = W.shape
    qmax = 2 ** (bits - 1) - 1
    Wl = W.tolist()
    mask = [[abs(Wl[o][j]) > tau for j in range(C)] for o in range(O)]
    recon = []
    for o in range(O):
        non = [j for j in range(C) if not mask[o][j]]
        scale = (max(abs(Wl[o][j]) for j in non) if non else 1.0) or 1.0
        row = []
        for j in range(C):
            if mask[o][j]:
                row.append(float(Wl[o][j]))
            else:
                q = max(-qmax - 1, min(qmax, round(float(Wl[o][j]) / scale * qmax)))
                row.append(q / qmax * scale)
        recon.append(row)
    return recon, mask


def check_example(solve):
    W = torch.tensor([[0.1, 9.0, 0.2], [0.4, 0.5, 8.0]])
    recon, mask = solve(W, 6.0, 8)
    assert torch.equal(mask, torch.tensor([[False, True, False], [False, False, True]])), mask
    exp = torch.tensor([[0.1008, 9.0, 0.2], [0.4016, 0.5, 8.0]])
    assert torch.allclose(recon, exp, atol=1e-3), recon


def check_matches_oracle(solve):
    torch.manual_seed(9)
    for _ in range(30):
        O = int(torch.randint(1, 6, (1,)))
        C = int(torch.randint(1, 8, (1,)))
        W = torch.randn(O, C) * 3
        tau = float(0.3 + torch.rand(1) * 2.7)
        bits = int(torch.randint(2, 9, (1,)))
        recon, mask = solve(W, tau, bits)
        orec, omask = _oracle(W, tau, bits)
        assert torch.equal(mask, torch.tensor(omask)), (mask, omask)
        assert torch.allclose(recon, torch.tensor(orec), atol=1e-4), (recon, orec)


def check_finer_than_column(solve):
    # 同一列里只豁免真正的大元素：col0 = [9, 0.1]，只有 9 离群、0.1 仍被量化
    W = torch.tensor([[9.0, 0.2], [0.1, 0.4]])
    recon, mask = solve(W, 6.0, 8)
    assert torch.equal(mask, torch.tensor([[True, False], [False, False]])), mask
    assert float(recon[0, 0]) == 9.0            # 离群原样
    # 0.1 落在 row1（scale=0.4）的 8-bit 格点上，非原样但极接近
    assert abs(float(recon[1, 0]) - 0.1) < 1e-3, recon


def check_all_outlier_passthrough(solve):
    W = torch.tensor([[10.0, -12.0], [30.0, 20.0]])
    recon, mask = solve(W, 1.0, 8)   # 全部 |W|>1 → 全离群，原样返回
    assert mask.all(), mask
    assert torch.equal(recon, W), recon


check_example.example = True
