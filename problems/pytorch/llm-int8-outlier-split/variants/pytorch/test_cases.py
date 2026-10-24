import torch


def _oracle(W, tau, bits=8):
    """独立纯 Python：逐列 absmax 判离群；普通列 per-tensor RTN；离群列原样。"""
    O, C = W.shape
    qmax = 2 ** (bits - 1) - 1
    out = [bool(max(abs(float(W[o][j])) for o in range(O)) > tau) for j in range(C)]
    non_idx = [j for j in range(C) if not out[j]]
    scale = 1.0
    if non_idx:
        m = max(abs(float(W[o][j])) for o in range(O) for j in non_idx)
        scale = m if m > 0 else 1.0
    recon = []
    for o in range(O):
        row = []
        for j in range(C):
            if out[j]:
                row.append(float(W[o][j]))
            else:
                q = max(-qmax - 1, min(qmax, round(float(W[o][j]) / scale * qmax)))
                row.append(q / qmax * scale)
        recon.append(row)
    return recon, out


def check_example(solve):
    W = torch.tensor([[0.1, 0.2, 9.0, 0.3], [0.4, 0.5, 8.0, 0.6], [0.7, 0.8, 7.0, 0.9]])
    recon, mask = solve(W, 6.0, 8)
    assert torch.equal(mask, torch.tensor([False, False, True, False])), mask
    assert torch.allclose(recon[0, 2:3], torch.tensor([9.0])), recon
    exp = torch.tensor([[0.0992, 0.1984, 9.0, 0.2976],
                        [0.3969, 0.5031, 8.0, 0.6024],
                        [0.7016, 0.8008, 7.0, 0.9000]])
    assert torch.allclose(recon, exp, atol=1e-3), recon


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for _ in range(30):
        O = int(torch.randint(1, 6, (1,)))
        C = int(torch.randint(1, 8, (1,)))
        W = torch.randn(O, C) * 2
        tau = float(0.5 + torch.rand(1) * 3)
        bits = int(torch.randint(2, 9, (1,)))
        recon, mask = solve(W, tau, bits)
        orec, omask = _oracle(W, tau, bits)
        assert torch.equal(mask, torch.tensor(omask)), (mask, omask)
        assert torch.allclose(recon, torch.tensor(orec), atol=1e-4), (recon, orec)


def check_all_outlier_passthrough(solve):
    W = torch.tensor([[10.0, -12.0], [30.0, 20.0]])
    recon, mask = solve(W, 1.0, 8)   # 全列 absmax>1 → 全离群
    assert mask.all(), mask
    assert torch.equal(recon, W), recon


def check_no_outlier_all_quant(solve):
    W = torch.tensor([[0.1, 0.2], [0.3, 0.5]])
    recon, mask = solve(W, 100.0, 2)  # 无离群 → 全部走 2-bit 量化
    assert not mask.any(), mask
    # scale=0.5, qmax=1: 商 0.2/0.4/0.6/1.0 → round 0/0/1/1 → {0.0,0.0,0.5,0.5}
    assert torch.allclose(recon, torch.tensor([[0.0, 0.0], [0.5, 0.5]]), atol=1e-6), recon


check_example.example = True
