import torch


def _oracle(scales, bits, block_size):
    """纯 python 逐组重算。"""
    qmax = 2 ** (bits - 1) - 1
    b = len(scales)
    codes, gs, deq = [], [], []
    start = 0
    while start < b:
        end = min(start + block_size, b)
        blk = [float(v) for v in scales[start:end]]
        mx = max(abs(v) for v in blk)
        s2 = mx / qmax if mx > 0 else 1.0
        c = [min(max(round(v / s2), -qmax), qmax) for v in blk]
        codes += [int(v) for v in c]
        gs.append(s2)
        deq += [v * s2 for v in c]
        start = end
    return (torch.tensor(codes, dtype=torch.int64), torch.tensor(gs, dtype=torch.float32),
            torch.tensor(deq, dtype=torch.float32))


def check_example(solve):
    s = torch.tensor([0.25, 0.25, 0.25, 0.25])
    codes, gs, deq = solve(s, 2, 2)
    assert codes.tolist() == [1, 1, 1, 1], f"got {codes.tolist()}"
    assert torch.allclose(gs, torch.tensor([0.25, 0.25]), atol=1e-6), f"got {gs.tolist()}"
    assert torch.allclose(deq, s, atol=1e-6), f"got {deq.tolist()}"


check_example.example = True


def check_group_boundaries(solve):
    """分组是连续 block_size 个、尾部单独成组：B=300、block=256 → 2 组，第二组只由后 44 个定 s2。"""
    torch.manual_seed(2)
    s = torch.cat([torch.full((256,), 0.01), torch.full((44,), 1.0)])
    codes, gs, deq = solve(s, 8, 256)
    assert gs.numel() == 2, f"应为 2 组，got {gs.numel()}"
    assert abs(float(gs[0]) - 0.01 / 127) < 1e-6, f"组 0 的 s2 由前 256 个决定：{float(gs[0])}"
    assert abs(float(gs[1]) - 1.0 / 127) < 1e-6, f"组 1 的 s2 只由后 44 个决定：{float(gs[1])}"
    assert int(codes[0]) == 127 and int(codes[-1]) == 127


def check_error_bound(solve):
    """量化误差 ≤ s2/2；每组绝对值最大元素精确还原。"""
    torch.manual_seed(5)
    s = torch.rand(300) * 1e-2 + 1e-4
    codes, gs, deq = solve(s, 8, 64)
    for g in range(gs.numel()):
        lo = g * 64
        hi = min((g + 1) * 64, 300)
        assert float(gs[g]) >= 0.0
        assert float((s[lo:hi] - deq[lo:hi]).abs().max()) <= float(gs[g]) / 2 + 1e-7
        mx_idx = int(s[lo:hi].abs().argmax())
        assert abs(float(s[lo + mx_idx]) - float(deq[lo + mx_idx])) <= 1e-6


def check_matches_oracle(solve):
    torch.manual_seed(9)
    s = torch.rand(37) * 1e-3
    got = solve(s, 4, 8)
    exp = _oracle(s, 4, 8)
    assert torch.equal(got[0], exp[0]), "码与重算不符"
    assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-5), "二级 scale 与重算不符"
    assert torch.allclose(got[2], exp[2], atol=1e-5, rtol=1e-5), "重建与重算不符"


def check_zero_group_and_validation(solve):
    """全零组 s2=1、码 0、重建 0；bits/block_size 非法抛 ValueError。"""
    s = torch.tensor([0.0, 0.0, 0.5, 0.0])
    codes, gs, deq = solve(s, 4, 2)
    assert codes[0:2].tolist() == [0, 0] and float(gs[0]) == 1.0, f"{codes.tolist()}, {gs.tolist()}"
    assert float(deq[0]) == 0.0 and float(deq[1]) == 0.0
    for bad_b, bad_bs in ((1, 4), (9, 4), (4, 0), (4, -3)):
        try:
            solve(s, bad_b, bad_bs)
            raise AssertionError(f"bits={bad_b}, block={bad_bs} 应抛 ValueError")
        except ValueError:
            pass
