import torch

CLIPMIN = 1e-4


def _oracle(xmax, xmin, W, num_groups, bits):
    """纯 python 逐通道重算：稳定排序 + 均衡切段 + 组内非对称 fake-quant。"""
    xmax = [float(v) for v in xmax]
    xmin = [float(v) for v in xmin]
    W = [[float(v) for v in row] for row in W]
    C = len(xmax)
    span = [xmax[i] - xmin[i] for i in range(C)]
    perm = sorted(range(C), key=lambda i: (span[i], i))
    sizes = [C // num_groups + (1 if j < C % num_groups else 0) for j in range(num_groups)]

    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scales, zeros = [], []
    dequant = [[0.0] * C for _ in range(len(W))]
    start = 0
    for j in range(num_groups):
        idx = perm[start:start + sizes[j]]
        start += sizes[j]
        gmax = max(xmax[i] for i in idx)
        gmin = min(xmin[i] for i in idx)
        scale = min(max((gmax - gmin) * (2.0 ** -bits), CLIPMIN), 1e4)
        zero = float(min(max(round((gmax + gmin) * (-0.5 / scale)), -1e4), 1e4))
        scales.append(scale)
        zeros.append(zero)
        for i in idx:
            for o in range(len(W)):
                q = round(W[o][i] / scale) + zero
                q = min(max(q, qmin), qmax) - zero
                dequant[o][i] = q * scale

    return perm, sizes, scales, zeros, dequant


def check_example(solve):
    xmax = torch.tensor([8.0, 0.5, 8.4, 0.6], dtype=torch.float32)
    xmin = torch.tensor([-8.0, -0.5, -8.4, -0.6], dtype=torch.float32)
    W = torch.tensor([[8.0, 0.5, 8.4, 0.6], [-8.0, -0.5, -8.4, -0.6]], dtype=torch.float32)
    out = solve(xmax, xmin, W, 2, 4)
    assert out["perm"].tolist() == [1, 3, 0, 2], f"got {out['perm'].tolist()}"
    assert out["counts"].tolist() == [2, 2], f"got {out['counts'].tolist()}"
    assert torch.allclose(out["scales"], torch.tensor([0.075, 1.05]), atol=1e-6), \
        f"got {out['scales'].tolist()}"
    assert out["zero_points"].tolist() == [0, 0], f"got {out['zero_points'].tolist()}"
    expect = torch.tensor([[7.35, 0.525, 7.35, 0.525], [-8.4, -0.525, -8.4, -0.6]])
    assert torch.allclose(out["dequant"], expect, atol=1e-5), f"got {out['dequant'].tolist()}"


check_example.example = True


def check_perm_is_span_sorted(solve):
    """perm 必须是 0..C-1 的排列，且按 (跨度, 通道号) 稳定升序。"""
    torch.manual_seed(7)
    C, g = 23, 4
    xmax = torch.rand(C) * 5
    xmin = xmax - torch.rand(C) * 3
    W = torch.randn(3, C)
    out = solve(xmax, xmin, W, g, 4)
    perm = out["perm"].tolist()
    assert sorted(perm) == list(range(C)), f"perm 不是排列：{perm}"
    span = (xmax - xmin).tolist()
    keys = [(span[i], i) for i in perm]
    assert keys == sorted(keys), f"perm 未按 (跨度, 通道号) 升序：{perm}"


def check_balanced_groups(solve):
    """counts 之和为 C，每段长度只能是 C//g 或 C//g + 1，且多出来的 +1 落在前 C%g 段。"""
    torch.manual_seed(11)
    C, g = 17, 5
    xmax = torch.rand(C) + 0.1
    xmin = -torch.rand(C)
    W = torch.randn(2, C)
    counts = solve(xmax, xmin, W, g, 3)["counts"].tolist()
    base, rem = divmod(C, g)
    assert sum(counts) == C, f"counts 总和 {sum(counts)} != {C}"
    assert counts == [base + (1 if j < rem else 0) for j in range(g)], f"got {counts}"


def check_group_scales(solve):
    """每组 scale/zero 必须由该段内通道 max/min 唯一确定，且 clamp 生效。"""
    torch.manual_seed(13)
    C, g, bits = 20, 3, 4
    xmax = torch.rand(C) * 4
    xmin = xmax - torch.rand(C) * 2
    W = torch.randn(4, C)
    out = solve(xmax, xmin, W, g, bits)
    perm = out["perm"].tolist()
    counts = out["counts"].tolist()
    scales = out["scales"].tolist()
    zeros = out["zero_points"].tolist()
    start = 0
    for j in range(g):
        idx = perm[start:start + counts[j]]
        start += counts[j]
        gmax = max(xmax[i].item() for i in idx)
        gmin = min(xmin[i].item() for i in idx)
        sc = min(max((gmax - gmin) * 2.0 ** -bits, CLIPMIN), 1e4)
        z = min(max(round((gmax + gmin) * (-0.5 / sc)), -1e4), 1e4)
        assert abs(scales[j] - sc) <= 1e-6 * max(1.0, abs(sc)), f"第 {j} 组 scale {scales[j]} vs {sc}"
        assert zeros[j] == int(z), f"第 {j} 组 zero {zeros[j]} vs {int(z)}"


def check_matches_oracle(solve):
    """40 组随机配置与纯 python oracle 逐项对齐（perm/counts/scale/zero/重建）。"""
    torch.manual_seed(29)
    for _ in range(40):
        C = int(torch.randint(1, 41, (1,)))
        g = int(torch.randint(1, C + 1, (1,)))
        bits = int(torch.randint(2, 9, (1,)))
        O = int(torch.randint(1, 6, (1,)))
        xmax = torch.randn(C).abs() * 3 + torch.randint(0, 3, (C,)).float()
        xmin = xmax - torch.rand(C) * 4
        W = torch.randn(O, C) * 2
        out = solve(xmax, xmin, W, g, bits)
        e_perm, e_sizes, e_scales, e_zeros, e_deq = _oracle(
            xmax.tolist(), xmin.tolist(), W.tolist(), g, bits)
        assert out["perm"].tolist() == e_perm, f"perm {out['perm'].tolist()} vs {e_perm}"
        assert out["counts"].tolist() == e_sizes, f"counts {out['counts'].tolist()} vs {e_sizes}"
        assert torch.allclose(out["scales"], torch.tensor(e_scales), atol=1e-6), \
            f"scales {out['scales'].tolist()} vs {e_scales}"
        assert out["zero_points"].tolist() == [int(z) for z in e_zeros], \
            f"zeros {out['zero_points'].tolist()} vs {e_zeros}"
        assert torch.allclose(out["dequant"], torch.tensor(e_deq), atol=1e-5), "重建不符"


def check_quantization_grid(solve):
    """重建值必须落在整数格点上：q=round(deq/scale)+zero 为整数且被 clamp 在 [qmin, qmax]。"""
    torch.manual_seed(31)
    C, g, bits = 12, 3, 4
    xmax = torch.rand(C) * 6 + 0.5
    xmin = -torch.rand(C) * 6 - 0.5
    W = torch.randn(3, C) * 3
    out = solve(xmax, xmin, W, g, bits)
    perm = out["perm"].tolist()
    counts = out["counts"].tolist()
    scales = out["scales"].tolist()
    zeros = out["zero_points"].tolist()
    deq = out["dequant"]
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    start = 0
    for j in range(g):
        idx = perm[start:start + counts[j]]
        start += counts[j]
        for i in idx:
            q = deq[:, i].to(torch.float64) / scales[j] + zeros[j]
            qint = q.round()
            assert torch.allclose(q, qint, atol=1e-4), f"通道 {i} 不在 {scales[j]} 的整数格点上"
            assert float(qint.max()) <= qmax and float(qint.min()) >= qmin, \
                f"通道 {i} 的整数超出 [{qmin}, {qmax}]：{qint.tolist()}"


def check_non_identity_and_single_group(solve):
    """重排必须真的发生；单组退化时 perm 仍然有序、counts=[C]、scale 由全体通道决定。"""
    xmax = torch.tensor([0.6, 0.5, 9.0, 8.0])
    xmin = torch.tensor([-0.6, -0.5, -9.0, -8.0])
    W = torch.tensor([[0.61, 0.49, 9.0, -8.0]])
    out = solve(xmax, xmin, W, 2, 4)
    assert out["perm"].tolist() == [1, 0, 3, 2], f"got {out['perm'].tolist()}"
    assert out["perm"].tolist() != sorted(out["perm"].tolist()), "重排不应是恒等映射"
    single = solve(xmax, xmin, W, 1, 4)
    assert single["counts"].tolist() == [4]
    assert single["perm"].tolist() == [1, 0, 3, 2], f"got {single['perm'].tolist()}"
    assert abs(float(single["scales"][0]) - 1.125) < 1e-6, f"got {single['scales'].tolist()}"
    assert single["zero_points"].tolist() == [0]
    assert torch.allclose(single["dequant"], torch.tensor([[1.125, 0.0, 7.875, -7.875]]),
                          atol=1e-5), f"got {single['dequant'].tolist()}"


def check_constant_channels(solve):
    """跨度全为 0：scale 落到 CLIPMIN；零通道重建为 0，非零常量由 zero clamp 决定"""
    xmax = torch.tensor([0.0, 0.0, 0.0])
    xmin = xmax.clone()
    W = torch.zeros(1, 3)
    out = solve(xmax, xmin, W, 2, 4)
    assert torch.allclose(out["scales"], torch.tensor([CLIPMIN, CLIPMIN]), atol=1e-9)
    assert out["zero_points"].tolist() == [0, 0]
    assert torch.allclose(out["dequant"], torch.zeros(1, 3), atol=1e-9)

    xmax = torch.tensor([1.0, 2.0, -3.0])
    xmin = xmax.clone()
    W = torch.randn(2, 3)
    got = solve(xmax, xmin, W, 2, 4)
    _, e_sizes, e_scales, e_zeros, e_deq = _oracle(
        xmax.tolist(), xmin.tolist(), W.tolist(), 2, 4)
    assert got["counts"].tolist() == e_sizes
    assert torch.allclose(got["scales"], torch.tensor(e_scales), atol=1e-6)
    assert got["zero_points"].tolist() == [int(z) for z in e_zeros], \
        f"got {got['zero_points'].tolist()} vs {e_zeros}"
    assert torch.allclose(got["dequant"], torch.tensor(e_deq), atol=1e-5)


def check_validation(solve):
    """类型/范围/形状/有限性/顺序非法时抛 ValueError。"""
    good_max = torch.tensor([1.0, 2.0])
    good_min = torch.tensor([-1.0, 0.0])
    W = torch.zeros(1, 2)

    def expect_error(*args, **kwargs):
        try:
            solve(*args, **kwargs)
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError：args={args} kwargs={kwargs}")

    expect_error(good_max, good_min, W, True, 4)
    expect_error(good_max, good_min, W, 2, 4.0)
    expect_error(good_max, good_min, W, 0, 4)
    expect_error(good_max, good_min, W, 3, 4)
    expect_error(good_max, good_min, W, 2, 1)
    expect_error(good_max, good_min, W, 2, 9)
    expect_error(good_max, good_min, torch.zeros(1, 3), 2, 4)
    expect_error(good_max, torch.tensor([-1.0]), W, 2, 4)
    expect_error(torch.tensor([1.0, float("nan")]), good_min, W, 2, 4)
    expect_error(good_max, torch.tensor([-1.0, 3.0]), W, 2, 4)
    expect_error(torch.zeros(2, 2), good_min, W, 2, 4)
