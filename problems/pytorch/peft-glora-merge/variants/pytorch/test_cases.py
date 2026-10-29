import torch

W0 = torch.tensor([[0.5, -0.2, 0.1, 0.3],
                   [0.2, 0.4, -0.1, 0.0],
                   [-0.3, 0.1, 0.5, 0.2]])           # (3,4)
B0 = torch.tensor([0.1, -0.2, 0.05])                 # (3,)
X = torch.tensor([[1.0, 0.5, -0.3, 0.2],
                  [0.1, -0.4, 0.6, 0.3]])            # (2,4)

ADAPTER = {
    "A_d": torch.tensor([[0.3, -0.1], [0.2, 0.4], [-0.5, 0.2]]),          # (3,2)
    "A_u": torch.tensor([[0.1, 0.2, -0.3, 0.4], [-0.2, 0.5, 0.1, 0.0]]),  # (2,4)
    "B_d": torch.tensor([[0.1, 0.2], [-0.3, 0.1], [0.0, -0.2]]),          # (3,2)
    "B_u": torch.tensor([[0.4, -0.1, 0.2, 0.0], [0.3, 0.2, -0.5, 0.1]]),  # (2,4)
    "C": torch.tensor([0.5, -0.4, 0.2, 0.1]),                            # (4,)
    "D": torch.tensor([0.2, -0.1, 0.3]),                                 # (3,)
    "E": torch.tensor([0.05, 0.1, -0.15]),                               # (3,)
}

ADAPTER2 = {
    "A_d": torch.tensor([[0.2, 0.3], [-0.1, 0.2], [0.4, -0.3]]),
    "A_u": torch.tensor([[0.3, -0.2, 0.1, 0.2], [0.1, 0.0, 0.4, -0.2]]),
    "B_d": torch.tensor([[-0.2, 0.1], [0.2, -0.3], [0.1, 0.2]]),
    "B_u": torch.tensor([[0.2, 0.3, -0.1, 0.0], [-0.3, 0.1, 0.2, 0.4]]),
    "C": torch.tensor([-0.3, 0.2, 0.0, 0.5]),
    "D": torch.tensor([0.1, 0.2, -0.2]),
    "E": torch.tensor([-0.05, 0.2, 0.1]),
}


def _dense(adapter, prefix):
    """显式秩一分解求和 A = Σ_k d[:,k] ⊗ u[k,:]，与 matmul 写法结构不同。"""
    d = adapter[f"{prefix}_d"].to(torch.float64)
    u = adapter[f"{prefix}_u"].to(torch.float64)
    out = torch.zeros(d.shape[0], u.shape[1], dtype=torch.float64)
    for k in range(d.shape[1]):
        out += d[:, k].unsqueeze(1) * u[k].unsqueeze(0)
    return out


def _merge_ref(w0, b0, adapters):
    w0 = w0.to(torch.float64)
    mul = w0.clone()
    add = b0.to(torch.float64).clone() if b0 is not None else None
    for ad in adapters:
        mul = mul + w0 * _dense(ad, "A") + _dense(ad, "B")
        if add is not None:
            add = add + b0.to(torch.float64) * ad["D"].to(torch.float64) + ad["E"].to(torch.float64) + w0 @ ad["C"].to(torch.float64)
    return mul, add


def _unmerged_forward(x, w0, b0, adapters):
    """PEFT 未合并路径：Σ_i F.linear(x, W0⊙(1+A_i)+B_i, b0⊙(1+D_i)+E_i+W0@C_i) − (n−1)·base。"""
    w0 = w0.to(torch.float64)
    x = x.to(torch.float64)
    base = x @ w0.T
    if b0 is not None:
        base = base + b0.to(torch.float64)
    total = torch.zeros_like(base)
    for ad in adapters:
        adapted = x @ (w0 * (1.0 + _dense(ad, "A")) + _dense(ad, "B")).T
        if b0 is not None:
            adapted = adapted + (b0.to(torch.float64) * (1.0 + ad["D"].to(torch.float64))
                                 + ad["E"].to(torch.float64) + w0 @ ad["C"].to(torch.float64))
        total = total + adapted
    return total - (len(adapters) - 1) * base


def check_example(solve):
    y, wm, bm = solve(X, W0, B0, [ADAPTER])
    exp_w = torch.tensor([[0.625, -0.172, 0.01, 0.356],
                          [0.098, 0.546, -0.208, 0.01],
                          [-0.333, 0.06, 0.685, 0.14]])
    exp_b = torch.tensor([0.55, -0.16, -0.155])
    exp_y = torch.tensor([[1.1572, 0.2754, -0.6355],
                          [0.7941, -0.4904, 0.2407]])
    assert wm.shape == (3, 4) and bm.shape == (3,), f"输出形状错: {tuple(wm.shape)}, {tuple(bm.shape)}"
    assert torch.allclose(wm, exp_w, atol=1e-5, rtol=1e-5), f"W' 错: {wm.tolist()}"
    assert torch.allclose(bm, exp_b, atol=1e-5, rtol=1e-5), f"b' 错: {bm.tolist()}"
    assert torch.allclose(y, exp_y, atol=1e-5, rtol=1e-5), f"y 错: {y.tolist()}"


check_example.example = True


def check_unmerged_forward_matches_merged(solve):
    # 合并路径的前向必须等于 PEFT 未合并路径（逐适配器 effective 权重求和再减去 (n-1) 份基座）
    torch.manual_seed(7)
    cases = [([ADAPTER], 2), ([ADAPTER, ADAPTER2], 3), ([ADAPTER, ADAPTER2, ADAPTER], 2)]
    for adapters, ndim in cases:
        w0 = torch.randn(3, 4)
        b0 = torch.randn(3)
        x = torch.randn(*([2] * (ndim - 1)), 4) if ndim > 1 else torch.randn(4)
        y, _, _ = solve(x, w0, b0, adapters)
        ref = _unmerged_forward(x, w0, b0, adapters).to(torch.float32)
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), f"{len(adapters)} 适配器未合并前向不匹配"
    # 无偏置层同样成立
    w0 = torch.randn(3, 4)
    x = torch.randn(5, 4)
    y, wm, bm = solve(x, w0, None, [ADAPTER, ADAPTER2])
    assert bm is None, "b0 为 None 时 b' 也必须是 None"
    ref = _unmerged_forward(x, w0, None, [ADAPTER, ADAPTER2]).to(torch.float32)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)


def check_unmerge_recovers_base(solve):
    # W0 = (W' − ΣB) / (1 + ΣA)；b0 = (b' − ΣE − Σ(W0 @ C)) / (1 + ΣD)
    adapters = [ADAPTER, ADAPTER2]
    _, wm, bm = solve(X, W0, B0, adapters)
    wm, bm = wm.to(torch.float64), bm.to(torch.float64)
    sum_a = torch.zeros(3, 4, dtype=torch.float64)
    sum_b = torch.zeros(3, 4, dtype=torch.float64)
    sum_d = torch.zeros(3, dtype=torch.float64)
    sum_e = torch.zeros(3, dtype=torch.float64)
    for ad in adapters:
        sum_a += _dense(ad, "A")
        sum_b += _dense(ad, "B")
        sum_d += ad["D"].to(torch.float64)
        sum_e += ad["E"].to(torch.float64)
    w0_rec = (wm - sum_b) / (1.0 + sum_a)
    wc = sum(w0_rec @ ad["C"].to(torch.float64) for ad in adapters)
    b0_rec = (bm - sum_e - wc) / (1.0 + sum_d)
    assert torch.allclose(w0_rec, W0.to(torch.float64), atol=1e-5, rtol=1e-5), f"逆解 W0 错: {w0_rec.tolist()}"
    assert torch.allclose(b0_rec, B0.to(torch.float64), atol=1e-5, rtol=1e-5), f"逆解 b0 错: {b0_rec.tolist()}"


def check_zero_init_identity(solve):
    # peft 初始化 Xd 全零 => A=B=C=D=E=0，合并后与原层完全一致
    zero = {
        "A_d": torch.zeros(3, 2), "A_u": torch.zeros(2, 4),
        "B_d": torch.zeros(3, 2), "B_u": torch.zeros(2, 4),
        "C": torch.zeros(4), "D": torch.zeros(3), "E": torch.zeros(3),
    }
    y, wm, bm = solve(X, W0, B0, [zero])
    assert torch.equal(wm, W0), "零初始化时 W' 必须严格等于 W0"
    assert torch.equal(bm, B0), "零初始化时 b' 必须严格等于 b0"
    base = X @ W0.T + B0
    assert torch.allclose(y, base, atol=1e-6, rtol=0), "零初始化时前向必须等于基座前向"


def check_bias_side_path(solve):
    # A=B=D=E=0 时只有 C 项生效：b' − b0 == W0 @ C，且 W' == W0
    ad = dict(ADAPTER, A_d=torch.zeros(3, 2), A_u=torch.zeros(2, 4),
              B_d=torch.zeros(3, 2), B_u=torch.zeros(2, 4),
              D=torch.zeros(3), E=torch.zeros(3))
    _, wm, bm = solve(X, W0, B0, [ad])
    assert torch.allclose(wm, W0, atol=1e-6, rtol=0), "只有 C 项时权重不应变化"
    assert torch.allclose(bm - B0, W0 @ ad["C"], atol=1e-5, rtol=1e-5), "C 旁路增量应为 W0 @ C"


def check_multi_adapter_no_compounding(solve):
    # 每个适配器的增量都以原始 W0 计算；顺序累积（用已合并权重当基座）是错的
    _, wm, _ = solve(X, W0, B0, [ADAPTER, ADAPTER2])
    ref, _ = _merge_ref(W0, B0, [ADAPTER, ADAPTER2])
    assert torch.allclose(wm, ref.to(torch.float32), atol=1e-5, rtol=1e-5), "多适配器累积与参考不一致"
    w = W0.to(torch.float64)
    for ad in (ADAPTER, ADAPTER2):
        w = w + w * _dense(ad, "A") + _dense(ad, "B")
    assert not torch.allclose(wm.to(torch.float64), w, atol=1e-4, rtol=1e-4), "多适配器出现了顺序复合，增量必须以原始 W0 计算"


def check_order_invariance(solve):
    _, w_ab, b_ab = solve(X, W0, B0, [ADAPTER, ADAPTER2])
    _, w_ba, b_ba = solve(X, W0, B0, [ADAPTER2, ADAPTER])
    assert torch.allclose(w_ab, w_ba, atol=1e-6, rtol=0), "合并顺序不应影响 W'"
    assert torch.allclose(b_ab, b_ba, atol=1e-6, rtol=0), "合并顺序不应影响 b'"


def check_invalid(solve):
    def expect(exc, fn):
        try:
            fn()
        except exc:
            return
        except Exception as err:  # noqa: BLE001
            raise AssertionError(f"应抛 {exc.__name__}，实得 {type(err).__name__}: {err}")
        raise AssertionError(f"应抛 {exc.__name__}，但没有报错")

    bad_r = dict(ADAPTER, B_u=torch.zeros(3, 4))              # r=3 与 A 的 r=2 不一致
    bad_shape = dict(ADAPTER, A_u=torch.zeros(2, 5))          # d_in 不匹配
    no_c = {k: v for k, v in ADAPTER.items() if k != "C"}
    expect(ValueError, lambda: solve(X, W0, B0, []))
    expect(ValueError, lambda: solve(X[:, :3], W0, B0, [ADAPTER]))
    expect(ValueError, lambda: solve(X, W0[:, :3], B0, [ADAPTER]))
    expect(ValueError, lambda: solve(X, W0, B0[:2], [ADAPTER]))
    expect(ValueError, lambda: solve(X, W0, B0, [bad_r]))
    expect(ValueError, lambda: solve(X, W0, B0, [bad_shape]))
    expect(ValueError, lambda: solve(X, W0, B0, [no_c]))
    expect(ValueError, lambda: solve(X, W0[None], B0, [ADAPTER]))
