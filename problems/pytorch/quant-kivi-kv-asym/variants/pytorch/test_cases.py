import torch


def _oracle(K, V, bits):
    """逐组 python 重算（与参考解的向量化归约结构无关）。"""
    qmax = 2 ** bits - 1
    T, D = K.shape
    kd = [[0.0] * D for _ in range(T)]
    vd = [[0.0] * D for _ in range(T)]
    ks, kz, vs, vz = [], [], [], []
    for j in range(D):
        col = [float(K[i, j]) for i in range(T)]
        mn, mx = min(col), max(col)
        sc = (mx - mn) / qmax if mx > mn else 1.0
        ks.append(sc)
        kz.append(mn)
        for i in range(T):
            q = min(max(round((col[i] - mn) / sc), 0.0), qmax)
            kd[i][j] = q * sc + mn
    for i in range(T):
        row = [float(V[i, j]) for j in range(D)]
        mn, mx = min(row), max(row)
        sc = (mx - mn) / qmax if mx > mn else 1.0
        vs.append(sc)
        vz.append(mn)
        for j in range(D):
            q = min(max(round((row[j] - mn) / sc), 0.0), qmax)
            vd[i][j] = q * sc + mn
    return (torch.tensor(kd, dtype=torch.float32), torch.tensor(vd, dtype=torch.float32),
            torch.tensor(ks, dtype=torch.float32), torch.tensor(kz, dtype=torch.float32),
            torch.tensor(vs, dtype=torch.float32), torch.tensor(vz, dtype=torch.float32))


def check_example(solve):
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    V = K.clone()
    kd, vd, ks, kz, vs, vz = solve(K, V, 4)
    assert torch.allclose(kd, torch.tensor([[0.0, 1.0], [2.0, 3.0]]), atol=1e-5), f"{kd.tolist()}"
    assert torch.allclose(vd, torch.tensor([[0.0, 1.0], [2.0, 3.0]]), atol=1e-5), f"{vd.tolist()}"
    assert torch.allclose(ks, torch.tensor([2 / 15, 2 / 15]), atol=1e-5), f"{ks.tolist()}"
    assert torch.allclose(kz, torch.tensor([0.0, 1.0]), atol=1e-5)
    assert torch.allclose(vs, torch.tensor([1 / 15, 1 / 15]), atol=1e-5)
    assert torch.allclose(vz, torch.tensor([0.0, 2.0]), atol=1e-5)


check_example.example = True


def check_granularity(solve):
    """K 参数长度 = D（每通道一套）、V 参数长度 = T（每 token 一套），且数值与逐组 min/max 一致。"""
    torch.manual_seed(3)
    K = torch.randn(5, 7) * 3
    V = torch.randn(5, 7) + 2.0
    _, _, ks, kz, vs, vz = solve(K, V, 4)
    assert ks.shape == (7,) and kz.shape == (7,), f"K 参数应为 (7,)，got {ks.shape}/{kz.shape}"
    assert vs.shape == (5,) and vz.shape == (5,), f"V 参数应为 (5,)，got {vs.shape}/{vz.shape}"
    exp_ks = (K.amax(0) - K.amin(0)) / 15
    exp_kz = K.amin(0)
    exp_vs = (V.amax(1) - V.amin(1)) / 15
    exp_vz = V.amin(1)
    assert torch.allclose(ks, exp_ks, atol=1e-5) and torch.allclose(kz, exp_kz, atol=1e-5)
    assert torch.allclose(vs, exp_vs, atol=1e-5) and torch.allclose(vz, exp_vz, atol=1e-5)


def check_roundtrip_bound(solve):
    """非对称往返：每元素误差 ≤ Δ/2，且每组的 min/max 落在码 0 / 2^b-1 上。"""
    torch.manual_seed(9)
    K = torch.randn(6, 4) * 5
    V = torch.randn(6, 4) * 5 + 1.0
    kd, vd, ks, _, vs, _ = solve(K, V, 4)
    assert float((K - kd).abs().max()) <= float(ks.max()) / 2 + 1e-5
    assert float((V - vd).abs().max()) <= float(vs.max()) / 2 + 1e-5
    assert torch.allclose(kd.amin(0), K.amin(0), atol=1e-5), "K 每通道 min 应精确还原"
    assert torch.allclose(vd.amax(1), V.amax(1), atol=1e-5), "V 每 token max 应精确还原"


def check_matches_oracle(solve):
    torch.manual_seed(17)
    K = torch.randn(4, 3) * 2
    V = torch.randn(4, 3) * 0.5 - 1.0
    got = solve(K, V, 6)
    exp = _oracle(K, V, 6)
    for g, e in zip(got, exp):
        assert torch.allclose(g, e, atol=1e-5, rtol=1e-5), f"max diff {float((g - e).abs().max())}"


def check_degenerate_and_validation(solve):
    """常量行/列精确还原；bits 越界抛 ValueError。"""
    K = torch.tensor([[1.0, -2.0], [1.0, -2.0]])   # 每通道常量
    V = torch.tensor([[3.0, 3.0], [-4.0, -4.0]])   # 每 token 常量
    kd, vd, _, _, _, _ = solve(K, V, 4)
    assert torch.allclose(kd, K, atol=1e-6) and torch.allclose(vd, V, atol=1e-6), (
        f"退化组应精确还原，got {kd.tolist()} / {vd.tolist()}")
    for bad in (1, 9):
        try:
            solve(K, V, bad)
            raise AssertionError(f"bits={bad} 应抛 ValueError")
        except ValueError:
            pass
