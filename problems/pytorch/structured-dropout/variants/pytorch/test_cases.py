import torch

# 示例：手写 (2, 3, 2, 2) 输入，droppath p=0.5、mask=[1, 0]（样本 0 存活、样本 1 整丢）
EX_X = [[[[0.5, -1.0], [2.0, 0.0]],
         [[3.5, 1.5], [-2.5, 4.0]],
         [[0.25, -0.75], [1.25, 2.75]]],
        [[[-3.0, 6.5], [0.5, -1.5]],
         [[2.25, -4.0], [5.5, 0.0]],
         [[1.0, 3.0], [-2.0, 7.25]]]]
EX_P = 0.5
EX_MASK = [1.0, 0.0]
# droppath 输出 (2, 3, 2, 2)（实测回填：样本 0 存活 → x/0.5，样本 1 被整丢 → ±0.0）
EXAMPLE_DROPPATH = [[[[1.0, -2.0], [4.0, 0.0]],
                     [[7.0, 3.0], [-5.0, 8.0]],
                     [[0.5, -1.5], [2.5, 5.5]]],
                    [[[0.0, 0.0], [0.0, 0.0]],
                     [[0.0, 0.0], [0.0, 0.0]],
                     [[0.0, 0.0], [0.0, 0.0]]]]


def _det(*shape, seed=0, lo=-1.0, hi=1.0):
    """无 RNG 的确定性伪随机张量：黄金比例哈希，CPU/float32 下完全可复现。"""
    numel = 1
    for d in shape:
        numel *= d
    base = torch.arange(numel, dtype=torch.float64)
    v = ((base + 1.0) * (seed + 1) * 0.6180339887498949) % 1.0
    return (v * (hi - lo) + lo).to(torch.float32).reshape(*shape)


def _det_mask(*shape, seed=0, keep=0.5):
    """确定性布尔 mask：黄金比例哈希阈值化（无 RNG）。"""
    numel = 1
    for d in shape:
        numel *= d
    base = torch.arange(numel, dtype=torch.float64)
    v = ((base + 1.0) * (seed + 1) * 0.6180339887498949) % 1.0
    return (v < keep).reshape(*shape)


def _oracle(x, p, mask, mode):
    """独立循环 oracle：纯 python 浮点逐元素做手工广播 + 1/(1−p) 缩放，与张量化解法分路径。"""
    scale = 1.0 / (1.0 - float(p))
    xl = x.tolist()
    if mode == "droppath":
        mflat = [float(v) for v in mask.reshape(-1).tolist()]

        def walk(sub, m):
            if isinstance(sub, list):
                return [walk(s2, m) for s2 in sub]
            return sub * m * scale

        out = [walk(xl[b], mflat[b]) for b in range(x.shape[0])]
        return torch.tensor(out, dtype=x.dtype)
    if mode == "dropout2d":
        n, c, h, w = x.shape
        ml = [list(map(float, r)) for r in mask.reshape(n, c).tolist()]
        out = [[[[xl[ni][ci][hi][wi] * ml[ni][ci] * scale for wi in range(w)]
                 for hi in range(h)] for ci in range(c)] for ni in range(n)]
        return torch.tensor(out, dtype=x.dtype)
    if mode == "dropout3d":
        n, c, d, h, w = x.shape
        ml = [list(map(float, r)) for r in mask.reshape(n, c).tolist()]
        out = [[[[[xl[ni][ci][di][hi][wi] * ml[ni][ci] * scale for wi in range(w)]
                  for hi in range(h)] for di in range(d)] for ci in range(c)] for ni in range(n)]
        return torch.tensor(out, dtype=x.dtype)
    raise ValueError(mode)


def check_example(solve):
    x = torch.tensor(EX_X)
    y = solve(x, EX_P, torch.tensor(EX_MASK), mode="droppath")
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y, torch.tensor(EXAMPLE_DROPPATH), atol=1e-6), y.tolist()
    # 探针：样本 1 被整丢 → 严格全 0；样本 0 存活 → 恰为 x/(1−p)
    assert torch.all(y[1] == 0), "被丢样本必须全 0"
    assert torch.allclose(y[0], x[0] / 0.5, atol=1e-6)
    # mask (B,1,1,1) 形式与 (B,) 等价
    y2 = solve(x, EX_P, torch.tensor(EX_MASK).reshape(2, 1, 1, 1), mode="droppath")
    assert torch.equal(y, y2)


check_example.example = True


def check_droppath_matches_oracle(solve):
    for rank, tail, seed in [(4, (3, 4, 5), 1), (3, (4, 2), 2), (2, (6,), 3)]:
        b = 4
        x = _det(b, *tail, seed=seed)
        for p in (0.0, 0.25, 0.5, 0.875):
            mask = _det_mask(b, seed=10 * seed + int(p * 100))
            y = solve(x, p, mask, mode="droppath")
            exp = _oracle(x, p, mask, mode="droppath")
            assert torch.allclose(y, exp, atol=1e-6, rtol=1e-6), (rank, p)
            assert y.shape == x.shape and y.dtype == x.dtype
            # (B,1,...,1) 形式与 (B,) 等价
            y2 = solve(x, p, mask.reshape(b, *([1] * (rank - 1))), mode="droppath")
            assert torch.equal(y, y2), (rank, p)
    # 0/1 浮点 mask 与 bool mask 同义
    x = _det(3, 2, 2, seed=9)
    mb = _det_mask(3, seed=11)
    mf = mb.to(torch.float32)
    assert torch.equal(solve(x, 0.5, mb, mode="droppath"),
                       solve(x, 0.5, mf, mode="droppath"))


def check_droppath_probes(solve):
    x = _det(4, 3, 5, 5, seed=4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y = solve(x, 0.25, mask, mode="droppath")
    for b in range(4):
        if mask[b].item() == 0:
            assert torch.all(y[b] == 0), f"样本 {b} 被丢必须整体为 0"
        else:
            assert torch.allclose(y[b], x[b] / 0.75, atol=1e-6, rtol=1e-6), \
                f"存活样本 {b} 应为 x/(1−p)"
    # mask 全 1：y == x/(1−p)
    y1 = solve(x, 0.25, torch.ones(4), mode="droppath")
    assert torch.allclose(y1, x / 0.75, atol=1e-6, rtol=1e-6)


def check_dropout2d(solve):
    x = _det(2, 3, 4, 5, seed=5)
    for mform in ("nc", "nc11"):
        mask = _det_mask(2, 3, seed=6)
        if mform == "nc11":
            mask = mask.reshape(2, 3, 1, 1)
        y = solve(x, 0.4, mask, mode="dropout2d")
        exp = _oracle(x, 0.4, mask, mode="dropout2d")
        assert torch.allclose(y, exp, atol=1e-6, rtol=1e-6), mform
        mflat = mask.reshape(2, 3)
        for ni in range(2):
            for ci in range(3):
                if mflat[ni, ci].item() == 0:
                    assert torch.all(y[ni, ci] == 0), "被丢通道必须整体为 0"
                else:
                    assert torch.allclose(y[ni, ci], x[ni, ci] / 0.6, atol=1e-6, rtol=1e-6)
    # mask 全 1：y == x/(1−p)，两种 mask 形状等价
    ones = torch.ones(2, 3)
    y1 = solve(x, 0.4, ones, mode="dropout2d")
    assert torch.allclose(y1, x / 0.6, atol=1e-6, rtol=1e-6)
    assert torch.equal(y1, solve(x, 0.4, ones.reshape(2, 3, 1, 1), mode="dropout2d"))


def check_dropout3d(solve):
    x = _det(2, 3, 2, 3, 4, seed=7)
    for mform in ("nc", "nc11"):
        mask = _det_mask(2, 3, seed=8)
        if mform == "nc11":
            mask = mask.reshape(2, 3, 1, 1, 1)
        y = solve(x, 0.5, mask, mode="dropout3d")
        assert torch.allclose(y, _oracle(x, 0.5, mask, mode="dropout3d"),
                              atol=1e-6, rtol=1e-6), mform
        mflat = mask.reshape(2, 3)
        assert torch.all(y[mflat == 0] == 0), "被丢通道必须整体为 0"
        assert torch.allclose(y[mflat == 1], x[mflat == 1] * 2.0, atol=1e-6, rtol=1e-6)


def check_p0_identity(solve):
    x4 = _det(2, 3, 4, 4, seed=10)
    x5 = _det(2, 3, 2, 3, 4, seed=11)
    assert torch.equal(solve(x4, 0.0, torch.ones(2, 3), mode="dropout2d"), x4), \
        "p=0 且 mask 全 1 必须严格恒等"
    assert torch.equal(solve(x4, 0.0, torch.ones(2), mode="droppath"), x4)
    assert torch.equal(solve(x5, 0.0, torch.ones(2, 3), mode="dropout3d"), x5)
    # p=0 且 mask 含 0：y == x·mask 逐元素精确
    mixed = torch.tensor([1.0, 0.0])
    y = solve(x4, 0.0, mixed, mode="droppath")
    assert torch.equal(y, x4 * mixed.reshape(2, 1, 1, 1))


def check_float64(solve):
    x2 = _det(3, 4, seed=12).to(torch.float64)
    m2 = _det_mask(3, seed=13)
    y = solve(x2, 0.5, m2, mode="droppath")
    assert y.dtype == torch.float64
    assert torch.allclose(y, _oracle(x2, 0.5, m2, "droppath"), atol=1e-12, rtol=1e-12)
    x4 = _det(2, 3, 2, 2, seed=14).to(torch.float64)
    m4 = _det_mask(2, 3, seed=15)
    y4 = solve(x4, 0.25, m4, mode="dropout2d")
    assert y4.dtype == torch.float64
    assert torch.allclose(y4, _oracle(x4, 0.25, m4, "dropout2d"), atol=1e-12, rtol=1e-12)
    x5 = _det(2, 3, 2, 2, 2, seed=16).to(torch.float64)
    y5 = solve(x5, 0.5, torch.ones(2, 3), mode="dropout3d")
    assert torch.allclose(y5, x5 * 2.0, atol=1e-12, rtol=1e-12)


def check_invalid_mode(solve):
    x = _det(2, 3, 2, 2, seed=17)
    for bad in ("dropout1d", "droppath2d", "", "DropPath"):
        try:
            solve(x, 0.5, torch.ones(2), mode=bad)
        except ValueError:
            continue
        raise AssertionError(f"非法 mode {bad!r} 应抛 ValueError")


def check_p_validation(solve):
    x = _det(2, 2, 2, seed=18)
    mask = torch.ones(2)
    for bad in (1.0, 1.5, -0.1, 2.0, float("inf"), float("nan")):
        try:
            solve(x, bad, mask, mode="droppath")
        except ValueError:
            continue
        raise AssertionError(f"p={bad} 应抛 ValueError")


def check_mask_shape_validation(solve):
    x3 = _det(2, 3, 4, seed=19)  # droppath（3 维 x）：允许 (2,) / (2,1,1)
    for bad in (torch.ones(2, 3), torch.ones(3), torch.ones(2, 1, 1, 1), torch.ones(2, 1)):
        try:
            solve(x3, 0.5, bad, mode="droppath")
        except ValueError:
            continue
        raise AssertionError(f"droppath mask 形状 {tuple(bad.shape)} 应抛 ValueError")
    x4 = _det(2, 3, 4, 5, seed=20)  # dropout2d：允许 (2,3) / (2,3,1,1)
    for bad in (torch.ones(2,), torch.ones(2, 3, 4, 5), torch.ones(2, 3, 1),
                torch.ones(2, 4, 1, 1), torch.ones(3, 3)):
        try:
            solve(x4, 0.5, bad, mode="dropout2d")
        except ValueError:
            continue
        raise AssertionError(f"dropout2d mask 形状 {tuple(bad.shape)} 应抛 ValueError")
    x5 = _det(2, 3, 2, 2, 2, seed=21)  # dropout3d：允许 (2,3) / (2,3,1,1,1)
    for bad in (torch.ones(2, 3, 1, 1), torch.ones(2, 3, 1, 1, 1, 1)):
        try:
            solve(x5, 0.5, bad, mode="dropout3d")
        except ValueError:
            continue
        raise AssertionError(f"dropout3d mask 形状 {tuple(bad.shape)} 应抛 ValueError")
    # 秩校验
    for bad_call in (lambda: solve(_det(2, 3, 4, seed=22), 0.5, torch.ones(2, 3), mode="dropout2d"),
                     lambda: solve(_det(2, 3, 2, 2, seed=23), 0.5, torch.ones(2, 3), mode="dropout3d")):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("x 秩与 mode 不符应抛 ValueError")
