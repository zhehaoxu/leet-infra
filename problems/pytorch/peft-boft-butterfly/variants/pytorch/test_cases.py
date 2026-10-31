import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _cayley_np(block):
    h = block.shape[0]
    skew = 0.5 * (block - block.T)
    eye = np.eye(h)
    return np.linalg.solve(eye + skew, eye - skew)


def _perm_np(dim, block, stage):
    """显式按「段 -> 偶组在前、奇组在后」重排，逐步列出新位置对应的原位置。"""
    r = block // 2
    seg = block * (2 ** stage)
    perm = []
    for base in range(0, dim, seg):
        groups = seg // r
        order = [g for g in range(groups) if g % 2 == 0] + [g for g in range(groups) if g % 2 == 1]
        for g in order:
            for t in range(r):
                perm.append(base + g * r + t)
    return perm


def _oracle(x, w0, b0, R, s):
    x = np.asarray(x, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    stages, blocks, height, _ = R.shape
    dim = blocks * height
    M = np.eye(dim)
    for i in range(stages):
        B = np.zeros((dim, dim))
        for j in range(blocks):
            q = _cayley_np(R[i, j])
            B[j * height:(j + 1) * height, j * height:(j + 1) * height] = q
        perm = _perm_np(dim, height, i)
        P = np.zeros((dim, dim))
        for k, pk in enumerate(perm):
            P[k, pk] = 1.0
        M = P @ B @ P.T @ M
    weight = (M @ w0.T).T * s
    flat = x.reshape(-1, dim)
    y = flat @ weight.T
    if b0 is not None:
        y = y + np.asarray(b0, dtype=np.float64)
    return y.reshape(x.shape[:-1] + (w0.shape[0],)), M


def _rand(seed, *shape, low=-1.0, high=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * (high - low) + low


def check_example(solve):
    # d_in = d_out = 2、block size = 2、单级：此时置换是单位，M 就是一个 2x2 的 Cayley 旋转
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    w0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    b0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    R = torch.tensor([[[[0.0, 2.0], [-2.0, 0.0]]]], dtype=torch.float64)
    s = torch.tensor([[2.0], [1.0]], dtype=torch.float64)
    y, M = solve(x, w0, b0, R, s)
    # t = (R01 - R10)/2 = 2 -> Q = 1/(1+4) * [[1-4, -4], [4, 1-4]] = [[-0.6, -0.8], [0.8, -0.6]]
    assert _close(M, torch.tensor([[-0.6, -0.8], [0.8, -0.6]], dtype=torch.float64), 1e-6), M
    assert y.shape == (1, 2) and M.shape == (2, 2)
    assert y.dtype == torch.float32 and M.dtype == torch.float32
    # W' = (M @ I).T * s = M.T * s，第 0 行再乘 2
    expect_W = M.to(torch.float64).T * s
    assert _close(y, x.to(torch.float64) @ expect_W.T + b0, 1e-6), y


check_example.example = True


def check_orthogonal_rotation(solve):
    in_dim = 8
    R = _rand(1, 3, in_dim // 2, 2, 2, low=-0.8, high=0.8)
    w0 = _rand(2, 5, in_dim)
    s = _rand(3, 5, 1, low=0.5, high=1.5)
    _, M = solve(torch.zeros(1, in_dim, dtype=torch.float64), w0, None, R, s)
    M64 = M.to(torch.float64)
    eye = torch.eye(in_dim, dtype=torch.float64)
    assert _close(M64 @ M64.T, eye, 1e-6), M64 @ M64.T
    assert _close(M64.T @ M64, eye, 1e-6)
    assert abs(float(torch.linalg.det(M64)) - 1.0) < 1e-6


def check_zero_R_is_identity(solve):
    in_dim, out_dim = 8, 6
    R = torch.zeros(3, in_dim // 2, 2, 2, dtype=torch.float64)
    w0 = _rand(11, out_dim, in_dim)
    b0 = _rand(12, out_dim)
    x = _rand(13, 2, in_dim)
    s = _rand(14, out_dim, 1, low=0.5, high=1.5)
    y, M = solve(x, w0, b0, R, s)
    # R=0 => 每块 Cayley 给出单位阵，整条旋转是单位阵：只剩逐行缩放
    assert _close(M, torch.eye(in_dim, dtype=torch.float64), 1e-9), M
    plain = x.to(torch.float64) @ (w0.to(torch.float64) * s.to(torch.float64)).T + b0.to(torch.float64)
    assert _close(y, plain, 1e-6), y


def check_only_skew_part_matters(solve):
    in_dim = 8
    R = _rand(21, 2, in_dim // 2, 2, 2, low=-0.8, high=0.8)
    sym = torch.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            a = _rand(100 + i * 10 + j, 2, 2)
            sym[i, j] = 0.5 * (a + a.T)  # 纯对称部分
    w0 = _rand(22, 4, in_dim)
    s = _rand(23, 4, 1, low=0.5, high=1.5)
    x = _rand(24, 2, in_dim)
    _, M1 = solve(x, w0, None, R, s)
    _, M2 = solve(x, w0, None, R + sym, s)
    assert _close(M1, M2, 1e-6), (M1 - M2).abs().max()
    y1, _ = solve(x, w0, None, R, s)
    y2, _ = solve(x, w0, None, R + sym, s)
    assert _close(y1, y2, 1e-6)


def check_single_stage_block_diagonal(solve):
    in_dim, height = 8, 2
    R = _rand(31, 1, in_dim // height, height, height, low=-0.8, high=0.8)
    w0 = _rand(32, in_dim, in_dim)
    s = torch.ones(in_dim, 1, dtype=torch.float64)
    _, M = solve(torch.zeros(1, in_dim, dtype=torch.float64), w0, None, R, s)
    M64 = M.to(torch.float64)
    # 单级时置换是单位阵，M 必须是严格 2x2 分块对角
    for a in range(in_dim // height):
        for b in range(in_dim // height):
            blk = M64[a * height:(a + 1) * height, b * height:(b + 1) * height]
            if a == b:
                assert _close(blk, _cayley_np(R[0, a].numpy()), 1e-6), blk
            else:
                assert torch.equal(blk, torch.zeros(height, height, dtype=torch.float64)), blk


def check_butterfly_mixes_blocks(solve):
    in_dim, height = 4, 2
    R = _rand(41, 2, in_dim // height, height, height, low=-1.0, high=1.0)
    w0 = _rand(42, in_dim, in_dim)
    s = torch.ones(in_dim, 1, dtype=torch.float64)
    x = _rand(43, 2, in_dim)
    _, M2 = solve(x, w0, None, R, s)
    _, M1 = solve(x, w0, None, R[:1], s)
    M2 = M2.to(torch.float64)
    # 两级时第 1 级的段长 4、组大小 1，重排 [0,2,1,3] 会把块间耦合起来：不再是 2x2 分块对角
    off = M2[0:2, 2:4]
    assert float(off.abs().max()) > 1e-3, off
    assert _close(M1.to(torch.float64)[0:2, 2:4], torch.zeros(2, 2, dtype=torch.float64), 1e-9)
    ref_y, ref_M = _oracle(x.numpy(), w0.numpy(), None, R.numpy(), s.numpy())
    assert _close(M2, ref_M, 1e-6)


def check_stage_product_order(solve):
    in_dim, height = 4, 2
    R = _rand(51, 2, in_dim // height, height, height, low=-1.0, high=1.0)
    x = _rand(52, in_dim)
    w0 = _rand(53, in_dim, in_dim)
    s = torch.ones(in_dim, 1, dtype=torch.float64)
    _, M = solve(x, w0, None, R, s)
    M = M.to(torch.float64)
    # 手工分别构造 M_0 与 M_1，验证连乘顺序是 M_1 @ M_0（反过来一般不相等）
    m0, m1 = _stage_matrices(R, height, in_dim)
    assert _close(M, m1 @ m0, 1e-6), M
    assert not _close(M, m0 @ m1, 1e-3), 'product order is not commutative here'


def _stage_matrices(R, height, dim):
    stages, blocks, h, _ = R.shape
    out = []
    for i in range(stages):
        B = np.zeros((dim, dim))
        for j in range(blocks):
            q = _cayley_np(R[i, j].numpy())
            B[j * h:(j + 1) * h, j * h:(j + 1) * h] = q
        perm = _perm_np(dim, height, i)
        P = np.zeros((dim, dim))
        for k, pk in enumerate(perm):
            P[k, pk] = 1.0
        out.append(torch.tensor(P @ B @ P.T, dtype=torch.float64))
    return out


def check_scale_applied_after_rotation(solve):
    in_dim, out_dim, height = 4, 3, 2
    R = _rand(61, 2, in_dim // height, height, height, low=-1.0, high=1.0)
    w0 = _rand(62, out_dim, in_dim)
    s = torch.tensor([[2.0], [0.5], [-1.0]], dtype=torch.float64)
    x = _rand(63, 2, in_dim)
    y, M = solve(x, w0, None, R, s)
    M64 = M.to(torch.float64)
    right = (M64 @ w0.to(torch.float64).T).T * s
    assert _close(y, x.to(torch.float64) @ right.T, 1e-6), y
    # s 是逐输出行：只放大某一行时，仅 y 的对应那一列变化，其余列逐位不变
    ones = torch.ones(out_dim, 1, dtype=torch.float64)
    y_base, _ = solve(x, w0, None, R, ones)
    s_row = ones.clone()
    s_row[1] = 3.0
    y_row, _ = solve(x, w0, None, R, s_row)
    y_base = y_base.to(torch.float64)
    y_row = y_row.to(torch.float64)
    assert _close(y_row[:, 1], y_base[:, 1] * 3.0, 1e-6)
    assert _close(y_row[:, [0, 2]], y_base[:, [0, 2]], 1e-6)
    # 缩放不能作用在输入维上：按输入列缩放得到的结果与参考答案不同
    wrong = (M64 @ (w0.to(torch.float64) * torch.tensor([[1.0], [2.0], [0.5], [3.0]]).T).T).T
    assert not _close(y, x.to(torch.float64) @ wrong.T, 1e-3)


def check_rotation_on_input_side(solve):
    dim, height = 4, 2
    R = _rand(65, 2, dim // height, height, height, low=-1.0, high=1.0)
    w0 = _rand(66, dim, dim)
    s = torch.ones(dim, 1, dtype=torch.float64)
    x = _rand(67, 2, dim)
    y, M = solve(x, w0, None, R, s)
    M64 = M.to(torch.float64)
    w64 = w0.to(torch.float64)
    # 旋转作用在权重矩阵的输入侧：W' = (M @ W0.T).T = W0 @ M.T
    assert _close(y, x.to(torch.float64) @ M64 @ w64.T, 1e-6), y
    # 把旋转放到输出侧（M @ W0）是另一种口径，结果不等
    wrong = M64 @ w64
    assert not _close(y, x.to(torch.float64) @ wrong.T, 1e-3)


def check_scale_orientation_shapes(solve):
    in_dim, out_dim, height = 4, 3, 2
    R = torch.zeros(1, in_dim // height, height, height, dtype=torch.float64)
    w0 = _rand(71, out_dim, in_dim)
    x = _rand(72, 2, in_dim)
    s = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y, M = solve(x, w0, None, R, s)
    assert _close(M, torch.eye(in_dim, dtype=torch.float64), 1e-9)
    # R=0 时 y = (x @ W0.T) * s，逐输出行缩放 => 展平后等价于按 y 的最后一维缩放
    base = x.to(torch.float64) @ w0.to(torch.float64).T
    assert _close(y, base * s.reshape(1, -1), 1e-6), y
    assert abs(float(y.to(torch.float64)[0, 2]) / float(base[0, 2]) - 3.0) < 1e-6
    assert abs(float(y.to(torch.float64)[1, 0]) / float(base[1, 0]) - 1.0) < 1e-6


def check_block_size_four(solve):
    in_dim, height = 8, 4
    R = _rand(81, 2, in_dim // height, height, height, low=-0.6, high=0.6)
    w0 = _rand(82, 5, in_dim)
    s = _rand(83, 5, 1, low=0.5, high=1.5)
    x = _rand(84, 2, in_dim)
    y, M = solve(x, w0, None, R, s)
    ref_y, ref_M = _oracle(x.numpy(), w0.numpy(), None, R.numpy(), s.numpy())
    assert _close(M, ref_M, 1e-6), M
    assert _close(y, ref_y, 1e-6), y
    # 第 1 级段长 8、组大小 2 => 组顺序 [0,2,1,3]，置换向量应是 [0,1,4,5,2,3,6,7]
    assert _perm_np(in_dim, height, 1) == [0, 1, 4, 5, 2, 3, 6, 7]
    assert _perm_np(in_dim, height, 0) == list(range(in_dim))
    M64 = M.to(torch.float64)
    assert _close(M64 @ M64.T, torch.eye(in_dim, dtype=torch.float64), 1e-6)


def check_norm_preserving(solve):
    in_dim, height = 8, 2
    R = _rand(91, 3, in_dim // height, height, height, low=-1.0, high=1.0)
    eye_w = torch.eye(in_dim, dtype=torch.float64)
    s = torch.ones(in_dim, 1, dtype=torch.float64)
    x = _rand(92, 4, in_dim)
    y, M = solve(x, eye_w, None, R, s)
    # 正交旋转 + 单位缩放 + 单位权重保范数；W' = (M @ I).T = M.T，故 y = x @ M
    assert _close(y, x.to(torch.float64) @ M.to(torch.float64), 1e-6)
    assert _close(y.norm(dim=-1), x.norm(dim=-1), 1e-5), (y.norm(dim=-1), x.norm(dim=-1))


def check_batch_and_broadcast(solve):
    in_dim, out_dim, height = 8, 4, 2
    R = _rand(101, 2, in_dim // height, height, height, low=-0.9, high=0.9)
    w0 = _rand(102, out_dim, in_dim)
    b0 = _rand(103, out_dim)
    s = _rand(104, out_dim, 1, low=0.5, high=1.5)
    x = _rand(105, 2, 3, in_dim)
    y, M = solve(x, w0, b0, R, s)
    assert y.shape == (2, 3, out_dim) and M.shape == (in_dim, in_dim)
    ref_y, ref_M = _oracle(x.numpy(), w0.numpy(), b0.numpy(), R.numpy(), s.numpy())
    assert _close(y, ref_y, 1e-6) and _close(M, ref_M, 1e-6)


def check_against_oracle(solve):
    cases = [
        (2, 2, 2, 1, False),
        (4, 4, 2, 2, True),
        (8, 5, 2, 3, False),
        (8, 8, 4, 2, True),
        (16, 6, 2, 2, False),
        (16, 9, 4, 2, True),
        (6, 3, 2, 1, False),
        (12, 4, 2, 2, True),
    ]
    for seed, (in_dim, out_dim, height, stages, use_bias) in enumerate(cases):
        blocks = in_dim // height
        assert blocks % (2 ** (stages - 1)) == 0, (in_dim, height, stages)
        R = _rand(110 + seed * 5, stages, blocks, height, height, low=-0.9, high=0.9)
        w0 = _rand(111 + seed * 5, out_dim, in_dim)
        s = _rand(112 + seed * 5, out_dim, 1, low=0.5, high=1.5)
        b0 = _rand(113 + seed * 5, out_dim) if use_bias else None
        x = _rand(114 + seed * 5, 2, 3, in_dim)
        y, M = solve(x, w0, b0, R, s)
        ref_y, ref_M = _oracle(x.numpy(), w0.numpy(), None if b0 is None else b0.numpy(), R.numpy(), s.numpy())
        assert _close(M, ref_M, 1e-6), (in_dim, out_dim, height, stages)
        assert _close(y, ref_y, 1e-6), (in_dim, out_dim, height, stages)
    # float32 输入可跑，输出仍是 float32
    R = _rand(160, 2, 4, 2, 2, low=-0.9, high=0.9)
    w0 = _rand(161, 3, 8)
    s = _rand(162, 3, 1, low=0.5, high=1.5)
    x = _rand(163, 2, 8)
    y32, M32 = solve(x.to(torch.float32), w0.to(torch.float32), None, R.to(torch.float32), s.to(torch.float32))
    assert y32.dtype == torch.float32 and M32.dtype == torch.float32
    ref_y, ref_M = _oracle(x.numpy(), w0.numpy(), None, R.numpy(), s.numpy())
    assert _close(M32, ref_M, 1e-5) and _close(y32, ref_y, 1e-5)


def check_invalid_inputs(solve):
    in_dim, out_dim, height = 8, 5, 2
    R = torch.zeros(2, in_dim // height, height, height, dtype=torch.float64)
    w0 = torch.zeros(out_dim, in_dim, dtype=torch.float64)
    x = torch.zeros(2, in_dim, dtype=torch.float64)
    s = torch.ones(out_dim, 1, dtype=torch.float64)
    bad_calls = [
        ("x-not-tensor", lambda: solve(x.tolist(), w0, None, R, s)),
        ("x-int", lambda: solve(x.to(torch.int64), w0, None, R, s)),
        ("x-bool", lambda: solve(x.to(torch.bool), w0, None, R, s)),
        ("x-nan", lambda: solve(torch.full((2, in_dim), float("nan"), dtype=torch.float64), w0, None, R, s)),
        ("x-wrong-last-dim", lambda: solve(torch.zeros(2, 7, dtype=torch.float64), w0, None, R, s)),
        ("w0-not-tensor", lambda: solve(x, w0.tolist(), None, R, s)),
        ("w0-1d", lambda: solve(x, torch.zeros(40, dtype=torch.float64), None, R, s)),
        ("w0-3d", lambda: solve(x, torch.zeros(1, out_dim, in_dim, dtype=torch.float64), None, R, s)),
        ("w0-int", lambda: solve(x, w0.to(torch.int32), None, R, s)),
        ("w0-inf", lambda: solve(x, torch.full((out_dim, in_dim), float("inf"), dtype=torch.float64), None, R, s)),
        ("R-not-tensor", lambda: solve(x, w0, None, R.tolist(), s)),
        ("R-3d", lambda: solve(x, w0, None, R.reshape(2, -1), s)),
        ("R-5d", lambda: solve(x, w0, None, R.unsqueeze(0), s)),
        ("R-int", lambda: solve(x, w0, None, R.to(torch.int64), s)),
        ("R-nan", lambda: solve(x, w0, None, torch.full_like(R, float("nan")), s)),
        ("R-nonsquare-block", lambda: solve(x, w0, None, torch.zeros(2, in_dim // height, 2, 3, dtype=torch.float64), s)),
        ("R-odd-block", lambda: solve(torch.zeros(2, 9, dtype=torch.float64), torch.zeros(out_dim, 9, dtype=torch.float64),
                                      None, torch.zeros(1, 3, 3, 3, dtype=torch.float64), s)),
        ("R-block-size-zero", lambda: solve(x, w0, None, torch.zeros(2, 4, 0, 0, dtype=torch.float64), s)),
        ("R-block-count-mismatch", lambda: solve(x, w0, None, torch.zeros(2, 5, 2, 2, dtype=torch.float64), s)),
        ("R-in-not-divisible", lambda: solve(torch.zeros(2, 7, dtype=torch.float64), torch.zeros(out_dim, 7, dtype=torch.float64),
                                            None, torch.zeros(1, 4, 2, 2, dtype=torch.float64), s)),
        ("R-stage-not-power-divisible", lambda: solve(torch.zeros(2, 12, dtype=torch.float64),
                                                     torch.zeros(out_dim, 12, dtype=torch.float64), None,
                                                     torch.zeros(3, 6, 2, 2, dtype=torch.float64), s)),
        ("s-not-tensor", lambda: solve(x, w0, None, R, s.tolist())),
        ("s-1d", lambda: solve(x, w0, None, R, s.reshape(-1))),
        ("s-col-per-input", lambda: solve(x, w0, None, R, torch.ones(in_dim, 1, dtype=torch.float64))),
        ("s-per-input-row", lambda: solve(x, w0, None, R, torch.ones(1, out_dim, dtype=torch.float64))),
        ("s-int", lambda: solve(x, w0, None, R, s.to(torch.int64))),
        ("s-nan", lambda: solve(x, w0, None, R, torch.full((out_dim, 1), float("nan"), dtype=torch.float64))),
        ("b0-not-tensor", lambda: solve(x, w0, [0.0] * out_dim, R, s)),
        ("b0-wrong-length", lambda: solve(x, w0, torch.zeros(3, dtype=torch.float64), R, s)),
        ("b0-2d", lambda: solve(x, w0, torch.zeros(out_dim, 1, dtype=torch.float64), R, s)),
        ("b0-nan", lambda: solve(x, w0, torch.full((out_dim,), float("nan"), dtype=torch.float64), R, s)),
        ("b0-int", lambda: solve(x, w0, torch.zeros(out_dim, dtype=torch.int64), R, s)),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
