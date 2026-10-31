import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _oracle(x, w0, kernel, b0=None):
    """numpy 逐块逐元素重算；块内 dyadic 下标的循环取值手写，不共享向量化代码。"""
    x = np.asarray(x, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    out_dim, in_dim = w0.shape
    bs = kernel.shape[-1]
    delta = np.zeros((out_dim, in_dim), dtype=np.float64)
    for m in range(out_dim // bs):
        for n in range(in_dim // bs):
            for i in range(bs):
                for j in range(bs):
                    src = j - i
                    while src < 0:
                        src += bs
                    delta[m * bs + i, n * bs + j] = kernel[m, n, src] / in_dim
    flat = x.reshape(-1, in_dim)
    y = flat @ w0.T
    if b0 is not None:
        y = y + np.asarray(b0, dtype=np.float64)
    y = y + flat @ delta.T
    return y.reshape(x.shape[:-1] + (out_dim,)), delta


def _rand(seed, *shape, low=-1.0, high=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * (high - low) + low


def check_example(solve):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    w0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    kernel = torch.tensor([[[1.0, 2.0], [-1.0, 0.0]]], dtype=torch.float64)
    y, delta = solve(x, w0, kernel)
    expect_delta = torch.tensor(
        [[0.25, 0.5, -0.25, 0.0], [0.5, 0.25, 0.0, -0.25]], dtype=torch.float64
    )
    expect_y = torch.tensor([[1.5, 2.0]], dtype=torch.float64)
    assert delta.shape == (2, 4) and y.shape == (1, 2)
    assert y.dtype == torch.float32 and delta.dtype == torch.float32
    assert _close(delta, expect_delta, 1e-6), delta
    assert _close(y, expect_y, 1e-6), y


check_example.example = True


def check_block_layout(solve):
    in_dim = out_dim = 6
    bs = 3
    kernel = torch.zeros(out_dim // bs, in_dim // bs, bs, dtype=torch.float64)
    kernel[0, 1] = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float64)
    w0 = torch.zeros(out_dim, in_dim, dtype=torch.float64)
    _, delta = solve(torch.zeros(1, in_dim, dtype=torch.float64), w0, kernel)
    delta = delta.to(torch.float64)
    block = delta[0:3, 3:6]
    # 每一行都是 [0.5,-1.5,2.0]/6 的循环左移
    expect = torch.tensor(
        [[0.5, -1.5, 2.0], [2.0, 0.5, -1.5], [-1.5, 2.0, 0.5]], dtype=torch.float64
    ) / 6
    assert _close(block, expect, 1e-6), block
    # 块内只有 bs 个不同取值，且块 (0,0)/(1,*)/(*,0) 全零
    assert len(set(round(float(v), 9) for v in block.reshape(-1))) == 3
    assert torch.equal(delta[:, 0:3], torch.zeros(6, 3, dtype=torch.float32))
    assert torch.equal(delta[3:6, :], torch.zeros(3, 6, dtype=torch.float32))


def check_orientation(solve):
    in_dim = out_dim = 3
    kernel = torch.tensor([[[1.0, 2.0, 4.0]]], dtype=torch.float64)
    w0 = torch.zeros(3, 3, dtype=torch.float64)
    _, delta = solve(torch.zeros(1, 3, dtype=torch.float64), w0, kernel)
    delta = delta.to(torch.float64)
    expect = torch.tensor(
        [[1.0, 2.0, 4.0], [4.0, 1.0, 2.0], [2.0, 4.0, 1.0]], dtype=torch.float64
    ) / 3
    assert _close(delta, expect, 1e-6), delta
    # 反方向 (i-j) 会得到 [[1,4,2],[2,1,4],[4,2,1]]/3，两者不可互换
    wrong = torch.tensor(
        [[1.0, 4.0, 2.0], [2.0, 1.0, 4.0], [4.0, 2.0, 1.0]], dtype=torch.float64
    ) / 3
    assert not _close(delta, wrong, 1e-4)


def check_scaling_by_in_features(solve):
    small_kernel = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]], dtype=torch.float64)
    # 同一组块值，d_in 从 4 翻倍到 8：block size 不变，因此 kernel 的 n 块数翻倍，只填前两块
    large_kernel = torch.tensor([[[1.0, -2.0], [0.5, 3.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    small = torch.zeros(2, 4, dtype=torch.float64)
    large = torch.zeros(2, 8, dtype=torch.float64)
    _, d_small = solve(torch.zeros(1, 4, dtype=torch.float64), small, small_kernel)
    _, d_large = solve(torch.zeros(1, 8, dtype=torch.float64), large, large_kernel)
    # 归一化分母是 d_in，不是 block size：d_in 翻倍，同 kernel 的块数值减半
    assert _close(d_large[:2, :4].to(torch.float64), d_small[:2, :4].to(torch.float64) / 2, 1e-6)
    # block size 仍是 2：每个 2 列的块内部各自循环，行 1 等于行 0 在该块内循环移位
    row0 = d_large[0, 2:4].to(torch.float64)
    assert _close(d_large[1, 2:4].to(torch.float64), torch.roll(row0, 1), 1e-6)
    assert abs(float(d_large.to(torch.float64).abs().max()) - 3.0 / 8) < 1e-9


def check_single_block_circulant(solve):
    in_dim = out_dim = 4
    kernel = torch.tensor([[[1.0, 2.0, -1.0, 0.5]]], dtype=torch.float64)
    w0 = _rand(11, out_dim, in_dim)
    x = _rand(12, 2, in_dim)
    y, delta = solve(x, w0, kernel)
    ref_y, ref_delta = _oracle(x.numpy(), w0.numpy(), kernel.numpy())
    assert _close(delta, ref_delta, 1e-6)
    assert _close(y, ref_y, 1e-6)
    # 整块循环：第 o 行等于 kernel 向右循环移位 o 位（等价于 kernel[(j-o) % n]）
    d = delta.to(torch.float64)
    for o in range(out_dim):
        assert _close(d[o], torch.roll(kernel.reshape(-1).to(torch.float64), o) / in_dim, 1e-6)


def check_block_size_one(solve):
    in_dim, out_dim = 4, 3
    kernel = _rand(21, out_dim, in_dim, 1)
    w0 = _rand(22, out_dim, in_dim)
    x = _rand(23, 5, in_dim)
    y, delta = solve(x, w0, kernel)
    d = delta.to(torch.float64)
    # bs=1 时每个块只有 1 个元素，(j-i) % 1 恒为 0：delta 退化成整份 kernel 除以 d_in
    assert d.shape == (out_dim, in_dim)
    assert _close(d, kernel.squeeze(-1).to(torch.float64) / in_dim, 1e-6)
    expect = x.to(torch.float64) @ (w0.to(torch.float64) + d).T
    assert _close(y, expect, 1e-6)


def check_constant_kernel(solve):
    in_dim, out_dim, bs = 8, 4, 4
    kernel = torch.full((out_dim // bs, in_dim // bs, bs), 0.75, dtype=torch.float64)
    _, delta = solve(torch.zeros(1, in_dim, dtype=torch.float64), torch.zeros(out_dim, in_dim, dtype=torch.float64), kernel)
    d = delta.to(torch.float64)
    # 常数核的循环矩阵是常数阵：delta 全等 0.75/d_in
    assert _close(d, torch.full((out_dim, in_dim), 0.75 / in_dim, dtype=torch.float64), 1e-6)
    # 每行之和等于 bs 个非零元之和：0.75 * bs / d_in * (in_dim/bs) = 0.75
    assert _close(d.sum(dim=1), torch.full((out_dim,), 0.75, dtype=torch.float64), 1e-6)
    assert _close(d.sum(), torch.tensor(out_dim * 0.75, dtype=torch.float64), 1e-6)


def check_merge_equivalence(solve):
    in_dim, out_dim, bs = 6, 6, 3
    w0 = _rand(31, out_dim, in_dim)
    kernel = _rand(32, out_dim // bs, in_dim // bs, bs)
    b0 = _rand(33, out_dim)
    x = _rand(34, 3, in_dim)
    y, delta = solve(x, w0, kernel, b0)
    d = delta.to(torch.float64)
    merged = x.to(torch.float64) @ (w0.to(torch.float64) + d).T + b0.to(torch.float64)
    assert _close(y, merged, 1e-6)
    # delta 与 x 无关：换一份 x，delta 逐位不变
    _, delta2 = solve(x * 5.0, w0, kernel, b0)
    assert torch.equal(delta, delta2)


def check_block_locality(solve):
    in_dim = out_dim = 6
    bs = 3
    kernel = _rand(41, out_dim // bs, in_dim // bs, bs)
    w0 = torch.zeros(out_dim, in_dim, dtype=torch.float64)
    _, base = solve(torch.zeros(1, in_dim, dtype=torch.float64), w0, kernel)
    perturbed = kernel.clone()
    perturbed[1, 0] = perturbed[1, 0] + 3.0
    _, changed = solve(torch.zeros(1, in_dim, dtype=torch.float64), w0, perturbed)
    base = base.to(torch.float64)
    changed = changed.to(torch.float64)
    # 只改 kernel[1,0]，只有 delta 的 (1,0) 块变化，其余块逐位不变
    diff = (changed - base).abs()
    mask = torch.ones(out_dim, in_dim, dtype=torch.bool)
    mask[3:6, 0:3] = False
    assert torch.equal(diff[mask], torch.zeros(int(mask.sum()), dtype=torch.float64))
    assert float(diff[3:6, 0:3].abs().max()) > 1e-3


def check_batch_and_broadcast(solve):
    in_dim, out_dim, bs = 6, 9, 3
    w0 = _rand(51, out_dim, in_dim)
    kernel = _rand(52, out_dim // bs, in_dim // bs, bs)
    b0 = _rand(53, out_dim)
    x = _rand(54, 2, 4, in_dim)
    y, delta = solve(x, w0, kernel, b0)
    assert y.shape == (2, 4, out_dim)
    ref_y, ref_delta = _oracle(x.numpy(), w0.numpy(), kernel.numpy(), b0.numpy())
    assert _close(delta, ref_delta, 1e-6)
    assert _close(y, ref_y, 1e-6)


def check_against_oracle(solve):
    cases = [
        (1, 1, 1, False),
        (4, 6, 2, True),
        (6, 6, 3, False),
        (12, 8, 4, True),
        (8, 12, 1, False),
    ]
    for seed, (in_dim, out_dim, bs, use_bias) in enumerate(cases):
        w0 = _rand(60 + seed, out_dim, in_dim)
        kernel = _rand(70 + seed, out_dim // bs, in_dim // bs, bs)
        b0 = _rand(80 + seed, out_dim) if use_bias else None
        x = _rand(90 + seed, 2, 3, in_dim)
        y, delta = solve(x, w0, kernel, b0)
        ref_y, ref_delta = _oracle(
            x.numpy(), w0.numpy(), kernel.numpy(), None if b0 is None else b0.numpy()
        )
        assert _close(delta, ref_delta, 1e-6), (in_dim, out_dim, bs, delta, ref_delta)
        assert _close(y, ref_y, 1e-6), (in_dim, out_dim, bs, y, ref_y)
    # float32 输入同样可跑，输出仍是 float32
    w0 = _rand(101, 4, 4)
    kernel = _rand(102, 2, 2, 2)
    x = _rand(103, 2, 4)
    y32, d32 = solve(x.to(torch.float32), w0.to(torch.float32), kernel.to(torch.float32))
    assert y32.dtype == torch.float32 and d32.dtype == torch.float32
    ref_y, ref_delta = _oracle(x.numpy(), w0.numpy(), kernel.numpy())
    assert _close(d32, ref_delta, 1e-5)
    assert _close(y32, ref_y, 1e-5)


def check_invalid_inputs(solve):
    x = torch.zeros(2, 6, dtype=torch.float64)
    w0 = torch.zeros(4, 6, dtype=torch.float64)
    kernel = torch.zeros(2, 3, 2, dtype=torch.float64)
    b0 = torch.zeros(4, dtype=torch.float64)
    bad_calls = [
        ("x-not-tensor", lambda: solve([[0.0] * 6], w0, kernel)),
        ("x-int", lambda: solve(x.to(torch.int64), w0, kernel)),
        ("x-bool", lambda: solve(x.to(torch.bool), w0, kernel)),
        ("x-nan", lambda: solve(torch.full((2, 6), float("nan"), dtype=torch.float64), w0, kernel)),
        ("x-wrong-last-dim", lambda: solve(torch.zeros(2, 5, dtype=torch.float64), w0, kernel)),
        ("w0-not-tensor", lambda: solve(x, w0.tolist(), kernel)),
        ("w0-1d", lambda: solve(x, torch.zeros(24, dtype=torch.float64), kernel)),
        ("w0-3d", lambda: solve(x, torch.zeros(1, 4, 6, dtype=torch.float64), kernel)),
        ("w0-int", lambda: solve(x, w0.to(torch.int32), kernel)),
        ("w0-inf", lambda: solve(x, torch.full((4, 6), float("inf"), dtype=torch.float64), kernel)),
        ("kernel-not-tensor", lambda: solve(x, w0, kernel.tolist())),
        ("kernel-2d", lambda: solve(x, w0, torch.zeros(2, 6, dtype=torch.float64))),
        ("kernel-int", lambda: solve(x, w0, kernel.to(torch.int64))),
        ("kernel-nan", lambda: solve(x, w0, torch.full((2, 3, 2), float("nan"), dtype=torch.float64))),
        ("kernel-empty-block", lambda: solve(x, w0, torch.zeros(2, 3, 0, dtype=torch.float64))),
        ("in-not-divisible", lambda: solve(torch.zeros(2, 5, dtype=torch.float64), torch.zeros(4, 5, dtype=torch.float64), kernel)),
        ("out-not-divisible", lambda: solve(x, torch.zeros(5, 6, dtype=torch.float64), kernel)),
        ("kernel-first-dim", lambda: solve(x, w0, torch.zeros(1, 3, 2, dtype=torch.float64))),
        ("kernel-second-dim", lambda: solve(x, w0, torch.zeros(2, 2, 2, dtype=torch.float64))),
        ("b0-wrong-length", lambda: solve(x, w0, kernel, torch.zeros(3, dtype=torch.float64))),
        ("b0-2d", lambda: solve(x, w0, kernel, torch.zeros(4, 1, dtype=torch.float64))),
        ("b0-nan", lambda: solve(x, w0, kernel, torch.full((4,), float("nan"), dtype=torch.float64))),
        ("b0-not-tensor", lambda: solve(x, w0, kernel, [0.0] * 4)),
        ("b0-int", lambda: solve(x, w0, kernel, torch.zeros(4, dtype=torch.int64))),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
