import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _select_np(logits, bank, topk):
    """numpy 版选择：值降序（同值按下标升序）取 topk，只对选中项做 softmax，再加权库向量。"""
    idx = np.argsort(-logits, axis=-1, kind="stable")[..., :topk]
    sel = np.take_along_axis(logits, idx, axis=-1)
    e = np.exp(sel - sel.max(axis=-1, keepdims=True))
    w = e / e.sum(axis=-1, keepdims=True)
    return (w[..., np.newaxis] * bank[idx]).sum(axis=-2)


def _oracle(x, w0, logits_a, logits_b, bank, b0=None, topk=2):
    x = np.asarray(x, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    logits_a = np.asarray(logits_a, dtype=np.float64)
    logits_b = np.asarray(logits_b, dtype=np.float64)
    bank = np.asarray(bank, dtype=np.float64)
    out_dim, in_dim = w0.shape
    rank = logits_a.shape[0]
    A = _select_np(logits_a, bank, topk).reshape(rank, -1)
    B = _select_np(logits_b, bank, topk).transpose(0, 2, 1).reshape(-1, rank)
    delta = B @ A
    flat = x.reshape(-1, in_dim)
    y = flat @ w0.T
    if b0 is not None:
        y = y + np.asarray(b0, dtype=np.float64)
    y = y + flat @ delta.T
    return y.reshape(x.shape[:-1] + (out_dim,)), A, B, delta


def _rand(seed, *shape, low=-1.0, high=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * (high - low) + low


def _cfg(seed, in_f, out_f, r, num_vectors, vl, topk):
    bank = _rand(seed, num_vectors, vl, low=-0.7, high=0.7)
    logits_a = _rand(seed + 1000, r, in_f // vl, num_vectors)
    logits_b = _rand(seed + 2000, out_f // vl, r, num_vectors)
    return bank, logits_a, logits_b


def check_example(solve):
    # d_in = d_out = 2、vector_length = 2 => A、B 各只有一个 tile，r = 2
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    w0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    logits_a = torch.tensor([[[0.0, 1.0, 0.5, -1.0]], [[0.0, 0.0, 3.0, 1.0]]], dtype=torch.float64)
    logits_b = torch.tensor([[[1.0, 0.0, 0.5, 2.0], [1.0, 1.0, 1.0, -2.0]]], dtype=torch.float64)
    y, A, B, delta = solve(x, w0, logits_a, logits_b, bank, None, 2)
    expect_A = torch.tensor([[0.7550823, 1.3775410], [1.6423913, 1.8211962]], dtype=torch.float64)
    expect_B = torch.tensor([[-0.4621179, 0.5], [0.3655292, 0.5]], dtype=torch.float64)
    assert A.shape == (2, 2) and B.shape == (2, 2) and delta.shape == (2, 2) and y.shape == (1, 2)
    assert all(t.dtype == torch.float32 for t in (y, A, B, delta))
    assert _close(A, expect_A, 1e-6), A
    assert _close(B, expect_B, 1e-6), B
    assert _close(delta, (expect_B @ expect_A), 1e-6), delta
    assert _close(y, x.to(torch.float64) @ (w0.to(torch.float64) + delta.to(torch.float64)).T, 1e-6), y


check_example.example = True


def check_selected_only_softmax(solve):
    # 未被选中的第三条 logit 指向一个巨大的库向量：全量 softmax 会被它带飞
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]], dtype=torch.float64)
    logits_a = torch.tensor([[[2.0, 1.0, 0.0]]], dtype=torch.float64)
    logits_b = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    w = torch.exp(torch.tensor([2.0, 1.0], dtype=torch.float64))
    w = w / w.sum()
    _, A, _, _ = solve(torch.zeros(1, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64),
                       logits_a, logits_b, bank, None, 2)
    assert A.shape == (1, 2)
    # 只对选中的两条（下标 0、1）归一化
    assert _close(A, torch.tensor([[w[0], w[1]]]), 1e-6), A
    assert float(A.to(torch.float64).abs().max()) < 1.0
    assert abs(float(A.to(torch.float64)[0, 1]) - 0.2689414) < 1e-6
    # 若把 num_vectors 条一起做 softmax，权重会分裂到 0.675/0.249/0.076，第二条落到 ~7.67
    full = torch.softmax(torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64), dim=-1)
    assert abs(float(full[0, 2]) - 0.090031) < 1e-5


def check_topk_one(solve):
    in_f, out_f, r, V, vl = 6, 3, 2, 6, 3
    bank = _rand(5, V, vl)
    logits_a = _rand(6, r, in_f // vl, V)
    logits_b = _rand(7, out_f // vl, r, V)
    _, A, B, delta = solve(torch.zeros(1, in_f, dtype=torch.float64), torch.zeros(out_f, in_f, dtype=torch.float64),
                           logits_a, logits_b, bank, None, 1)
    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)
    assert A64.shape == (r, in_f) and B64.shape == (out_f, r)
    # topk=1 时权重恒为 1，A/B 的每一段就是被 argmax 选中的那一整行库向量
    for j in range(r):
        for t in range(in_f // vl):
            pick = int(torch.argmax(logits_a[j, t]).item())
            assert _close(A64[j, t * vl:(t + 1) * vl], bank[pick], 1e-6)
    for j in range(r):
        pick = int(torch.argmax(logits_b[0, j]).item())
        assert _close(B64[:, j], bank[pick], 1e-6)
    assert _close(delta.to(torch.float64), B64 @ A64, 1e-6)


def check_topk_full(solve):
    bank = _rand(8, 5, 2)
    logits_a = _rand(9, 2, 3, 5)
    logits_b = _rand(10, 2, 2, 5)
    w0 = torch.zeros(4, 6, dtype=torch.float64)
    _, A, B, delta = solve(torch.zeros(1, 6, dtype=torch.float64), w0, logits_a, logits_b, bank, None, 5)
    # topk == num_vectors 时「只选 topk」退化成全量 softmax，可与显式全量 softmax 对齐
    full_a = (torch.softmax(logits_a, dim=-1).unsqueeze(-1) * bank).sum(-2).reshape(2, -1)
    full_b = (torch.softmax(logits_b, dim=-1).unsqueeze(-1) * bank).sum(-2).transpose(1, 2).reshape(-1, 2)
    assert _close(A, full_a, 1e-6), A
    assert _close(B, full_b, 1e-6), B
    assert _close(delta, full_b @ full_a, 1e-6), delta


def check_tie_break(solve):
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float64)
    # logits_a[0,0] 的最大值 1.0 并列于下标 0/1/2，topk=2 必须取 0 和 1（下标升序）
    logits_a = torch.tensor([[[1.0, 1.0, 1.0, -5.0]], [[2.0, 2.0, -1.0, -5.0]]], dtype=torch.float64)
    logits_b = torch.tensor([[[0.5, 0.5, 9.0, -5.0], [1.0, 1.0, 1.0, -5.0]]], dtype=torch.float64)
    _, A, B, _ = solve(torch.zeros(1, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64),
                       logits_a, logits_b, bank, None, 2)
    # 并列取下标 0、1 => 权重 (0.5, 0.5)，得到 [0.5, 0.5]；若取 1、2 会得到 [1.5, 2.0]
    assert _close(A[0], torch.tensor([0.5, 0.5], dtype=torch.float64), 1e-6), A
    # 第二行同样并列于 0、1，取前两个
    assert _close(A[1], torch.tensor([0.5, 0.5], dtype=torch.float64), 1e-6), A
    # B 的第 0 列：9.0 唯一最大 + 并列的 0.5 里取下标 0；第 1 列又是 0、1 并列
    e = torch.exp(torch.tensor([9.0 - 0.5], dtype=torch.float64))
    w_big = (e / (1.0 + e)).item()
    assert _close(B[:, 0], torch.tensor([w_big * 3.0 + (1 - w_big) * 1.0,
                                         w_big * 3.0 + (1 - w_big) * 0.0], dtype=torch.float64), 1e-6), B
    assert _close(B[:, 1], torch.tensor([0.5, 0.5], dtype=torch.float64), 1e-6), B


def check_a_tile_major(solve):
    in_f, out_f, r, V, vl = 6, 9, 2, 5, 3
    bank, logits_a, logits_b = _cfg(20, in_f, out_f, r, V, vl, 1)
    _, A, _, _ = solve(torch.zeros(1, in_f, dtype=torch.float64), torch.zeros(out_f, in_f, dtype=torch.float64),
                       logits_a, logits_b, bank, None, 1)
    A64 = A.to(torch.float64)
    assert A64.shape == (r, in_f)
    # A[:, 0:3] 只由 tile 0 的 logit 决定，A[:, 3:6] 只由 tile 1 决定：展平是 tile 主序，不是交错
    for j in range(r):
        for t in range(in_f // vl):
            pick = int(torch.argmax(logits_a[j, t]).item())
            assert _close(A64[j, t * vl:(t + 1) * vl], bank[pick], 1e-6)
    assert not _close(A64[:, 0:3], A64[:, 3:6], 1e-3)
    # 交换两个 tile 的 logit，A 的两段也整段交换
    swapped = logits_a.clone()
    swapped[:, 0], swapped[:, 1] = swapped[:, 1].clone(), swapped[:, 0].clone()
    _, A2, _, _ = solve(torch.zeros(1, in_f, dtype=torch.float64), torch.zeros(out_f, in_f, dtype=torch.float64),
                        swapped, logits_b, bank, None, 1)
    A2 = A2.to(torch.float64)
    assert _close(A2[:, 0:3], A64[:, 3:6], 1e-6) and _close(A2[:, 3:6], A64[:, 0:3], 1e-6)


def check_b_transpose_layout(solve):
    in_f, out_f, r, V, vl = 4, 6, 3, 5, 2
    bank, logits_a, logits_b = _cfg(30, in_f, out_f, r, V, vl, 1)
    _, _, B, _ = solve(torch.zeros(1, in_f, dtype=torch.float64), torch.zeros(out_f, in_f, dtype=torch.float64),
                       logits_a, logits_b, bank, None, 1)
    B64 = B.to(torch.float64)
    assert B64.shape == (out_f, r)
    # B[o*vl + s, j]：行按「输出 tile 的 vector_length 分量」排，列是 logits_B 的第 1 维 r
    for o in range(out_f // vl):
        for j in range(r):
            pick = int(torch.argmax(logits_b[o, j]).item())
            assert _close(B64[o * vl:(o + 1) * vl, j], bank[pick], 1e-6)
    # 未经 transpose(1, 2) 的错误展平会把 r 混进行号，两者不可能同时成立
    wrong = _select_torch(logits_b, bank, 1).reshape(-1, r)
    assert not _close(B64, wrong.to(torch.float64), 1e-3)


def _select_torch(logits, bank, topk):
    top_logits, indices = logits.to(torch.float64).topk(topk, dim=-1)
    weights = torch.softmax(top_logits.to(torch.float64), dim=-1)
    return (weights.unsqueeze(-1) * bank.to(torch.float64)[indices]).sum(-2)


def check_delta_no_extra_scaling(solve):
    in_f, out_f, r, V, vl, topk = 8, 12, 4, 7, 4, 3
    bank, logits_a, logits_b = _cfg(40, in_f, out_f, r, V, vl, topk)
    w0 = _rand(41, out_f, in_f)
    x = _rand(42, 3, in_f)
    _, A, B, delta = solve(x, w0, logits_a, logits_b, bank, None, topk)
    d64 = delta.to(torch.float64)
    # 没有 alpha/r、也没有除以 r 或 d_in：delta 就是 B @ A
    assert _close(d64, B.to(torch.float64) @ A.to(torch.float64), 1e-6)
    scaled = d64 * r
    assert not _close(d64, scaled, 1e-3)
    # 秩至多为 r：out 方向的列空间由 B 张成
    assert torch.linalg.matrix_rank(d64.float()) <= r


def check_bank_is_shared(solve):
    in_f, out_f, r, V, vl, topk = 4, 4, 2, 6, 2, 2
    bank, logits_a, logits_b = _cfg(50, in_f, out_f, r, V, vl, topk)
    w0 = _rand(51, out_f, in_f)
    x = _rand(52, 2, in_f)
    _, A0, B0, d0 = solve(x, w0, logits_a, logits_b, bank, None, topk)
    perturbed = bank.clone()
    perturbed[3] = perturbed[3] + 10.0
    _, A1, B1, d1 = solve(x, w0, logits_a, logits_b, perturbed, None, topk)
    # 库向量 3 被 logits_A 用到时才会改动 A；同一张库也被 B 复用
    assert float((A1.to(torch.float64) - A0.to(torch.float64)).abs().max()) > 1.0
    assert float((B1.to(torch.float64) - B0.to(torch.float64)).abs().max()) > 0.0
    assert _close(d1.to(torch.float64), B1.to(torch.float64) @ A1.to(torch.float64), 1e-6)
    # A、B 的每个 tile 权重都归一：库行全 1 时选中值恒为 1
    ones_bank = torch.ones(V, vl, dtype=torch.float64)
    _, Ao, Bo, _ = solve(x, w0, logits_a, logits_b, ones_bank, None, topk)
    assert _close(Ao, torch.full_like(Ao, 1.0), 1e-6)
    assert _close(Bo, torch.full_like(Bo, 1.0), 1e-6)


def check_merge_equivalence(solve):
    in_f, out_f, r, V, vl, topk = 6, 6, 2, 5, 3, 2
    bank, logits_a, logits_b = _cfg(60, in_f, out_f, r, V, vl, topk)
    w0 = _rand(61, out_f, in_f)
    b0 = _rand(62, out_f)
    x = _rand(63, 3, in_f)
    y, A, B, delta = solve(x, w0, logits_a, logits_b, bank, b0, topk)
    d64 = delta.to(torch.float64)
    merged = x.to(torch.float64) @ (w0.to(torch.float64) + d64).T + b0.to(torch.float64)
    assert _close(y, merged, 1e-6), y
    # 未合并路径：先乘 A 再乘 B，与直接乘 delta 等价
    two_step = torch.nn.functional.linear(torch.nn.functional.linear(x.to(torch.float64), A.to(torch.float64)),
                                          B.to(torch.float64))
    assert _close(y.to(torch.float64) - x.to(torch.float64) @ w0.to(torch.float64).T - b0.to(torch.float64),
                  two_step, 1e-6)
    _, _, _, d2 = solve(x * 7.0, w0, logits_a, logits_b, bank, b0, topk)
    assert torch.equal(delta, d2)


def check_batch_and_broadcast(solve):
    in_f, out_f, r, V, vl, topk = 6, 9, 2, 4, 3, 2
    bank, logits_a, logits_b = _cfg(70, in_f, out_f, r, V, vl, topk)
    w0 = _rand(71, out_f, in_f)
    b0 = _rand(72, out_f)
    x = _rand(73, 2, 4, in_f)
    y, A, B, delta = solve(x, w0, logits_a, logits_b, bank, b0, topk)
    assert y.shape == (2, 4, out_f)
    assert A.shape == (r, in_f) and B.shape == (out_f, r) and delta.shape == (out_f, in_f)
    ref_y, ref_A, ref_B, ref_d = _oracle(x.numpy(), w0.numpy(), logits_a.numpy(), logits_b.numpy(),
                                         bank.numpy(), b0.numpy(), topk)
    assert _close(y, ref_y, 1e-6) and _close(A, ref_A, 1e-6)
    assert _close(B, ref_B, 1e-6) and _close(delta, ref_d, 1e-6)


def check_against_oracle(solve):
    cases = [
        (2, 2, 1, 2, 1, 1),
        (6, 12, 2, 8, 3, 2),
        (8, 4, 3, 5, 4, 3),
        (4, 4, 1, 4, 2, 4),
        (6, 9, 2, 6, 3, 1),
        (12, 8, 4, 7, 4, 3),
    ]
    for seed, (in_f, out_f, r, V, vl, topk) in enumerate(cases):
        bank, logits_a, logits_b = _cfg(100 + seed * 7, in_f, out_f, r, V, vl, topk)
        w0 = _rand(200 + seed * 7, out_f, in_f)
        b0 = _rand(300 + seed * 7, out_f) if seed % 2 == 0 else None
        x = _rand(400 + seed * 7, 2, 3, in_f)
        y, A, B, delta = solve(x, w0, logits_a, logits_b, bank, b0, topk)
        ref = _oracle(x.numpy(), w0.numpy(), logits_a.numpy(), logits_b.numpy(), bank.numpy(),
                      None if b0 is None else b0.numpy(), topk)
        assert _close(A, ref[1], 1e-6), (in_f, out_f, r, vl, topk)
        assert _close(B, ref[2], 1e-6), (in_f, out_f, r, vl, topk)
        assert _close(delta, ref[3], 1e-6), (in_f, out_f, r, vl, topk)
        assert _close(y, ref[0], 1e-6), (in_f, out_f, r, vl, topk)
    # float32 输入可跑，输出仍是 float32
    in_f, out_f, r, V, vl, topk = 6, 6, 2, 5, 3, 2
    bank, logits_a, logits_b = _cfg(500, in_f, out_f, r, V, vl, topk)
    w0 = _rand(501, out_f, in_f)
    x = _rand(502, 2, in_f)
    y32, A32, B32, d32 = solve(x.to(torch.float32), w0.to(torch.float32), logits_a.to(torch.float32),
                               logits_b.to(torch.float32), bank.to(torch.float32), None, topk)
    assert all(t.dtype == torch.float32 for t in (y32, A32, B32, d32))
    ref = _oracle(x.numpy(), w0.numpy(), logits_a.numpy(), logits_b.numpy(), bank.numpy(), None, topk)
    assert _close(d32, ref[3], 1e-5) and _close(y32, ref[0], 1e-5)


def check_invalid_inputs(solve):
    in_f, out_f, r, V, vl, topk = 6, 6, 2, 5, 3, 2
    bank, logits_a, logits_b = _cfg(600, in_f, out_f, r, V, vl, topk)
    w0 = torch.zeros(out_f, in_f, dtype=torch.float64)
    x = torch.zeros(2, in_f, dtype=torch.float64)
    b0 = torch.zeros(out_f, dtype=torch.float64)
    bad_calls = [
        ("x-not-tensor", lambda: solve(x.tolist(), w0, logits_a, logits_b, bank, None, topk)),
        ("x-int", lambda: solve(x.to(torch.int64), w0, logits_a, logits_b, bank, None, topk)),
        ("x-bool", lambda: solve(x.to(torch.bool), w0, logits_a, logits_b, bank, None, topk)),
        ("x-nan", lambda: solve(torch.full((2, in_f), float("nan"), dtype=torch.float64), w0, logits_a, logits_b, bank, None, topk)),
        ("x-wrong-last-dim", lambda: solve(torch.zeros(2, 5, dtype=torch.float64), w0, logits_a, logits_b, bank, None, topk)),
        ("w0-not-tensor", lambda: solve(x, w0.tolist(), logits_a, logits_b, bank, None, topk)),
        ("w0-1d", lambda: solve(x, torch.zeros(36, dtype=torch.float64), logits_a, logits_b, bank, None, topk)),
        ("w0-int", lambda: solve(x, w0.to(torch.int32), logits_a, logits_b, bank, None, topk)),
        ("w0-inf", lambda: solve(x, torch.full((out_f, in_f), float("inf"), dtype=torch.float64), logits_a, logits_b, bank, None, topk)),
        ("logits_a-2d", lambda: solve(x, w0, logits_a.reshape(r, -1), logits_b, bank, None, topk)),
        ("logits_a-4d", lambda: solve(x, w0, logits_a.unsqueeze(0), logits_b, bank, None, topk)),
        ("logits_a-num_vectors", lambda: solve(x, w0, torch.zeros(r, in_f // vl, V + 1, dtype=torch.float64), logits_b, bank, None, topk)),
        ("logits_a-tiles", lambda: solve(x, w0, torch.zeros(r, in_f, V, dtype=torch.float64), logits_b, bank, None, topk)),
        ("logits_a-r", lambda: solve(x, w0, torch.zeros(3, in_f // vl, V, dtype=torch.float64), logits_b, bank, None, topk)),
        ("logits_a-int", lambda: solve(x, w0, logits_a.to(torch.int64), logits_b, bank, None, topk)),
        ("logits_a-nan", lambda: solve(x, w0, torch.full_like(logits_a, float("nan")), logits_b, bank, None, topk)),
        ("logits_b-2d", lambda: solve(x, w0, logits_a, logits_b.reshape(-1, V), bank, None, topk)),
        ("logits_b-out_tiles", lambda: solve(x, w0, logits_a, torch.zeros(3, r, V, dtype=torch.float64), bank, None, topk)),
        ("logits_b-r", lambda: solve(x, w0, logits_a, torch.zeros(out_f // vl, 3, V, dtype=torch.float64), bank, None, topk)),
        ("logits_b-inf", lambda: solve(x, w0, logits_a, torch.full_like(logits_b, float("inf")), bank, None, topk)),
        ("bank-not-tensor", lambda: solve(x, w0, logits_a, logits_b, bank.tolist(), None, topk)),
        ("bank-1d", lambda: solve(x, w0, logits_a, logits_b, bank.reshape(-1), None, topk)),
        ("bank-3d", lambda: solve(x, w0, logits_a, logits_b, bank.unsqueeze(0), None, topk)),
        ("bank-num_vectors", lambda: solve(x, w0, logits_a, logits_b, torch.zeros(V + 1, vl, dtype=torch.float64), None, topk)),
        ("bank-vector_length", lambda: solve(x, w0, logits_a, logits_b, torch.zeros(V, 2, dtype=torch.float64), None, topk)),
        ("bank-empty", lambda: solve(x, w0, logits_a, logits_b, torch.zeros(0, vl, dtype=torch.float64), None, topk)),
        ("bank-int", lambda: solve(x, w0, logits_a, logits_b, bank.to(torch.int64), None, topk)),
        ("in-not-divisible", lambda: solve(torch.zeros(2, 5, dtype=torch.float64), torch.zeros(out_f, 5, dtype=torch.float64),
                                           torch.zeros(r, 1, V, dtype=torch.float64), torch.zeros(out_f // vl, r, V, dtype=torch.float64),
                                           torch.zeros(V, 5, dtype=torch.float64), None, topk)),
        ("out-not-divisible", lambda: solve(x, torch.zeros(7, in_f, dtype=torch.float64), logits_a,
                                           torch.zeros(1, r, V, dtype=torch.float64), torch.zeros(V, 1, dtype=torch.float64), None, topk)),
        ("topk-zero", lambda: solve(x, w0, logits_a, logits_b, bank, None, 0)),
        ("topk-negative", lambda: solve(x, w0, logits_a, logits_b, bank, None, -1)),
        ("topk-over-num_vectors", lambda: solve(x, w0, logits_a, logits_b, bank, None, V + 1)),
        ("topk-float", lambda: solve(x, w0, logits_a, logits_b, bank, None, 2.0)),
        ("topk-bool", lambda: solve(x, w0, logits_a, logits_b, bank, None, True)),
        ("r-zero", lambda: solve(x, w0, torch.zeros(0, in_f // vl, V, dtype=torch.float64),
                                torch.zeros(out_f // vl, 0, V, dtype=torch.float64), bank, None, topk)),
        ("b0-wrong-length", lambda: solve(x, w0, logits_a, logits_b, bank, torch.zeros(3, dtype=torch.float64), topk)),
        ("b0-2d", lambda: solve(x, w0, logits_a, logits_b, bank, torch.zeros(out_f, 1, dtype=torch.float64), topk)),
        ("b0-nan", lambda: solve(x, w0, logits_a, logits_b, bank, torch.full((out_f,), float("nan"), dtype=torch.float64), topk)),
        ("b0-not-tensor", lambda: solve(x, w0, logits_a, logits_b, bank, [0.0] * out_f, topk)),
        ("b0-int", lambda: solve(x, w0, logits_a, logits_b, bank, torch.zeros(out_f, dtype=torch.int64), topk)),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
