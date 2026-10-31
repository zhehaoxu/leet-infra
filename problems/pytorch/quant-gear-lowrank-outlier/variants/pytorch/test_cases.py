import math

import numpy as np
import torch

KEYS = {"outlier_mask", "sparse_filled", "quantized", "residual", "low_rank", "output"}
TOL = {torch.float32: 1e-6, torch.float64: 1e-11}


def _oracle(kv, bits, group_size, sparsity, rank, loop, p_base, q_base):
    """纯 numpy 独立重算：两次取整的名额、含离群元的行均值、逐组仿射量化、投影式幂迭代。"""
    x = np.asarray(kv, dtype=np.float64)                       # (B,H,L,D)
    b_n, h_n, l_n, d_n = x.shape
    c = h_n * d_n
    merged = x.transpose(0, 2, 1, 3).reshape(b_n, l_n, c)
    n_out = int(int(b_n * h_n * l_n * d_n * sparsity) / b_n / l_n / 2)
    mask, filled, base = np.zeros((b_n, l_n, c), bool), merged.copy(), np.empty_like(merged)
    for b in range(b_n):
        for l in range(l_n):
            row, mean = merged[b, l], merged[b, l].sum() / c
            small = sorted(range(c), key=lambda j: (row[j], j))[:n_out]
            large = sorted(range(c), key=lambda j: (-row[j], j))[:n_out]
            for j in small + large:                            # 先小后大，重合时大值覆盖
                mask[b, l, j], filled[b, l, j] = True, mean
            out = []
            for g in range(c // group_size):
                seg = filled[b, l, g * group_size:(g + 1) * group_size]
                mn, mx = min(seg), max(seg)
                scale = (mx - mn) / (2 ** bits - 1)
                out += [mn if scale == 0 else
                        round(max((v - mn) / scale, 0.0)) * scale + mn for v in seg]
            base[b, l] = out
            for j in small + large:                             # 原始值写回
                base[b, l, j] = row[j]
    quant = base.reshape(b_n, l_n, h_n, d_n).transpose(0, 2, 1, 3)
    resid, low = x - quant, np.empty_like(x)
    for b in range(b_n):
        for h in range(h_n):
            a, p, q = resid[b, h], np.asarray(p_base, np.float64)[b, h], np.asarray(q_base, np.float64)[b, h]
            for i in range(loop):
                if i == loop - 1:
                    p = np.linalg.qr(p)[0]
                q = a @ p
                if i == loop - 1:
                    q = np.linalg.qr(q)[0]
                p = a.T @ q
            low[b, h] = q @ p.T                                 # = 把 a 投到 span(aP) 上
    return {"outlier_mask": mask.reshape(b_n, l_n, h_n, d_n).transpose(0, 2, 1, 3),
            "sparse_filled": filled.reshape(b_n, l_n, h_n, d_n).transpose(0, 2, 1, 3),
            "quantized": quant, "residual": resid, "low_rank": low, "output": quant + low}


def _verify(solve, kv, args):
    before = [t.clone() for t in [kv] + list(args[-2:])]
    got = solve(kv, *args)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(kv, before[0]) and torch.equal(args[-2], before[1]) \
        and torch.equal(args[-1], before[2]), "输入被就地修改了"
    want = _oracle(kv.numpy(), *args)
    assert got["outlier_mask"].dtype == torch.bool and got["outlier_mask"].shape == kv.shape
    assert torch.equal(got["outlier_mask"], torch.from_numpy(want["outlier_mask"])), "离群掩码不对"
    for key in ("sparse_filled", "quantized", "residual", "low_rank", "output"):
        assert got[key].dtype == kv.dtype, f"{key} 的 dtype 应跟随输入"
        assert got[key].shape == kv.shape, f"{key} 的形状应为 {tuple(kv.shape)}"
        tol = TOL[kv.dtype]
        torch.testing.assert_close(got[key], torch.tensor(want[key], dtype=kv.dtype),
                                   rtol=tol, atol=tol, msg=f"{key} 与 oracle 不一致")
    return got


def _bases(shape_b, shape_h, l_n, d_n, rank, p_fill=1.0, q_fill=0.0, dtype=torch.float64):
    p = torch.full((shape_b, shape_h, d_n, rank), p_fill, dtype=dtype)
    q = torch.full((shape_b, shape_h, l_n, rank), q_fill, dtype=dtype)
    return p, q


def _example_inputs():
    kv = torch.tensor([[[[-4., 1., 2., 3.], [-4., 1., 2., 3.], [-4., 1., 2., 3.]],
                        [[4., 5., 6., 7.], [4., 5., 6., 8.], [4., 5., 6., 9.]]]],
                      dtype=torch.float64)
    p, q = _bases(1, 2, 3, 4, 1)
    return kv, (2, 4, 0.25, 1, 1, p, q)


def _want(v):
    return torch.tensor(v, dtype=torch.float64)


def check_example(solve):
    """题面示例：每行 1 小 1 大离群元，head1 残差恰为秩 1、head0 残差秩 2。"""
    kv, args = _example_inputs()
    got = _verify(solve, kv, args)
    assert got["outlier_mask"][0, 0].tolist() == [[True, False, False, False]] * 3
    assert got["outlier_mask"][0, 1].tolist() == [[False, False, False, True]] * 3
    assert got["sparse_filled"][0, 0].tolist() == [[3.0, 1.0, 2.0, 3.0], [3.125, 1.0, 2.0, 3.0],
                                                   [3.25, 1.0, 2.0, 3.0]], "行均值换掉最小端 -4"
    assert got["sparse_filled"][0, 1].tolist() == [[4.0, 5.0, 6.0, 3.0], [4.0, 5.0, 6.0, 3.125],
                                                   [4.0, 5.0, 6.0, 3.25]], "同一均值也换掉最大端 7/8/9"
    torch.testing.assert_close(got["quantized"][0, 0], _want(
        [[-4.0, 1.0, 7 / 3, 3.0], [-4.0, 1.0, 41 / 24, 3.125], [-4.0, 1.0, 1.75, 3.25]]),
        rtol=0, atol=1e-12, msg="离群元写回原值 -4，组内其余三点量化")
    torch.testing.assert_close(got["quantized"][0, 1], _want(
        [[4.0, 5.0, 6.0, 7.0], [49 / 12, 121 / 24, 6.0, 8.0], [25 / 6, 61 / 12, 6.0, 9.0]]),
        rtol=0, atol=1e-12)
    torch.testing.assert_close(got["residual"][0, 0], _want(
        [[0.0, 0.0, -1 / 3, 0.0], [0.0, 0.0, 7 / 24, -1 / 8], [0.0, 0.0, 1 / 4, -1 / 4]]),
        rtol=0, atol=1e-12)
    torch.testing.assert_close(got["residual"][0, 1], _want(
        [[0.0] * 4, [-1 / 12, -1 / 24, 0.0, 0.0], [-1 / 6, -1 / 12, 0.0, 0.0]]),
        rtol=0, atol=1e-12)
    torch.testing.assert_close(got["low_rank"][0, 1], got["residual"][0, 1], rtol=0, atol=1e-9,
                               msg="head1 残差秩 1，被全额补偿")
    torch.testing.assert_close(got["low_rank"][0, 0], _want(
        [[0.0, 0.0, -23 / 60, 1 / 20], [0.0, 0.0, 23 / 120, -1 / 40], [0.0] * 4]),
        rtol=0, atol=1e-9)
    torch.testing.assert_close(got["output"][0, 1], kv[0, 1], rtol=0, atol=1e-9,
                               msg="head1 逐元素还原回 kv")
    assert not torch.allclose(got["low_rank"][0, 0], got["residual"][0, 0], atol=1e-3), "head0 补不满"
    assert float((kv - got["quantized"]).norm()) > float((kv - got["output"]).norm())


def check_outlier_budget_is_row_wide_and_rounds_down(solve):
    """名额按合并后的整行（跨 head）算，且按 2/C 的粒度向下取整。"""
    kv, args = _example_inputs()
    p, q = args[-2], args[-1]
    for sparsity, want in ((0.24, 0), (0.25, 2), (0.5, 4)):
        got = _verify(solve, kv, (2, 4, sparsity, 1, 1, p, q))
        row_sum = got["outlier_mask"].sum(dim=(0, 1, 3))        # 每个 (b,l) 行的离群元个数
        assert row_sum.tolist() == [want] * 3, (sparsity, row_sum.tolist())
    wide = _verify(solve, kv, (2, 4, 0.5, 1, 1, p, q))
    assert wide["outlier_mask"][0, 0].tolist() == [[True, True, False, False]] * 3, "两个最小值都在 head0"
    assert wide["outlier_mask"][0, 1].tolist() == [[False, False, True, True]] * 3, "两个最大值都在 head1"


def check_row_mean_includes_outliers(solve):
    """替换值是整行原始均值（含离群元），不是去掉两端后的截尾均值。"""
    kv = torch.tensor([[[[-100., 0., 0., 0., 0., 0., 0., 100.]]]], dtype=torch.float64)
    kv = kv.reshape(1, 1, 1, 8)
    p, q = _bases(1, 1, 1, 8, 1)
    got = _verify(solve, kv, (2, 4, 0.25, 1, 1, p, q))
    assert got["sparse_filled"][0, 0, 0].tolist() == [0.0] * 8, "全行均值 0，替换后整行都是 0"
    assert got["outlier_mask"][0, 0, 0].tolist() == [True, False, False, False,
                                                    False, False, False, True]
    assert got["quantized"][0, 0, 0].tolist() == [-100.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 100.0], "常数组反量化成 mn"


def check_ties_pick_smaller_index(solve):
    """同分离群元先拿下标小的：三个并列的 9.0 只有前两个进掩码，行 3 的 5.0 全不进。"""
    kv = torch.tensor([[[[5., 5., 5., 1., 9., 9., 9., 2.]]]], dtype=torch.float64).reshape(1, 1, 1, 8)
    p, q = _bases(1, 1, 1, 8, 1)
    got = _verify(solve, kv, (2, 4, 0.5, 1, 1, p, q))
    assert got["outlier_mask"][0, 0, 0].tolist() == [False, False, False, True,
                                                    True, True, False, True], (
        "小端 = 1.0(idx3) 与 2.0(idx7)；大端两个名额给 idx4/5 这两个并列的 9.0")
    assert got["sparse_filled"][0, 0, 0].tolist() == [5.0, 5.0, 5.0, 45 / 8, 45 / 8, 45 / 8, 9.0, 45 / 8]


def check_low_rank_is_an_orthogonal_projection(solve):
    """loop=1 的 low_rank 把残差投到 span(R·qr(P).Q)：误差不放大，且秩不超过 rank。"""
    torch.manual_seed(5)
    kv = torch.randn(1, 3, 8, 5, dtype=torch.float64)
    p = torch.randn(1, 3, 5, 2, dtype=torch.float64)
    q = torch.randn(1, 3, 8, 2, dtype=torch.float64)
    got = _verify(solve, kv, (2, 5, 0.1, 2, 1, p, q))
    assert float((kv - got["output"]).norm()) <= float((kv - got["quantized"]).norm()) + 1e-12
    assert torch.linalg.matrix_rank(got["low_rank"], rtol=1e-9).max() <= 2, "秩不超过 rank"
    wide = _verify(solve, kv, (2, 5, 0.1, 2, 3, p, q))
    assert float((kv - wide["output"]).norm()) <= float((kv - wide["quantized"]).norm()) + 1e-12


def check_q_base_is_overwritten_and_p_base_matters(solve):
    """loop>=1 时 q_base 第一轮就被覆盖，换它不影响结果；p_base 的方向才决定子空间。"""
    kv, args = _example_inputs()
    a = _verify(solve, kv, args[:-2] + _bases(1, 2, 3, 4, 1, p_fill=1.0, q_fill=0.0))
    b = _verify(solve, kv, args[:-2] + _bases(1, 2, 3, 4, 1, p_fill=1.0, q_fill=-7.5))
    assert torch.equal(a["low_rank"], b["low_rank"]), "q_base 对 loop>=1 完全不起作用"
    c = _verify(solve, kv, args[:-2] + (torch.arange(1 * 2 * 4 * 1,
                                                    dtype=torch.float64).reshape(1, 2, 4, 1) + 1.0,
                                        args[-1]))
    assert not torch.equal(a["low_rank"], c["low_rank"]), "p_base 换成非均匀方向后子空间变了"


def check_dtype_and_no_mutation(solve):
    """float32 入 float32 出，且与 float64 结果一致（内部一律 float64）。"""
    rng = np.random.default_rng(9)
    kv64 = torch.tensor(rng.normal(size=(1, 2, 6, 4)) * 2.0, dtype=torch.float64)
    # 基必须满秩：常数矩阵的 QR 只剩一个方向，第二列是补空间，跨实现不确定
    p64 = torch.tensor(rng.normal(size=(1, 2, 4, 2)), dtype=torch.float64)
    q64 = torch.tensor(rng.normal(size=(1, 2, 6, 2)), dtype=torch.float64)
    lo = _verify(solve, kv64.float(), (3, 4, 0.25, 2, 2, p64.float(), q64.float()))
    hi = _verify(solve, kv64, (3, 4, 0.25, 2, 2, p64, q64))
    assert lo["output"].dtype == torch.float32 and lo["quantized"].dtype == torch.float32
    torch.testing.assert_close(lo["output"].double(), hi["output"], rtol=1e-5, atol=1e-5)
    assert lo["outlier_mask"].tolist() == hi["outlier_mask"].tolist()


def check_low_rank_sign_is_irrelevant(solve):
    """low_rank = Q Qᵀ A 只依赖子空间：把基整体取负，结果逐元素不变。"""
    rng = np.random.default_rng(11)
    kv = torch.tensor(rng.normal(size=(2, 3, 7, 4)) * 1.5, dtype=torch.float64)
    p = torch.tensor(rng.normal(size=(2, 3, 4, 2)), dtype=torch.float64)
    q = torch.tensor(rng.normal(size=(2, 3, 7, 2)), dtype=torch.float64)
    a = _verify(solve, kv, (2, 4, 0.25, 2, 3, p, q))["low_rank"]
    b = _verify(solve, kv, (2, 4, 0.25, 2, 3, -p, -q))["low_rank"]
    torch.testing.assert_close(a, b, rtol=0, atol=1e-9, msg="正交基的符号/补空间选择不影响 low_rank")


def check_random_against_oracle(solve):
    """多形状网格（含非连续输入、多位宽、多 loop）逐键对拍。"""
    rng = np.random.default_rng(20261030)
    for b_n, h_n, l_n, d_n, gs, bits, sp, rank, loop in [
            (2, 3, 8, 4, 6, 2, 0.25, 1, 1), (1, 2, 9, 6, 3, 3, 0.12, 2, 2),
            (2, 1, 6, 8, 8, 4, 0.5, 3, 1), (1, 4, 5, 4, 4, 2, 0.0, 1, 3),
            (3, 2, 7, 4, 8, 2, 0.375, 2, 2)]:
        kv = torch.tensor(rng.normal(size=(b_n, h_n, l_n, d_n)) * 3.0, dtype=torch.float64)
        p = torch.tensor(rng.normal(size=(b_n, h_n, d_n, rank)), dtype=torch.float64)
        q = torch.tensor(rng.normal(size=(b_n, h_n, l_n, rank)), dtype=torch.float64)
        _verify(solve, kv, (bits, gs, sp, rank, loop, p, q))
    big = torch.tensor(rng.normal(size=(1, 2, 10, 4)), dtype=torch.float64)
    p, q = _bases(1, 2, 10, 4, 1, dtype=torch.float64)
    sliced = big[:, :, ::2, :]
    assert sliced.stride(2) != 4, "非连续视图才能验证 reshape 的口径"
    _verify(solve, sliced, (2, 4, 0.25, 1, 2, p, q[:, :, ::2, :]))


def check_value_errors(solve):
    """维度、dtype、整除、名额与基张量形状不合法都要抛 ValueError。"""
    kv, args = _example_inputs()
    ok = _verify(solve, kv, args)
    assert ok["output"].shape == kv.shape
    p, q = args[-2], args[-1]
    illegal = [
        dict(bits=0), dict(bits=2.0), dict(bits=True), dict(group_size=3), dict(group_size=0),
        dict(sparsity=1.0), dict(sparsity=-0.1), dict(sparsity=True),
        dict(rank=0), dict(rank=5), dict(loop=0), dict(loop=1.5),
    ]
    for name, over in enumerate(illegal):
        call = dict(zip(("bits", "group_size", "sparsity", "rank", "loop"), args[:5]))
        call.update(over)
        try:
            solve(kv, call["bits"], call["group_size"], call["sparsity"], call["rank"],
                  call["loop"], p, q)
        except ValueError:
            continue
        raise AssertionError(f"第 {name} 组参数应抛 ValueError：{over}")
    for wrong in (torch.zeros(1, 2, 4, 2, dtype=torch.float64), torch.zeros(2, 2, 4, 1,
                                                                            dtype=torch.float64),
                  torch.zeros(1, 2, 4, 1, dtype=torch.float32)):
        try:
            solve(kv, 2, 4, 0.25, 1, 1, wrong, q)
        except ValueError:
            continue
        raise AssertionError(f"p_base={tuple(wrong.shape)}/{wrong.dtype} 应抛 ValueError")
    for bad_kv in (kv[0], kv.to(torch.int32)):
        try:
            solve(bad_kv, 2, 4, 0.25, 1, 1, p, q)
        except ValueError:
            continue
        raise AssertionError("非四维 / 整数 dtype 的 kv 应抛 ValueError")
    try:
        solve(kv.reshape(2, 1, 3, 4), 2, 4, 0.25, 1, 1, p, q)
    except ValueError:
        pass
    else:
        raise AssertionError("group_size 不整除 H*D 应抛 ValueError")


check_example.example = True
