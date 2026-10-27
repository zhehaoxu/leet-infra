import itertools

import torch
import torch.nn.functional as F

# 示例期望值（实测回填）：2x3 手写小图 unfold(k=2, s=1) + 一维 roll 2 步
EX_IMAGE = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]  # (1, 1, 2, 3)
EXAMPLE_UNFOLD_COLS = [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0], [5.0, 6.0]]
EXAMPLE_ROLL_INPUT = [1.0, 2.0, 3.0, 4.0, 5.0]
EXAMPLE_ROLL = [4.0, 5.0, 1.0, 2.0, 3.0]


def check_example(solve):
    x = torch.tensor(EX_IMAGE)
    cols = solve(x, 2, 1, mode="unfold")
    assert cols.shape == (1, 4, 2), cols.shape
    assert torch.equal(cols, torch.tensor([EXAMPLE_UNFOLD_COLS])), cols.tolist()
    r = torch.tensor(EXAMPLE_ROLL_INPUT)
    rolled = solve(r, 1, mode="roll", shifts=2)  # dims=None -> 最后一维
    assert torch.equal(rolled, torch.tensor(EXAMPLE_ROLL)), rolled.tolist()


check_example.example = True


def check_unfold_matches_native(solve):
    combos = [
        (2, 1, 1, 0), (2, 2, 1, 0), (3, 1, 1, 1), (2, 1, 2, 0),
        (2, 2, 1, 1), ((2, 3), (1, 2), 1, 0), ((3, 2), (2, 2), (1, 2), (1, 0)),
    ]
    for seed, shape in ((0, (2, 3, 5, 6)), (1, (1, 2, 4, 4))):
        torch.manual_seed(10 + seed)
        x = torch.randn(shape)
        for k, s, d, p in combos:
            got = solve(x, k, s, dilation=d, padding=p, mode="unfold")
            exp = F.unfold(x, k, stride=s, dilation=d, padding=p)
            assert got.shape == exp.shape, (k, s, d, p, got.shape, exp.shape)
            assert torch.allclose(got, exp, atol=1e-6), (k, s, d, p)
    # stride=None -> kernel_size（本题约定；原生默认 stride=1，需显式传入对齐）
    torch.manual_seed(13)
    x = torch.randn(1, 2, 4, 5)
    assert torch.allclose(solve(x, 2, mode="unfold"), F.unfold(x, 2, stride=2), atol=1e-6)
    assert torch.allclose(solve(x, (2, 3), mode="unfold"),
                          F.unfold(x, (2, 3), stride=(2, 3)), atol=1e-6)


def check_unfold_window_ordering(solve):
    """行排列 c -> kh -> kw、列排列 hi -> wi（与 F.unfold 实测一致）。"""
    x = torch.arange(24.0).reshape(1, 2, 3, 4)
    cols = solve(x, 2, 1, mode="unfold")
    assert cols.shape == (1, 8, 6)
    # 窗口 (0,0)：c0 的 [[0,1],[4,5]] 在前 4 行，c1 的 [[12,13],[16,17]] 在后 4 行
    assert torch.equal(cols[0, :, 0], torch.tensor([0.0, 1.0, 4.0, 5.0, 12.0, 13.0, 16.0, 17.0])), \
        cols[0, :, 0].tolist()
    assert torch.allclose(cols, F.unfold(x, 2, stride=1), atol=1e-6)
    # 零填充窗口：pad=1, k=3 的窗口 (0,0)
    x1 = torch.arange(1.0, 13.0).reshape(1, 1, 3, 4)
    cols1 = solve(x1, 3, 1, padding=1, mode="unfold")
    assert torch.equal(cols1[0, :, 0],
                       torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 5.0, 6.0])), \
        cols1[0, :, 0].tolist()


def check_unfold_dilation(solve):
    """dilation>1 跳格窗口：3x4 图、k=2、d=2 时 H_out=1、W_out=2。"""
    x = torch.arange(1.0, 13.0).reshape(1, 1, 3, 4)
    cols = solve(x, 2, 1, dilation=2, mode="unfold")
    assert cols.shape == (1, 4, 2)
    assert torch.equal(cols, torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 10.0],
                                            [11.0, 12.0]]])), cols.tolist()
    assert torch.allclose(cols, F.unfold(x, 2, stride=1, dilation=2), atol=1e-6)


def check_fold_matches_native(solve):
    torch.manual_seed(21)
    x = torch.randn(2, 3, 5, 6)
    for k, s, d, p in ((2, 1, 1, 0), (2, 2, 1, 0), (3, 1, 1, 1), (2, 1, 2, 0),
                       ((2, 3), (1, 2), 1, 0)):
        cols = F.unfold(x, k, stride=s, dilation=d, padding=p)
        oh, ow = 5, 6
        got = solve(cols, k, s, dilation=d, padding=p, mode="fold", output_size=(oh, ow))
        exp = F.fold(cols, output_size=(oh, ow), kernel_size=k, stride=s,
                     dilation=d, padding=p)
        assert got.shape == exp.shape, (k, s, d, p)
        assert torch.allclose(got, exp, atol=1e-6), (k, s, d, p)
    # 重叠区求和累加：全 1 cols、k=2 s=1 -> 角 1、边 2、中心 4
    cols = torch.ones(1, 4, 6)
    got = solve(cols, 2, 1, mode="fold", output_size=(3, 4))
    assert torch.equal(got, torch.tensor([[[[1.0, 2.0, 2.0, 1.0],
                                            [2.0, 4.0, 4.0, 2.0],
                                            [1.0, 2.0, 2.0, 1.0]]]])), got.tolist()


def check_roundtrip_identity(solve):
    """stride=kernel、padding=0（且空间维可整除）时 fold(unfold(x)) == x 恒等。"""
    torch.manual_seed(33)
    for shape, k in (((2, 3, 4, 4), 2), ((1, 2, 6, 6), 3), ((1, 1, 4, 4), 4)):
        x = torch.randn(shape)
        cols = solve(x, k, k, mode="unfold")
        rec = solve(cols, k, k, mode="fold", output_size=shape[2:])
        assert torch.equal(rec, x), (shape, k)
    # padding>0 且 (H+2p) % k == 0：裁边后同样恒等
    x = torch.randn(1, 2, 4, 4)
    cols = solve(x, 2, 2, padding=1, mode="unfold")
    rec = solve(cols, 2, 2, padding=1, mode="fold", output_size=(4, 4))
    assert torch.equal(rec, x)


def check_roll_matches_native(solve):
    torch.manual_seed(44)
    r2 = torch.randn(2, 3)
    r3 = torch.randn(2, 3, 4)
    r1 = torch.randn(5)
    cases = [
        (r2, 1, 0), (r2, -2, 1), (r2, (1, 2), (0, 1)), (r2, (0, 0), (0, 1)),
        (r2, 1, -1), (r2, 1, -2),
        (r3, 2, -1), (r3, (1, 2), (0, 2)), (r3, (2, -3), (-1, 1)),
        (r3, 7, 0),  # 幅度超过维度长 -> 取模
        (r1, 7, 0), (r1, -7, 0), (r1, 0, 0),
    ]
    for x, shifts, dims in cases:
        got = solve(x, 1, mode="roll", shifts=shifts, dims=dims)
        exp = torch.roll(x, shifts, dims)
        assert torch.equal(got, exp), (shifts, dims, got.tolist(), exp.tolist())


def check_roll_conventions(solve):
    """dims=None -> 最后一维；shifts 为 int + dims 元组 -> 只 roll dims[0]。
    这两条是本题约定（与原生 torch.roll 的展平行为/报错行为不同，见 starter）。"""
    r2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = solve(r2, 1, mode="roll", shifts=1)  # dims=None -> 最后一维
    assert torch.equal(got, torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])), got.tolist()
    # 原生是展平后 roll：[[6,1,2],[3,4,5]]——刻意不同，钉死约定
    assert torch.equal(torch.roll(r2, 1), torch.tensor([[6.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
    got = solve(r2, 1, mode="roll", shifts=1, dims=(0, 1))  # 只 roll dims[0]
    assert torch.equal(got, torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])), got.tolist()
    # shifts=0 恒等
    assert torch.equal(solve(r2, 1, mode="roll", shifts=0, dims=1), r2)
    assert torch.equal(solve(r2, 1, mode="roll", shifts=(0, 0), dims=(0, 1)), r2)
    # 取模：长度 5 上 shift 7 等价 shift 2；负 shift 等价正 shift
    r1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.equal(solve(r1, 1, mode="roll", shifts=7, dims=0),
                       solve(r1, 1, mode="roll", shifts=2, dims=0))
    assert torch.equal(solve(r1, 1, mode="roll", shifts=-2, dims=0),
                       solve(r1, 1, mode="roll", shifts=3, dims=0))
    # 错误约定
    for kwargs in ({"shifts": None}, {"shifts": (1, 2)}, {"shifts": (1, 2), "dims": (0,)}):
        try:
            solve(r2, 1, mode="roll", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"roll 参数不合法应抛 ValueError: {kwargs}")


def check_errors(solve):
    x = torch.arange(1.0, 13.0).reshape(1, 1, 3, 4)
    # unfold：kernel 大于 pad 后尺寸 -> RuntimeError（与原生一致）
    for k in (4, (5, 5)):
        try:
            solve(x, k, 1, mode="unfold")
        except RuntimeError:
            continue
        raise AssertionError(f"kernel 大于 pad 后尺寸应抛 RuntimeError: {k}")
    # unfold：非 4 维输入
    try:
        solve(torch.zeros(2, 3), 1, 1, mode="unfold")
    except ValueError:
        pass
    else:
        raise AssertionError("unfold 非 4 维输入应抛 ValueError")
    # fold：C*kh*kw 不匹配
    try:
        solve(torch.ones(1, 5, 6), 2, 1, mode="fold", output_size=(3, 4))
    except RuntimeError:
        pass
    else:
        raise AssertionError("fold 通道数不匹配应抛 RuntimeError")
    # fold：L 与 output_size 算出的滑窗数不一致（过大）
    cols = F.unfold(x, 2, stride=2)  # L=2
    try:
        solve(cols, 2, 2, mode="fold", output_size=(5, 6))
    except RuntimeError:
        pass
    else:
        raise AssertionError("fold output_size 不匹配应抛 RuntimeError")
    # fold：output_size 缺省
    try:
        solve(cols, 2, 2, mode="fold")
    except ValueError:
        pass
    else:
        raise AssertionError("fold 缺 output_size 应抛 ValueError")
    # fold：cols 非 3 维
    try:
        solve(torch.ones(4, 6), 2, 1, mode="fold", output_size=(3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("fold 非 3 维输入应抛 ValueError")
    # 非法 mode
    try:
        solve(x, 2, 1, mode="unf")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 mode 应抛 ValueError")


def _unfold_oracle(x, k, s, d, p):
    """独立多重循环 oracle：逐窗口逐核元素写 col（含零填充边界判断）。"""
    n, c, h, w = x.shape
    kh, kw = k
    sh, sw = s
    dh, dw = d
    ph, pw = p
    h_out = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    xl = x.tolist()
    cols = [[[0.0] * (h_out * w_out) for _ in range(c * kh * kw)] for _ in range(n)]
    for nn in range(n):
        for cc in range(c):
            for hi in range(h_out):
                for wi in range(w_out):
                    for i in range(kh):
                        for j in range(kw):
                            hh = hi * sh + i * dh - ph
                            ww = wi * sw + j * dw - pw
                            v = xl[nn][cc][hh][ww] if 0 <= hh < h and 0 <= ww < w else 0.0
                            cols[nn][cc * kh * kw + i * kw + j][hi * w_out + wi] = v
    return torch.tensor(cols)


def _fold_oracle(cols, output_size, k, s, d, p):
    """独立多重循环 oracle：逐窗口逐核元素累加（重叠求和）。"""
    n, ckk, l = cols.shape
    kh, kw = k
    sh, sw = s
    dh, dw = d
    ph, pw = p
    oh, ow = output_size
    c = ckk // (kh * kw)
    h_out = (oh + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (ow + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cl = cols.tolist()
    out = [[[[0.0] * ow for _ in range(oh)] for _ in range(c)] for _ in range(n)]
    for nn in range(n):
        for cc in range(c):
            for hi in range(h_out):
                for wi in range(w_out):
                    for i in range(kh):
                        for j in range(kw):
                            v = cl[nn][cc * kh * kw + i * kw + j][hi * w_out + wi]
                            ohh = hi * sh + i * dh - ph
                            oww = wi * sw + j * dw - pw
                            if 0 <= ohh < oh and 0 <= oww < ow:
                                out[nn][cc][ohh][oww] += v
    return torch.tensor(out)


def _roll_oracle(x, shifts, dims):
    """独立循环 oracle：out[pos] = x[(pos - shift) mod n]，即 (i+shift) mod n 重排。"""
    ndim = x.dim()
    if isinstance(shifts, (tuple, list)):
        shift_list = [int(v) for v in shifts]
        dim_list = [int(v) for v in dims]
    else:
        s = int(shifts)
        if dims is None:
            dim_list = [ndim - 1]
        elif isinstance(dims, (tuple, list)):
            dim_list = [int(dims[0])]
        else:
            dim_list = [int(dims)]
        shift_list = [s]
    shift_map = {}
    for s, d in zip(shift_list, dim_list):
        d = d + ndim if d < 0 else d
        shift_map[d] = s
    xl = x.tolist()
    shape = tuple(x.shape)

    def nested(sz):
        if len(sz) == 1:
            return [0.0] * sz[0]
        return [nested(sz[1:]) for _ in range(sz[0])]

    ol = nested(shape)
    for pos in itertools.product(*(range(v) for v in shape)):
        src = [(pos[k] - shift_map.get(k, 0)) % shape[k] for k in range(ndim)]
        t_o, t_i = ol, xl
        for k in range(ndim - 1):
            t_o, t_i = t_o[pos[k]], t_i[src[k]]
        t_o[pos[-1]] = t_i[src[-1]]
    return torch.tensor(ol, dtype=x.dtype).reshape(shape)


def check_python_oracles(solve):
    torch.manual_seed(55)
    x = torch.randn(2, 2, 4, 5)
    for k, s, d, p in (((2, 2), (1, 1), (1, 1), (1, 1)), (2, 2, 2, 0), ((3, 2), (1, 2), 1, (0, 1))):
        got = solve(x, k, s, dilation=d, padding=p, mode="unfold")
        exp = _unfold_oracle(x, _p(k), _p(s), _p(d), _p(p))
        assert torch.allclose(got, exp, atol=1e-6), (k, s, d, p)
        cols = got
        oh, ow = 4, 5
        got_f = solve(cols, k, s, dilation=d, padding=p, mode="fold", output_size=(oh, ow))
        exp_f = _fold_oracle(cols, (oh, ow), _p(k), _p(s), _p(d), _p(p))
        assert torch.allclose(got_f, exp_f, atol=1e-6), (k, s, d, p)
    r3 = torch.randn(2, 3, 4)
    for shifts, dims in ((1, None), (2, 0), ((1, 2), (0, 2)), (-3, 1), (5, -1)):
        got = solve(r3, 1, mode="roll", shifts=shifts, dims=dims)
        exp = _roll_oracle(r3, shifts, dims)
        assert torch.equal(got, exp), (shifts, dims, got.tolist(), exp.tolist())


def _p(v):
    if isinstance(v, int):
        return (v, v)
    return tuple(v)
