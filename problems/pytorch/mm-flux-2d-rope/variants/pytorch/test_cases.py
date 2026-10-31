"""FLUX.1 二维 RoPE 判题用例。

`_oracle` 是独立的纯 Python 标量实现：嵌套循环搭网格 ids、逐对用 math.pow 现算
inv_freq、math.cos/math.sin 现算三角值、手工复制做 repeat_interleave 的语义、
逐对做 2x2 旋转；不调用 torch.outer / repeat_interleave / 张量 cos-sin。
"""

import math
import struct

import torch


def _f32(value):
    """把一个 Python float 按 IEEE-754 binary32 舍入，等价上游的 `.float()` 落表。"""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _oracle(query, image_grid, text_len, axes_dim, theta):
    height = int(image_grid[0])
    width = int(image_grid[1])

    # 1) ids：文本全零在前，图像按 (行, 列) 行主序在后，帧轴恒 0。
    grid_ids = []
    for row in range(height):
        for column in range(width):
            grid_ids.append([0.0, float(row), float(column)])
    ids = [[0.0, 0.0, 0.0] for _ in range(text_len)] + grid_ids

    # 2) 每个轴各自的 inv_freq，指数分母是该轴 dim。
    inv_freqs = []
    pairs_per_axis = []
    for axis_dim in axes_dim:
        pairs = axis_dim // 2
        pairs_per_axis.append(pairs)
        for index in range(pairs):
            inv_freqs.append(float(theta) ** (-float(2 * index) / float(axis_dim)))

    # 3) 角度 = 位置（先过 float32）× inv_freq，逐对标量相乘。
    angles = []
    for position_row in ids:
        row_angles = []
        offset = 0
        for axis_index, pairs in enumerate(pairs_per_axis):
            position = _f32(position_row[axis_index])
            for index in range(pairs):
                row_angles.append(position * inv_freqs[offset + index])
            offset += pairs
        angles.append(row_angles)

    # 4) 三角值在 double 里算、再舍到 float32，并把每对复制成两列。
    freqs_cos, freqs_sin = [], []
    for row_angles in angles:
        cos_row, sin_row = [], []
        for value in row_angles:
            cosine = _f32(math.cos(value))
            sine = _f32(math.sin(value))
            cos_row.extend((cosine, cosine))
            sin_row.extend((sine, sine))
        freqs_cos.append(cos_row)
        freqs_sin.append(sin_row)

    # 5) 相邻对 2x2 旋转，逐对标量算术。
    values = query.to(torch.float64).tolist()
    pair_count = len(angles[0])
    rotated = []
    for batch_row in values:
        for sequence_index, token in enumerate(batch_row):
            out_row = []
            for pair in range(pair_count):
                cosine = freqs_cos[sequence_index][2 * pair]
                sine = freqs_sin[sequence_index][2 * pair]
                even = token[2 * pair]
                odd = token[2 * pair + 1]
                out_row.append(even * cosine - odd * sine)
                out_row.append(odd * cosine + even * sine)
            rotated.append(out_row)

    return (
        torch.tensor(rotated, dtype=query.dtype).reshape(
            len(values), len(ids), sum(axes_dim)
        ),
        torch.tensor(freqs_cos, dtype=torch.float32),
        torch.tensor(freqs_sin, dtype=torch.float32),
        torch.tensor(ids, dtype=torch.float64),
        torch.tensor(angles, dtype=torch.float64),
        torch.tensor(inv_freqs, dtype=torch.float64),
    )


def _close(actual, expected):
    tolerance = 4e-5 if actual.dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _case(b=2, text_len=3, height=2, width=3, axes_dim=(16, 56, 56), theta=10000.0,
          dtype=torch.float32, seed=19):
    generator = torch.Generator().manual_seed(seed)
    seq_len = text_len + height * width
    query = torch.randn(b, seq_len, sum(axes_dim), dtype=dtype, generator=generator) * 3.0
    return query, torch.tensor([height, width], dtype=torch.int64), text_len, tuple(axes_dim), theta


def _unit_rows(batch, seq_len, head_dim, dtype=torch.float32):
    """偶数维全 1、奇数维全 0 的 query：旋转后每对直接读出 (cos, sin)。"""
    row = [value for pair in range(head_dim // 2) for value in (1.0, 0.0)]
    return torch.tensor([[row] * seq_len] * batch, dtype=dtype)


def _example_small():
    return (_unit_rows(1, 3, 10), torch.tensor([1, 2]), 1, (2, 4, 4), 10000.0)


def _example_flux():
    return (_unit_rows(1, 8, 128), torch.tensor([2, 3]), 2, (16, 56, 56), 10000.0)


def _verify(solve, args):
    query = args[0]
    expected = _oracle(*args)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 6, f"须返回 6 元组，得到 {type(result)}"
    names = ("rotated", "freqs_cos", "freqs_sin", "ids", "angles", "inv_freqs")
    for name, actual, target in zip(names, result, expected):
        assert isinstance(actual, torch.Tensor), f"{name} 必须是张量"
        assert actual.shape == target.shape, (name, tuple(actual.shape), tuple(target.shape))
        assert actual.dtype == target.dtype, (name, actual.dtype, target.dtype)
        assert actual.device == query.device, name
        assert torch.isfinite(actual).all(), f"{name} 出现非有限值"
        if name == "ids":
            assert torch.equal(actual, target), "ids 必须逐位等于网格编号"
        else:
            _close(actual, target)
    rotated, freqs_cos, freqs_sin, _ids, _angles, _inv = result
    assert rotated.dtype == query.dtype, "rotated 必须保持 query dtype"
    assert freqs_cos.dtype == torch.float32 and freqs_sin.dtype == torch.float32, "频率表恒为 float32"
    assert torch.equal(freqs_cos[:, 0::2], freqs_cos[:, 1::2]), "cos 必须成对复制"
    assert torch.equal(freqs_sin[:, 0::2], freqs_sin[:, 1::2]), "sin 必须成对复制"
    return result


def _expect_value_error(solve, args, label):
    try:
        solve(*args)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {label}, got a normal return")


# ---- 样例 1 / 样例 2 -----------------------------------------------------------------
def check_example_small_axes(solve):
    query, grid, text_len, axes_dim, theta = _example_small()
    result = _verify(solve, (query, grid, text_len, axes_dim, theta))
    rotated, freqs_cos, freqs_sin, ids, angles, inv_freqs = result
    assert ids.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], ids.tolist()
    assert angles[2].tolist() == [0.0, 0.0, 0.0, 1.0, 0.01], angles[2].tolist()
    assert inv_freqs.tolist() == [1.0, 1.0, 0.01, 1.0, 0.01], inv_freqs.tolist()
    expected_cos = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5403022766113281, 0.5403022766113281,
                    0.9999499917030334, 0.9999499917030334]
    expected_sin = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8414709568023682, 0.8414709568023682,
                    0.009999833069741726, 0.009999833069741726]
    assert torch.allclose(freqs_cos[2], torch.tensor(expected_cos), atol=1e-6), freqs_cos[2].tolist()
    assert torch.allclose(freqs_sin[2], torch.tensor(expected_sin), atol=1e-6), freqs_sin[2].tolist()
    expected_rotated = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5403022766113281, 0.8414709568023682,
                        0.9999499917030334, 0.009999833069741726]
    assert torch.allclose(rotated[0, 2], torch.tensor(expected_rotated), atol=1e-6), rotated[0, 2].tolist()
    assert torch.equal(rotated[0, 0], query[0, 0]), "文本行须逐位恒等"
    assert torch.equal(rotated[0, 1], query[0, 1]), "(0,0) 图像 token 须逐位恒等"


check_example_small_axes.example = True


def check_example_flux_defaults(solve):
    query, grid, text_len, axes_dim, theta = _example_flux()
    result = _verify(solve, (query, grid, text_len, axes_dim, theta))
    rotated, freqs_cos, freqs_sin, ids, angles, inv_freqs = result
    assert tuple(rotated.shape) == (1, 8, 128) and tuple(freqs_cos.shape) == (8, 128)
    assert tuple(angles.shape) == (8, 64) and tuple(inv_freqs.shape) == (64,)
    assert ids.tolist()[:3] == [[0.0, 0.0, 0.0]] * 3
    assert ids[3].tolist() == [0.0, 0.0, 1.0] and ids[5].tolist() == [0.0, 1.0, 0.0]
    assert ids[7].tolist() == [0.0, 1.0, 2.0]
    assert torch.allclose(inv_freqs[:8],
                          torch.tensor([1.0, 0.31622776601683794, 0.1, 0.03162277660168379,
                                        0.01, 0.0031622776601683794, 0.001,
                                        0.00031622776601683794], dtype=torch.float64),
                          atol=1e-12), inv_freqs[:8].tolist()
    assert abs(inv_freqs[8].item() - 1.0) < 1e-15
    assert abs(inv_freqs[9].item() - 0.719685673001152) < 1e-12, inv_freqs[9].item()
    assert abs(inv_freqs[36].item() - 1.0) < 1e-15, "轴 2 从下标 36 起，自己的第一频为 1"
    assert torch.count_nonzero(angles[:3]) == 0, "文本行角度全 0"
    assert abs(angles[7][8].item() - 1.0) < 1e-15 and abs(angles[7][36].item() - 2.0) < 1e-15
    assert torch.allclose(freqs_cos[7][16:22],
                          torch.tensor([0.5403022766113281, 0.5403022766113281,
                                        0.7520129680633545, 0.7520129680633545,
                                        0.8688372373580933, 0.8688372373580933]),
                          atol=1e-6), freqs_cos[7][16:22].tolist()
    assert torch.allclose(rotated[0, 7][16:22],
                          torch.tensor([0.5403022766113281, 0.8414709568023682,
                                        0.7520129680633545, 0.6591483354568481,
                                        0.8688372373580933, 0.4950978755950928]),
                          atol=1e-6), rotated[0, 7][16:22].tolist()
    assert torch.equal(rotated[0, 0], query[0, 0]) and torch.equal(rotated[0, 1], query[0, 1])


check_example_flux_defaults.example = True


# ---- 对拍与不变量 --------------------------------------------------------------------
def check_matches_oracle_configs(solve):
    for text_len, height, width, axes_dim in (
        (0, 1, 1, (2, 2, 2)),
        (1, 1, 2, (2, 4, 4)),
        (2, 2, 3, (4, 4, 4)),
        (3, 4, 5, (16, 56, 56)),
        (5, 2, 2, (2, 8, 22)),
    ):
        _verify(solve, _case(b=2, text_len=text_len, height=height, width=width,
                             axes_dim=axes_dim, theta=10000.0))


def check_seeded_random_sweep(solve):
    configs = [
        dict(b=1, text_len=0, height=1, width=1, axes_dim=(2, 2, 2), theta=10000.0),
        dict(b=3, text_len=1, height=1, width=4, axes_dim=(2, 4, 4), theta=1.0),
        dict(b=2, text_len=2, height=3, width=2, axes_dim=(4, 4, 4), theta=100.0),
        dict(b=1, text_len=4, height=2, width=5, axes_dim=(8, 16, 16), theta=10000.0),
        dict(b=3, text_len=0, height=4, width=3, axes_dim=(16, 56, 56), theta=10000.0),
        dict(b=2, text_len=3, height=2, width=2, axes_dim=(2, 4, 54), theta=7.0,
             dtype=torch.float64),
        dict(b=1, text_len=6, height=5, width=4, axes_dim=(6, 10, 2), theta=1e6,
             dtype=torch.float64),
        dict(b=3, text_len=2, height=3, width=3, axes_dim=(16, 56, 56), theta=500.0,
             dtype=torch.float64),
        dict(b=2, text_len=1, height=2, width=7, axes_dim=(2, 2, 14), theta=3.0),
        dict(b=1, text_len=7, height=6, width=1, axes_dim=(16, 56, 56), theta=1e9),
    ]
    for index, config in enumerate(configs):
        _verify(solve, _case(seed=101 + index, **config))


def check_first_sixteen_columns_identity(solve):
    # FLUX.1 的帧轴恒 0 且 axes_dim[0]=16，故每张表前 16 列恒为 cos=1 / sin=0。
    args = _case(b=3, text_len=4, height=5, width=3, axes_dim=(16, 56, 56), theta=10000.0,
                 dtype=torch.float64)
    rotated, freqs_cos, freqs_sin, ids = _verify(solve, args)[:4]
    assert (freqs_cos[:, :16] == 1).all(), freqs_cos[:, :16].unique().tolist()
    assert (freqs_sin[:, :16] == 0).all(), freqs_sin[:, :16].unique().tolist()
    assert torch.equal(freqs_cos[:, :16], torch.ones_like(freqs_cos[:, :16]))
    assert torch.equal(rotated[..., :16], args[0][..., :16]), "前 16 列必须逐位不动"
    assert (ids[:, 0] == 0).all()


def check_text_rows_fully_identity(solve):
    for dtype in (torch.float32, torch.float64):
        args = _case(b=2, text_len=3, height=2, width=4, dtype=dtype, seed=5)
        rotated, freqs_cos, freqs_sin, ids = _verify(solve, args)[:4]
        assert torch.equal(rotated[:, :3], args[0][:, :3]), "文本 token 不受旋转"
        assert torch.equal(freqs_cos[:3], torch.ones_like(freqs_cos[:3]))
        assert torch.equal(freqs_sin[:3], torch.zeros_like(freqs_sin[:3]))
        assert torch.equal(ids[:3], torch.zeros(3, 3, dtype=torch.float64))


def check_theta_one_degeneracy(solve):
    # theta=1 时每个 inv_freq 都恰为 1.0，角度退化为位置本身；位置全 0 的行
    # （所有文本行 + (0,0) 格）角度为 0，于是 cos=1、sin=0、rotated 与 query 逐位相等。
    query, grid, text_len, axes_dim, _theta = _case(b=2, text_len=3, height=2, width=3,
                                                    axes_dim=(4, 4, 4), dtype=torch.float64)
    rotated, freqs_cos, freqs_sin, ids, angles, inv_freqs = _verify(
        solve, (query, grid, text_len, axes_dim, 1.0))
    assert torch.equal(inv_freqs, torch.ones(6, dtype=torch.float64))
    for index in range(ids.shape[0]):
        expected_angles = [float(ids[index, axis].item()) for axis in (0, 0, 1, 1, 2, 2)]
        assert angles[index].tolist() == expected_angles, (index, angles[index].tolist())
    zero_rows = [index for index in range(ids.shape[0]) if not ids[index].any()]
    assert zero_rows == [0, 1, 2, 3], zero_rows
    assert torch.equal(angles[zero_rows], torch.zeros(4, 6, dtype=torch.float64))
    assert torch.equal(freqs_cos[zero_rows], torch.ones(4, 12, dtype=torch.float32))
    assert torch.equal(freqs_sin[zero_rows], torch.zeros(4, 12, dtype=torch.float32))
    assert torch.equal(rotated[:, zero_rows], query[:, zero_rows]), "角度 0 的行必须严格恒等"
    # theta=1 时轴 1 的两对角度都等于行号本身，于是都退化成 cos(1.0)
    assert freqs_cos[6, 4] == _f32(math.cos(1.0)) and freqs_sin[6, 4] == _f32(math.sin(1.0))
    cosine = _f32(math.cos(1.0))
    assert torch.equal(freqs_cos[6], torch.tensor(
        [1.0] * 4 + [cosine] * 4 + [1.0] * 4, dtype=torch.float32)), freqs_cos[6].tolist()


def check_single_cell_grid_and_zero_text(solve):
    # 1x1 网格 + text_len=0：只有一行 ids 且全 0，整表 cos=1 / sin=0。
    query = _unit_rows(1, 1, 128, dtype=torch.float64)
    args = (query, torch.tensor([1, 1]), 0, (16, 56, 56), 10000.0)
    rotated, freqs_cos, freqs_sin, ids, angles = _verify(solve, args)[:5]
    assert tuple(ids.shape) == (1, 3) and torch.equal(ids, torch.zeros(1, 3, dtype=torch.float64))
    assert torch.equal(freqs_cos, torch.ones(1, 128, dtype=torch.float32))
    assert torch.equal(freqs_sin, torch.zeros(1, 128, dtype=torch.float32))
    assert torch.equal(rotated, query)
    assert torch.count_nonzero(angles) == 0
    # text_len=0 时表只含图像行，行主序 (row, col)
    zero_text = _case(b=2, text_len=0, height=3, width=2, axes_dim=(2, 4, 4), seed=8)
    table_ids = _verify(solve, zero_text)[3]
    assert table_ids.tolist() == [[0.0, row, col] for row in range(3) for col in range(2)]


def check_per_axis_denominator(solve):
    # 轴维为 4 时第二频是 10000**(-2/4)=0.01，而不是按 head_dim=12 归一的 10000**(-2/12)。
    query = _unit_rows(1, 4, 12, dtype=torch.float64)
    args = (query, torch.tensor([2, 2]), 0, (4, 4, 4), 10000.0)
    rotated, freqs_cos, freqs_sin, ids, angles, inv_freqs = _verify(solve, args)
    assert inv_freqs.tolist() == [1.0, 0.01, 1.0, 0.01, 1.0, 0.01], inv_freqs.tolist()
    assert abs(inv_freqs[1].item() - 10000.0 ** (-2.0 / 12.0)) > 1e-3
    assert angles[3].tolist() == [0.0, 0.0, 1.0, 0.01, 1.0, 0.01], angles[3].tolist()
    # 同一 head_dim 换一种切分，频率必须随之改变
    other = _verify(solve, (query, torch.tensor([2, 2]), 0, (2, 4, 6), 10000.0))[5]
    assert other.tolist() == [1.0, 1.0, 0.01, 1.0,
                              0.046415888336127795, 0.0021544346900318843], other.tolist()


def check_repeat_interleave_duplication(solve):
    args = _case(b=2, text_len=3, height=4, width=5, axes_dim=(16, 56, 56), theta=10000.0,
                 seed=23)
    _rotated, freqs_cos, freqs_sin, _ids, angles = _verify(solve, args)[:5]
    assert torch.equal(freqs_cos[:, 0::2], freqs_cos[:, 1::2]), "cos[2m] 必须等于 cos[2m+1]"
    assert torch.equal(freqs_sin[:, 0::2], freqs_sin[:, 1::2]), "sin[2m] 必须等于 sin[2m+1]"
    assert torch.equal(freqs_cos[:, 0::2], angles.cos().float())
    assert torch.equal(freqs_sin[:, 0::2], angles.sin().float())
    assert freqs_cos.shape[-1] == 2 * angles.shape[-1]
    assert torch.count_nonzero(angles[8:]) > 0, "非零角度才能证明复制是真的"


def check_query_dtype_policy(solve):
    base = _case(b=2, text_len=3, height=2, width=3, axes_dim=(2, 8, 14), theta=250.0,
                 dtype=torch.float64, seed=31)
    query64, grid, text_len, axes_dim, theta = base
    result64 = _verify(solve, base)
    assert result64[0].dtype == torch.float64, "float64 query 的 rotated 须是 float64"
    query32 = query64.float()
    result32 = _verify(solve, (query32, grid, text_len, axes_dim, theta))
    assert result32[0].dtype == torch.float32
    for index in (1, 2, 3, 4):
        assert result64[index].dtype == result32[index].dtype, "表 dtype 与 query dtype 无关"
    assert result64[1].dtype == torch.float32 and result64[4].dtype == torch.float64
    assert result64[5].dtype == torch.float64
    # float32 逐位落表，float64 只是把同一张 float32 表提升精度后相乘
    assert torch.equal(result64[1], result32[1]) and torch.equal(result64[2], result32[2])
    assert torch.equal(result64[3], result32[3])
    assert torch.allclose(result64[0], result32[0].double(), rtol=1e-5, atol=1e-5)


def check_noncontiguous_input(solve):
    for dtype in (torch.float32, torch.float64):
        args = list(_case(b=2, text_len=2, height=2, width=3, axes_dim=(4, 8, 4),
                          theta=10000.0, dtype=dtype, seed=47))
        query = args[0]
        contiguous = query.clone()
        args[0] = torch.stack((contiguous, torch.zeros_like(contiguous)), dim=-1)[..., 0]
        assert not args[0].is_contiguous()
        wide = torch.zeros(query.shape[0], query.shape[1] + 1, query.shape[2], dtype=dtype)
        wide[:, :query.shape[1]] = contiguous
        assert not wide[:, :query.shape[1]].is_contiguous()
        first = _verify(solve, args)
        second = _verify(solve, (wide[:, :query.shape[1]], *args[1:]))
        _close(first[0], second[0])
        _close(first[1], second[1])
        _close(first[0], _oracle(contiguous, *args[1:])[0])


def check_no_input_mutation(solve):
    for grid in (torch.tensor([2, 3]), [2, 3], (2, 3)):
        args = list(_case(b=2, text_len=2, height=2, width=3, axes_dim=(2, 4, 4),
                          dtype=torch.float32, seed=61))
        args[1] = grid
        copies = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
        _verify(solve, args)
        for arg, copy in zip(args, copies):
            if isinstance(arg, torch.Tensor):
                assert torch.equal(arg, copy), "输入被修改"
        assert args[2] == copies[2] and args[3] == copies[3] and args[4] == copies[4]


def check_invalid_axes_dim(solve):
    query, grid, text_len = _unit_rows(1, 5, 128), torch.tensor([2, 2]), 1
    bad_values = [(3, 56, 56), (16, 0, 56), (16, -2, 56), (16, 56, 55), (True, 56, 56),
                  (16, 56.0, 56), (16, 56, torch.tensor(3)), (16, 56), (16, 56, 56, 2),
                  (16, 56, None)]
    for axes_dim in bad_values:
        _expect_value_error(solve, (query, grid, text_len, axes_dim, 10000.0),
                            f"axes_dim={axes_dim!r}")


def check_invalid_scalar_and_shape_args(solve):
    head_dim = 128
    good = (_unit_rows(1, 5, head_dim), torch.tensor([2, 2]), 1, (16, 56, 56), 10000.0)
    for index, values in (
        (4, [0.5, 0.9999, -1.0, 0.0, True, "10000", float("nan"), float("inf"),
             torch.tensor(10000.0)]),
        (2, [-1, True, False, 2.0, torch.tensor(2), None, "2"]),
        (1, [torch.tensor([0, 2]), torch.tensor([2, -1]), torch.tensor([2.5, 2]),
             torch.tensor([True, True]), torch.tensor([2]), [1, 2, 3], 5]),
    ):
        for value in values:
            args = list(good)
            args[index] = value
            _expect_value_error(solve, args, f"arg[{index}]={value!r}")
    # 形状类
    for query, label in (
        (torch.randn(5, head_dim), "query 二维"),
        (torch.randn(1, 2, 5, head_dim), "query 四维"),
        (torch.randn(1, 5, 64), "末维不等于 sum(axes_dim)"),
        (torch.randn(1, 6, head_dim), "S 与 text_len + h*w 不一致"),
        (torch.randn(1, 5), "query 一维"),
    ):
        _expect_value_error(solve, (query, good[1], good[2], good[3], good[4]), label)
