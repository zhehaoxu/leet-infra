"""Independent pure-Python oracle (scalar lists, math.log/ldexp) plus targeted checks."""

import math

import torch

KEYS = ("group_shape", "planes", "alpha", "alpha_apot", "apot_terms", "recon", "mse")

LN2 = math.log(2.0)


def _sign(v):
    return 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)


def _round_half_even(v):
    """torch.round 的口径：整数并列时落到偶数（-2.5 -> -2，-1.5 -> -2，2.5 -> 2）。"""
    floor = math.floor(v)
    frac = v - floor
    if frac > 0.5:
        return floor + 1
    if frac < 0.5:
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _round_pow2(v):
    """get_shift_and_sign + 2**shift * sign；v == 0 时 log(0) = -inf、2**-inf = 0、sign(0) = 0。"""
    if v == 0.0:
        return 0.0
    exponent = _round_half_even(math.log(abs(v)) / LN2)
    return math.ldexp(1.0, int(exponent)) * _sign(v)


def _apot(a, nums):
    """additive_power_of_2：每一项都取"原标度减去已提交和"的最近幂，返回累加和与各项。"""
    terms = []
    total = 0.0
    for _ in range(nums):
        term = _round_pow2(a - total)
        terms.append(term)
        total = total + term
    return total, terms


def _oracle(weight, num_bits, group_size=-1, apot_nums=1):
    """纯 Python 标量实现：逐组、逐元素手算位平面与 ApOT，完全不碰张量归约。"""
    rows = len(weight)
    cols = len(weight[0])
    flat = [float(v) for row in weight for v in row]
    total = rows * cols
    g = cols if group_size <= 0 else group_size
    if total % g != 0:
        raise ValueError("group_size must divide the flattened element count")
    n = total // g
    planes, alpha, alpha_apot, apot_terms, rec_flat = [], [], [], [], []
    for gi in range(n):
        r = list(flat[gi * g:(gi + 1) * g])
        p_grp, a_grp = [], []
        for _ in range(num_bits):
            b = [_sign(v) for v in r]
            a = sum(abs(v) for v in r) / g
            a_grp.append(a)
            for j in range(g):
                r[j] = r[j] - b[j] * a
            p_grp.append(b)
        planes.append(p_grp)
        alpha.append(a_grp)
        apot_grp, terms_grp = [], []
        for a in a_grp:
            t_sum, t_list = _apot(a, apot_nums)
            apot_grp.append(t_sum)
            terms_grp.append(t_list)
        alpha_apot.append(apot_grp)
        apot_terms.append(terms_grp)
        for j in range(g):
            v = 0.0
            for i in range(num_bits):
                v = v + planes[gi][i][j] * apot_grp[i]
            rec_flat.append(v)
    recon = [rec_flat[k * cols:(k + 1) * cols] for k in range(rows)]
    mse = sum((flat[i] - rec_flat[i]) ** 2 for i in range(total)) / total
    return {
        "group_shape": (n, g),
        "planes": planes,
        "alpha": alpha,
        "alpha_apot": alpha_apot,
        "apot_terms": apot_terms,
        "recon": recon,
        "mse": mse,
    }


def _to_tensors(want, dtype):
    return {
        "group_shape": want["group_shape"],
        "planes": torch.tensor(want["planes"], dtype=dtype).transpose(1, 2),
        "alpha": torch.tensor(want["alpha"], dtype=dtype),
        "alpha_apot": torch.tensor(want["alpha_apot"], dtype=dtype),
        "apot_terms": torch.tensor(want["apot_terms"], dtype=dtype),
        "recon": torch.tensor(want["recon"], dtype=dtype),
        "mse": torch.tensor(want["mse"], dtype=dtype),
    }


def _close(a, b):
    tol = 3e-5 if a.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _w(rows, dtype=torch.float64):
    return torch.tensor(rows, dtype=dtype)


def _example():
    return (_w([[1.0, 1.0, -3.0, 5.0], [2.0, -1.0, 0.0, 1.0]]), 2, -1, 2)


def _example_two():
    return (_w([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]), 1, 2, 1)


def _check_structure(solve, args):
    weight, num_bits, group_size, apot_nums = args
    before = weight.clone()
    got = solve(*args)
    assert isinstance(got, dict) and set(got) == set(KEYS), f"键不对：{sorted(got)}"
    assert torch.equal(weight, before), "输入被就地修改了"
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    g = cols if group_size <= 0 else group_size
    n = rows * cols // g
    assert tuple(got["group_shape"]) == (n, g), f"group_shape {got['group_shape']} != {(n, g)}"
    assert tuple(got["planes"].shape) == (n, g, num_bits), got["planes"].shape
    assert tuple(got["alpha"].shape) == (n, num_bits), got["alpha"].shape
    assert tuple(got["alpha_apot"].shape) == (n, num_bits), got["alpha_apot"].shape
    assert tuple(got["apot_terms"].shape) == (n, num_bits, apot_nums), got["apot_terms"].shape
    assert tuple(got["recon"].shape) == tuple(weight.shape), got["recon"].shape
    assert got["mse"].dim() == 0
    assert torch.equal(got["planes"].abs(), (got["planes"] != 0).to(got["planes"].dtype)), \
        "planes 的元素必须只取 -1/0/+1"
    for key in ("planes", "alpha", "alpha_apot", "apot_terms", "recon", "mse"):
        assert got[key].dtype == weight.dtype, f"{key} dtype {got[key].dtype} != {weight.dtype}"
        assert torch.isfinite(got[key]).all(), f"{key} 出现非有限值"
    return got


def _verify(solve, args):
    got = _check_structure(solve, args)
    weight, num_bits, group_size, apot_nums = args
    want = _to_tensors(_oracle(weight.tolist(), num_bits, group_size, apot_nums), weight.dtype)
    assert tuple(got["group_shape"]) == want["group_shape"]
    for key in ("planes", "alpha", "alpha_apot", "apot_terms", "recon", "mse"):
        _close(got[key], want[key])
    return got


def check_example(solve):
    """题面示例一：2x4、整行一组、num_bits=2、apot_nums=2 的全部七个键。"""
    got = _verify(solve, _example())
    assert tuple(got["group_shape"]) == (2, 4)
    _close(got["planes"], torch.tensor([[[1, -1], [1, -1], [-1, -1], [1, 1]],
                                        [[1, 1], [-1, 0], [0, 0], [1, 0]]], dtype=torch.float64))
    _close(got["alpha"], torch.tensor([[2.5, 1.5], [1.0, 0.25]], dtype=torch.float64))
    _close(got["alpha_apot"], torch.tensor([[2.5, 1.5], [1.0, 0.25]], dtype=torch.float64))
    _close(got["apot_terms"], torch.tensor([[[2.0, 0.5], [2.0, -0.5]],
                                            [[1.0, 0.0], [0.25, 0.0]]], dtype=torch.float64))
    _close(got["recon"], torch.tensor([[1.0, 1.0, -4.0, 4.0],
                                       [1.25, -1.0, 0.0, 1.0]], dtype=torch.float64))
    assert abs(float(got["mse"]) - 0.3203125) < 1e-12


check_example.example = True


def check_example_straddling(solve):
    """题面示例二：2x3、group_size=2，组跨行；alpha=[1.5,6,24] 的最近幂是 2/8/32。"""
    got = _verify(solve, _example_two())
    assert tuple(got["group_shape"]) == (3, 2)
    _close(got["alpha"], torch.tensor([[1.5], [6.0], [24.0]], dtype=torch.float64))
    _close(got["alpha_apot"], torch.tensor([[2.0], [8.0], [32.0]], dtype=torch.float64))
    _close(got["recon"], torch.tensor([[2.0, 2.0, 8.0], [8.0, 32.0, 32.0]], dtype=torch.float64))
    assert abs(float(got["mse"]) - 45.5) < 1e-12


check_example_straddling.example = True


def check_nearest_power_not_floor(solve):
    """取最近幂而不是向下取幂：mean|r| = 3 -> 4（不是 2），= 0.4 -> 0.5（不是 0.25）。"""
    got = solve(_w([[3.0, 3.0, 3.0, 3.0]]), 1, -1, 1)
    _close(got["alpha"], torch.tensor([[3.0]], dtype=torch.float64))
    _close(got["alpha_apot"], torch.tensor([[4.0]], dtype=torch.float64))
    got = solve(_w([[0.4, 0.4, 0.4, 0.4]]), 1, -1, 1)
    _close(got["alpha_apot"], torch.tensor([[0.5]], dtype=torch.float64))
    _close(got["recon"], torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64))
    # 两个 1x1 的组：各自独立取最近幂，互不影响
    got = solve(_w([[3.0], [0.4]]), 1, -1, 1)
    _close(got["recon"], torch.tensor([[4.0], [0.5]], dtype=torch.float64))


def check_apot_residual_is_signed(solve):
    """ApOT 第 k 项吃的是残差：1.5 -> [2, -0.5]，2.5 -> [2, +0.5]，1.625 -> [2, -0.5, +0.125]。"""
    got = solve(_w([[1.5, 1.5, 1.5, 1.5]]), 1, -1, 2)
    _close(got["apot_terms"], torch.tensor([[[2.0, -0.5]]], dtype=torch.float64))
    _close(got["alpha_apot"], torch.tensor([[1.5]], dtype=torch.float64))
    got = solve(_w([[2.5, 2.5, 2.5, 2.5]]), 1, -1, 2)
    _close(got["apot_terms"], torch.tensor([[[2.0, 0.5]]], dtype=torch.float64))
    got = solve(_w([[1.625] * 4]), 1, -1, 3)
    _close(got["apot_terms"], torch.tensor([[[2.0, -0.5, 0.125]]], dtype=torch.float64))
    _close(got["alpha_apot"], torch.tensor([[1.625]], dtype=torch.float64))


def check_zero_plane_is_absorbing(solve):
    """残差落到 0 之后 sign(0)=0 让它之后所有平面都是 0，且全零标度的 ApOT 项也是 0。"""
    got = solve(_w([[1.0, 2.0, 3.0, 4.0]]), 4, -1, 1)
    _close(got["alpha"], torch.tensor([[2.5, 1.0, 0.5, 0.0]], dtype=torch.float64))
    _close(got["planes"][0][2], torch.tensor([1.0, 1.0, -1.0, 0.0], dtype=torch.float64))
    _close(got["planes"][0][3], torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64))
    _close(got["alpha_apot"][0][3], torch.tensor(0.0, dtype=torch.float64))
    _close(got["apot_terms"][0][3], torch.tensor([0.0], dtype=torch.float64))


def check_zero_weight_group(solve):
    """全零权重：planes/alpha/apot 全 0、recon 全 0、mse 为 0（没有 2 的幂也没有 NaN）。"""
    for bits, ap in ((1, 1), (3, 2), (2, 4)):
        got = solve(torch.zeros(2, 4, dtype=torch.float64), bits, -1, ap)
        assert torch.equal(got["planes"], torch.zeros(2, 4, bits, dtype=torch.float64))
        assert torch.equal(got["alpha"], torch.zeros(2, bits, dtype=torch.float64))
        assert torch.equal(got["alpha_apot"], torch.zeros(2, bits, dtype=torch.float64))
        assert torch.equal(got["apot_terms"], torch.zeros(2, bits, ap, dtype=torch.float64))
        assert torch.equal(got["recon"], torch.zeros(2, 4, dtype=torch.float64))
        assert float(got["mse"]) == 0.0


def check_first_plane_is_sign(solve):
    """第 0 个平面就是权重的符号（按展平顺序），与 group_size 无关。"""
    generator = torch.Generator().manual_seed(20261030)
    w = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    for gs in (-1, 6, 4, 3, 2):
        got = solve(w, 3, gs, 2)
        n, g = got["group_shape"]
        _close(got["planes"][:, :, 0], torch.sign(w).reshape(n, g))


def check_group_size_whole_row(solve):
    """group_size 的任何非正值都退化成整行一组，等价于 group_size = C。"""
    w = _w([[1.0, 2.0, -4.0, 8.0], [-1.0, 3.0, 0.0, -2.0]])
    base = solve(w, 2, -1, 2)
    for gs in (w.shape[1], 0, -1, -7):
        other = solve(w, 2, gs, 2)
        assert other["group_shape"] == base["group_shape"] == (2, 4)
        for key in ("planes", "alpha", "alpha_apot", "apot_terms", "recon", "mse"):
            _close(other[key], base[key])


def check_recon_uses_apot_scale(solve):
    """planes/alpha 只由贪心决定，recon/mse 只由 alpha_apot 决定；两者混用会只错后两个键。"""
    w = _w([[0.7, -1.3, 2.2, -0.4]])
    got = solve(w, 2, -1, 1)
    assert not torch.allclose(got["alpha"], got["alpha_apot"]), "这组数据应当让最近幂偏离原标度"
    manual = torch.einsum("ijl,il->ij", got["planes"], got["alpha_apot"])
    _close(got["recon"], manual.reshape(1, 4))
    _close(got["mse"], ((w - got["recon"]) ** 2).mean())
    # 用 alpha 而不是 alpha_apot 重建会得到不同的 mse
    wrong = torch.einsum("ijl,il->ij", got["planes"], got["alpha"]).reshape(1, 4)
    assert not torch.allclose(wrong, got["recon"])


def check_more_apot_terms_help(solve):
    """apot_nums 从 1 涨到 4：|alpha_apot - alpha| 单调不增，且 planes/alpha 逐位不变。"""
    w = _w([[0.7, -1.3, 2.2, -0.4, 1.1, -3.9]])
    base = solve(w, 3, -1, 1)
    first = None
    prev = None
    for ap in (1, 2, 3, 4):
        got = solve(w, 3, -1, ap)
        _close(got["planes"], base["planes"])
        _close(got["alpha"], base["alpha"])
        dist = float((got["alpha_apot"] - got["alpha"]).abs().max())
        if first is None:
            first = dist
        if prev is not None:
            assert dist <= prev + 1e-12, f"apot_nums={ap} 反而更差：{dist} > {prev}"
        prev = dist
    assert prev <= 0.05 * first + 1e-12, f"4 项之后仍有明显残差：{prev} vs {first}"


def check_mse_against_explicit_numbers(solve):
    """num_bits=1 时逐元素闭合：recon = sign(w) * apot(mean|w|)，mse 手算。"""
    w = _w([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
    got = solve(w, 1, 2, 1)
    _close(got["recon"], torch.tensor([[2.0, 2.0, 8.0], [8.0, 32.0, 32.0]],
                                     dtype=torch.float64))
    assert abs(float(got["mse"]) - float(((w - got["recon"]) ** 2).sum()) / 6.0) < 1e-12


def check_dtype_sweep(solve):
    """float32 输入按 float32 返回，且数值与 float64 路径一致（内部一律 float64）。"""
    w64 = _w([[0.7, -1.3, 2.2, -0.4], [1.1, -3.9, 0.0, 5.0]])
    hi = solve(w64, 3, -1, 2)
    lo = solve(w64.float(), 3, -1, 2)
    for key in ("planes", "alpha", "alpha_apot", "apot_terms", "recon", "mse"):
        assert lo[key].dtype == torch.float32, f"{key} 没有按输入 dtype 返回"
        torch.testing.assert_close(lo[key], hi[key].float(), rtol=1e-5, atol=1e-5)


def check_non_contiguous_matches_copy(solve):
    """非连续输入按行主序读取：转置视图与 contiguous 副本结果逐位相同。"""
    base = _w([[1.0, 2.0, -4.0, 8.0], [-1.0, 3.0, 0.0, -2.0], [5.0, -6.0, 7.0, 1.0]])
    view = base.t()  # (4, 3)，非连续
    got = solve(view, 2, 2, 2)
    want = solve(view.contiguous(), 2, 2, 2)
    for key in ("planes", "alpha", "alpha_apot", "apot_terms", "recon", "mse"):
        _close(got[key], want[key])
    assert tuple(got["group_shape"]) == tuple(want["group_shape"]) == (6, 2)


def check_inputs_not_mutated(solve):
    w = _w([[1.0, 2.0, -4.0, 8.0], [-1.0, 3.0, 0.0, -2.0]])
    before = w.clone()
    solve(w, 2, -1, 2)
    assert torch.equal(w, before)


def check_value_errors(solve):
    w = _w([[1.0, 2.0, -4.0, 8.0]])
    bad = [
        ("一维权重", lambda: solve(torch.tensor([1.0, 2.0, 3.0]), 2, -1, 1)),
        ("三维权重", lambda: solve(torch.randn(2, 2, 2, dtype=torch.float64), 2, -1, 1)),
        ("空张量", lambda: solve(torch.zeros(0, 4, dtype=torch.float64), 2, -1, 1)),
        ("整型 dtype", lambda: solve(torch.tensor([[1, 2], [3, 4]]), 2, -1, 1)),
        ("num_bits=0", lambda: solve(w, 0, -1, 1)),
        ("apot_nums=0", lambda: solve(w, 2, -1, 0)),
        ("group_size 不整除展平元素数", lambda: solve(w, 2, 3, 1)),
        ("num_bits 非 int", lambda: solve(w, 2.0, -1, 1)),
        ("group_size 非 int", lambda: solve(w, 2, "2", 1)),
        ("apot_nums 是 bool", lambda: solve(w, 2, -1, True)),
    ]
    for tag, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{tag} 抛了 {type(exc).__name__} 而不是 ValueError") from exc
        raise AssertionError(f"{tag} 没有抛 ValueError")
    # 合法边界：一组吃下整个矩阵、分组粒度等于列数
    solve(w, 1, 4, 1)
    solve(w, 2, 1, 3)
    solve(w, 1, -5, 1)


def check_random_against_oracle(solve):
    """60 组随机实例（多形状、多分组、多位平面/项数）与纯 Python oracle 一致。"""
    generator = torch.Generator().manual_seed(20261031)
    for case in range(60):
        rows = int(torch.randint(1, 5, (1,), generator=generator))
        cols = int(torch.randint(1, 7, (1,), generator=generator))
        numel = rows * cols
        divisors = [d for d in range(1, numel + 1) if numel % d == 0]
        g = int(torch.tensor(divisors, dtype=torch.int64)[
            int(torch.randint(len(divisors), (1,), generator=generator))])
        bits = int(torch.randint(1, 5, (1,), generator=generator))
        ap = int(torch.randint(1, 4, (1,), generator=generator))
        w = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
        if case % 3 == 0:
            w = w * torch.randint(0, 2, w.shape, generator=generator).double()
        if case % 5 == 0:
            w = torch.round(w)
        if case % 7 == 0:
            w = w * 0.5 ** torch.randint(-3, 4, w.shape, generator=generator).double()
        if case % 4 == 0:
            w = w.float()
        whole_row = case % 6 == 0
        args = (w, bits, -g if whole_row else g, ap)
        _verify(solve, args)
        eff = cols if whole_row else g
        assert tuple(solve(*args)["group_shape"]) == (numel // eff, eff)
