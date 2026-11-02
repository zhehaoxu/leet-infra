import math

import torch

SPECS = {"e3m2": (3, 2, 3), "e2m3": (2, 3, 1)}


def _limit(fmt):
    ebits, mbits, bias = SPECS[fmt]
    return (1.0 + (2 ** mbits - 1) / float(2 ** mbits)) * float(2 ** ((2 ** ebits - 1) - bias))


def _decode(fmt, code):
    ebits, mbits, bias = SPECS[fmt]
    neg = bool(code & 32)
    e = (code >> mbits) & (2 ** ebits - 1)
    m = code & (2 ** mbits - 1)
    mag = m / float(2 ** mbits) * float(2 ** (1 - bias)) if e == 0 else \
        (1.0 + m / float(2 ** mbits)) * float(2 ** (e - bias))
    return -mag if neg else mag


def _table(fmt):
    return [(_decode(fmt, c), c) for c in range(64)]


def _oracle(fmt, x):
    """闭式实现：先定指数，再对尾数做 round-half-even（不复用解答的暴力最近邻）。"""
    ebits, mbits, bias = SPECS[fmt]
    limit = _limit(fmt)
    neg = math.copysign(1.0, x) < 0.0
    mag = abs(x)
    max_e = 2 ** ebits - 1
    max_code = (max_e << mbits) | (2 ** mbits - 1)
    if mag > limit:
        return (max_code | (32 if neg else 0)), (-limit if neg else limit), True
    if mag == 0.0:
        return (32 if neg else 0), (-0.0 if neg else 0.0), False
    exact = [c for val, c in _table(fmt) if val == x]
    if exact:
        # 精确命中时取码值最小者（E3M2 的 0.25 有两个编码）
        code = min(exact)
        return code, _decode(fmt, code), False
    e = int(math.floor(math.log2(mag))) + bias
    if e < 1:
        m = int(round(mag / float(2 ** (1 - bias)) * (2 ** mbits)))
        if m >= 2 ** mbits:
            code = 1 << mbits
        else:
            code = m
    else:
        e = min(e, max_e)
        m = int(round((mag / float(2 ** (e - bias)) - 1.0) * (2 ** mbits)))
        if m >= 2 ** mbits:
            e, m = e + 1, 0
        if e > max_e:
            return (max_code | (32 if neg else 0)), (-limit if neg else limit), True
        code = (e << mbits) | m
    if neg:
        code |= 32
    return code, _decode(fmt, code), False


def check_format_selection(solve):
    # auto：≤7.5 用 E2M3（精度优先），(7.5, 28] 用 E3M2（范围优先）
    assert solve(torch.tensor([7.5], dtype=torch.float64), "auto")["fmt"] == "e2m3"
    assert solve(torch.tensor([7.5000001], dtype=torch.float64), "auto")["fmt"] == "e3m2"
    assert solve(torch.tensor([-28.0], dtype=torch.float64), "auto")["fmt"] == "e3m2"
    assert solve(torch.tensor([0.0], dtype=torch.float64), "auto")["fmt"] == "e2m3"
    try:
        solve(torch.tensor([28.0000001], dtype=torch.float64), "auto")
    except ValueError:
        pass
    else:
        raise AssertionError("amax 超 28 应拒绝")


def check_e3m2_extremes(solve):
    v = torch.tensor([28.0, 0.0625, 0.25, -28.0], dtype=torch.float64)
    out = solve(v, "e3m2")
    assert out["codes"].tolist() == [31, 1, 4, 63], out["codes"].tolist()
    assert out["dequant"].tolist() == [28.0, 0.0625, 0.25, -28.0], out["dequant"].tolist()
    assert not bool(out["sat_mask"].any())


def check_e2m3_extremes(solve):
    v = torch.tensor([7.5, 0.125, 1.875, -7.5], dtype=torch.float64)
    out = solve(v, "e2m3")
    assert out["codes"].tolist() == [31, 1, 15, 63], out["codes"].tolist()
    assert out["dequant"].tolist() == [7.5, 0.125, 1.875, -7.5], out["dequant"].tolist()


def check_subnormal_top_vs_normal_floor(solve):
    # E3M2 次正规顶格 0.1875（码 3）与最小正规 0.25（码 4）首尾相接、不重叠
    out = solve(torch.tensor([0.1875, 0.25, -0.25], dtype=torch.float64), "e3m2")
    assert out["codes"].tolist() == [3, 4, 36], out["codes"].tolist()
    assert out["dequant"].tolist() == [0.1875, 0.25, -0.25], out["dequant"].tolist()
    assert not bool(out["tie_mask"].any()), out["tie_mask"]


def check_rtne_per_format(solve):
    # 平局取尾数位 0：E3M2 0.09375 落在 0.0625(码1) 与 0.125(码2) 之间 → 0.125
    out = solve(torch.tensor([0.09375], dtype=torch.float64), "e3m2")
    assert out["codes"].tolist() == [2], out["codes"].tolist()
    assert bool(out["tie_mask"].all())
    # E2M3 0.0625 落在 0 与 0.125 之间 → 0
    out2 = solve(torch.tensor([0.0625], dtype=torch.float64), "e2m3")
    assert out2["codes"].tolist() == [0], out2["codes"].tolist()
    assert out2["dequant"].tolist() == [0.0]
    assert bool(out2["tie_mask"].all())
    # 非平局点不置位
    out3 = solve(torch.tensor([0.1, 1.1], dtype=torch.float64), "e2m3")
    assert not bool(out3["tie_mask"].any()), out3["tie_mask"]


def check_saturation_explicit(solve):
    v = torch.tensor([8.0, 29.0, -30.0], dtype=torch.float64)
    out = solve(v, "e2m3")
    assert out["codes"].tolist() == [31, 31, 63], out["codes"].tolist()
    assert out["dequant"].tolist() == [7.5, 7.5, -7.5], out["dequant"].tolist()
    assert out["sat_mask"].tolist() == [True, True, True]
    out2 = solve(v, "e3m2")
    assert out2["sat_mask"].tolist() == [False, True, True], out2["sat_mask"].tolist()
    assert out2["dequant"][1].item() == 28.0


def check_negative_zero(solve):
    out = solve(torch.tensor([-0.0, 0.0], dtype=torch.float64), "e3m2")
    assert out["codes"].tolist() == [32, 0], out["codes"].tolist()
    assert math.copysign(1.0, out["dequant"][0].item()) == -1.0


def check_pack_bytes(solve):
    # e3m2 的 0 / 0.25 / 0.5 / 0.75 → 码 0,4,8,10 → 三字节 0x00, 0x81, 0x28
    out = solve(torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64), "e3m2")
    assert out["codes"].tolist() == [0, 4, 8, 10], out["codes"].tolist()
    assert out["packed"].tolist() == [0x00, 0x81, 0x28], out["packed"].tolist()
    # 同一组码在不同格式下字节相同（打包与格式无关）
    codes = [5, 17, 34, 63]
    v = torch.tensor([_decode("e3m2", c) for c in codes], dtype=torch.float64)
    got = solve(v, "e3m2")["packed"].tolist()
    got2 = solve(torch.tensor([_decode("e2m3", c) for c in codes], dtype=torch.float64), "e2m3")["packed"].tolist()
    assert got == got2, (got, got2)


def check_pack_padding(solve):
    for n in range(1, 9):
        v = torch.tensor([_decode("e2m3", (i * 7) % 64) for i in range(n)], dtype=torch.float64)
        out = solve(v, "e2m3")
        assert out["pad_count"] == (-n) % 4, (n, out["pad_count"])
        assert out["packed"].numel() == 3 * ((n + 3) // 4), (n, out["packed"].numel())
        assert out["unpacked"].tolist() == out["codes"].tolist(), n
        # 独立解包：把 3 字节当小端 24 位整数，按 6 位切
        packed = out["packed"].tolist()
        flat = []
        for i in range(0, len(packed), 3):
            bits = packed[i] | (packed[i + 1] << 8) | (packed[i + 2] << 16)
            flat.extend((bits >> (6 * k)) & 0x3F for k in range(4))
        assert flat[:n] == out["codes"].tolist(), (n, flat[:n], out["codes"].tolist())


def check_validation(solve):
    for bad in ("E3M2", "fp6", ""):
        try:
            solve(torch.tensor([1.0], dtype=torch.float64), bad)
        except ValueError:
            continue
        raise AssertionError(f"fmt={bad!r} 应拒绝")
    try:
        solve(torch.tensor([float("nan")], dtype=torch.float64), "e2m3")
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 应拒绝")


def check_empty_input(solve):
    out = solve(torch.tensor([], dtype=torch.float64), "auto")
    for key in ("codes", "unpacked", "packed", "dequant", "sat_mask", "tie_mask"):
        assert out[key].numel() == 0, (key, out[key])
    assert out["pad_count"] == 0


def check_no_alias(solve):
    p = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = solve(p, "e2m3")
    assert p.tolist() == [1.0, 2.0], "入参不应被改"
    out["dequant"][0] = 123.0
    out["codes"][1] = 63
    assert out["unpacked"][1].item() != 63, "unpacked 不应与 codes 共享存储"
    assert out["packed"].numel() == 3


def check_against_oracle(solve):
    torch.manual_seed(20261103)
    for fmt in ("e3m2", "e2m3"):
        grid = sorted({val for val, _c in _table(fmt)})
        mids = [(grid[i] + grid[i + 1]) / 2.0 for i in range(len(grid) - 1)]
        cases = list(grid) + mids + [0.0, -0.0, _limit(fmt), _limit(fmt) * 1.01]
        for trial in range(60):
            n = int(torch.randint(1, 25, (1,)).item())
            src = []
            for _ in range(n):
                pick = cases[int(torch.randint(0, len(cases), (1,)).item())]
                if trial % 3 == 2:
                    pick = float(torch.randn(1, dtype=torch.float64).item()) * 4.0
                src.append(-pick if trial % 2 else pick)
            v = torch.tensor(src, dtype=torch.float64)
            out = solve(v, fmt)
            ref = [_oracle(fmt, x) for x in src]
            assert out["codes"].tolist() == [r[0] for r in ref], (fmt, trial, out["codes"].tolist(), [r[0] for r in ref])
            assert torch.equal(out["dequant"], torch.tensor([r[1] for r in ref], dtype=torch.float64)), (fmt, trial)
            assert out["sat_mask"].tolist() == [r[2] for r in ref], (fmt, trial)
            assert bool((out["codes"] < 64).all())


check_format_selection.example = True
check_pack_bytes.example = True
