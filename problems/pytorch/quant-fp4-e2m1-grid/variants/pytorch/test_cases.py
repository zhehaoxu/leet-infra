import math

import torch


def _decode_code(code):
    """独立参照：按位域公式（1 符号 + 2 指数 + 1 尾数，bias=1）还原 E2M1，不复用解答的网格表。"""
    sign = -1.0 if code & 8 else 1.0
    e = (code >> 1) & 3
    m = code & 1
    mag = m * 0.5 if e == 0 else (1.0 + m / 2.0) * (2.0 ** (e - 1))
    return sign * mag


_REF_TABLE = [(_decode_code(c), c) for c in range(16)]
_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _oracle(scaled):
    """暴力最近邻：绝对误差最小 → 符号一致（区分 ±0）→ 尾数位 0（偶）→ 码值小。"""
    neg = math.copysign(1.0, scaled) < 0.0
    best = None
    for val, code in _REF_TABLE:
        sign_ok = 0 if ((code & 8) != 0) == neg else 1
        key = (abs(scaled - val), sign_ok, code & 1, code)
        if best is None or key < best[0]:
            best = (key, code, val)
    return best[1], best[2]


def check_grid_is_e2m1(solve):
    # 8 个正码逐一回读：0.5 是唯一非零次正规，最大 6 饱和，没有 inf/NaN 码
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    out = solve(vals, 8, [1.0])
    assert out["codes"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7], out["codes"].tolist()
    assert torch.equal(out["dequant"], vals), out["dequant"]
    assert torch.equal(out["grid"], vals), out["grid"]
    assert not bool(out["sat_mask"].any())


def check_rtne_ties_to_even(solve):
    # 平局全表：0.25→0, 0.75→1.0, 1.25→1.0, 1.75→2.0, 2.5→2.0, 3.5→4.0, 5.0→4.0
    v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float64)
    out = solve(v, 7, [1.0])
    assert out["codes"].tolist() == [0, 2, 2, 4, 4, 6, 6], out["codes"].tolist()
    assert out["dequant"].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], out["dequant"].tolist()
    assert bool(out["tie_mask"].all()), out["tie_mask"]
    # 非平局点不得置 tie 位
    out2 = solve(torch.tensor([0.3, 0.9, 1.4], dtype=torch.float64), 3, [1.0])
    assert not bool(out2["tie_mask"].any()), out2["tie_mask"]


def check_only_subnormal_is_half(solve):
    # 0.5 是唯一的非零次正规：0.4 只能到 0.5，0.1 只能回 0（不存在 0.25 这一档）
    out = solve(torch.tensor([0.4, 0.1, 0.26], dtype=torch.float64), 3, [1.0])
    assert out["codes"].tolist() == [1, 0, 1], out["codes"].tolist()
    assert out["dequant"].tolist() == [0.5, 0.0, 0.5], out["dequant"].tolist()


def check_saturation_at_six(solve):
    out = solve(torch.tensor([6.0, 6.0000001, 7.0, -7.0, 100.0], dtype=torch.float64), 5, [1.0])
    assert out["codes"].tolist() == [7, 7, 7, 15, 7], out["codes"].tolist()
    assert out["sat_mask"].tolist() == [False, True, True, True, True], out["sat_mask"].tolist()
    assert out["dequant"].tolist() == [6.0, 6.0, 6.0, -6.0, 6.0], out["dequant"].tolist()
    assert out["block_sat"].tolist() == [4]


def check_inf_saturates_nan_rejected(solve):
    inf = float("inf")
    out = solve(torch.tensor([inf, -inf], dtype=torch.float64), 2, [1.0])
    assert out["codes"].tolist() == [7, 15], out["codes"].tolist()
    assert bool(out["sat_mask"].all())
    try:
        solve(torch.tensor([float("nan")], dtype=torch.float64), 1, [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 应被拒绝（FP4 无 NaN 编码）")


def check_negative_zero_keeps_sign(solve):
    out = solve(torch.tensor([-0.0, 0.0], dtype=torch.float64), 2, [1.0])
    assert out["codes"].tolist() == [8, 0], out["codes"].tolist()
    assert math.copysign(1.0, out["dequant"][0].item()) == -1.0
    assert math.copysign(1.0, out["dequant"][1].item()) == 1.0


def check_two_level_scale(solve):
    # NVFP4 式两级：每张量 global × 每块 block，元素先除乘积再进网格
    v = torch.tensor([1.5, 3.0, 12.0], dtype=torch.float64)
    out = solve(v, 1, [0.5, 2.0, 4.0], global_scale=3.0)
    assert out["eff_scale"].tolist() == [1.5, 6.0, 12.0], out["eff_scale"].tolist()
    assert out["codes"].tolist() == [2, 1, 2], out["codes"].tolist()
    assert out["dequant"].tolist() == [1.5, 3.0, 12.0], out["dequant"].tolist()
    # 同样的 block scale 但 global=1：MX 式单级，元素全落在最大码上
    out2 = solve(v, 1, [0.25, 0.5, 2.0])
    assert out2["eff_scale"].tolist() == [0.25, 0.5, 2.0]
    assert out2["codes"].tolist() == [7, 7, 7], out2["codes"].tolist()
    assert out2["dequant"].tolist() == [1.5, 3.0, 12.0], out2["dequant"]


def check_short_tail_block(solve):
    # 尾块不足 block_size 仍按自己的 scale 记账，平局计入自己块号
    v = torch.tensor([0.75, 0.75, 0.75, 0.75, 3.0], dtype=torch.float64)
    out = solve(v, 4, [1.0, 4.0])
    assert out["codes"].tolist() == [2, 2, 2, 2, 2], out["codes"].tolist()
    assert out["block_ties"].tolist() == [4, 1], out["block_ties"].tolist()
    assert out["dequant"].tolist() == [1.0, 1.0, 1.0, 1.0, 4.0], out["dequant"].tolist()
    assert out["block_sat"].tolist() == [0, 0]


def check_block_wall(solve):
    # 相邻两块同值不同 scale：跨块不许串用 scale
    v = torch.tensor([0.9, 0.9, 0.9, 0.9], dtype=torch.float64)
    out = solve(v, 2, [1.0, 3.0])
    assert out["codes"].tolist() == [2, 2, 1, 1], out["codes"].tolist()
    assert out["dequant"].tolist() == [1.0, 1.0, 1.5, 1.5], out["dequant"].tolist()


def check_validation(solve):
    v = torch.ones(3, dtype=torch.float64)
    bad = [
        dict(block_size=0),
        dict(block_size=-2),
        dict(block_size=True),
        dict(block_scales=[1.0, 1.0]),
        dict(block_scales=[0.0]),
        dict(block_scales=[-1.0]),
        dict(block_scales=[float("nan")]),
        dict(global_scale=0.0),
        dict(global_scale=-1.0),
        dict(global_scale=float("inf")),
        dict(global_scale=float("nan")),
    ]
    for kw in bad:
        args = dict(block_size=4, block_scales=[1.0])
        args.update(kw)
        try:
            solve(v, **args)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {kw}")
    # bool 是 int 的子类：单独验一次（否则会被长度检查先拦下，掩盖这条门）
    try:
        solve(v, True, [1.0, 1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("block_size=True 应被拒绝")


def check_empty_input(solve):
    out = solve(torch.tensor([], dtype=torch.float64), 8, [])
    for key in ("codes", "dequant", "eff_scale", "block_sat", "block_ties"):
        assert out[key].numel() == 0, (key, out[key])
    assert out["grid"].numel() == 8


def check_no_alias(solve):
    p = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = solve(p, 2, [1.0])
    assert p.tolist() == [1.0, 2.0], "入参不应被修改"
    out["dequant"][0] = 99.0
    assert out["codes"][0].item() == 2, "codes 不应与 dequant 共享存储"
    assert out["eff_scale"][0].item() == 1.0, "eff_scale 不应与 dequant 共享存储"


def check_against_bruteforce(solve):
    torch.manual_seed(20261102)
    for block_size in (1, 2, 3, 16, 32):
        for trial in range(12):
            n = int(torch.randint(1, 40, (1,)).item())
            g = [1.0, 2.0, 0.5, 3.0, 0.125][trial % 5]
            nb = (n + block_size - 1) // block_size
            scales = [float(2.0 ** int(torch.randint(-4, 4, (1,)).item())) for _ in range(nb)]
            v = torch.randn(n, dtype=torch.float64) * (2.0 ** int(torch.randint(-3, 4, (1,)).item()))
            out = solve(v, block_size, scales, global_scale=g)
            eff = torch.tensor([g * scales[i // block_size] for i in range(n)], dtype=torch.float64)
            codes_ref, deq_ref, ties_ref = [], [], []
            for s in (v / eff).tolist():
                code, val = _oracle(s)
                codes_ref.append(code)
                deq_ref.append(val)
                ties_ref.append(abs(s) in _MIDS)
            assert out["codes"].tolist() == codes_ref, (block_size, trial, out["codes"].tolist(), codes_ref)
            assert torch.equal(out["eff_scale"], eff)
            assert torch.equal(out["dequant"], torch.tensor(deq_ref, dtype=torch.float64) * eff)
            assert out["tie_mask"].tolist() == ties_ref, (block_size, trial)


check_grid_is_e2m1.example = True
check_rtne_ties_to_even.example = True
