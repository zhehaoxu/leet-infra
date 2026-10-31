import math

import numpy as np
import torch

KEYS = ("phi", "radii", "theta", "rho", "indices", "recon")
TOL = {torch.float32: 1e-5, torch.float64: 1e-10}


def _quant(vals, bits):
    """逐组 min/max + 显式 floor 的标量量化；常数组（scale==0）码字取 0。"""
    mn, mx = min(vals), max(vals)
    scale = (mx - mn) / 2 ** bits
    if scale == 0:
        return [0] * len(vals), scale, mn
    return [min(max(int(math.floor((v - mn) / scale)), 0), 2 ** bits - 1) for v in vals], scale, mn


def _oracle(keys, group_size, rbits, tbits):
    """独立纯 python 重算：(d, Dh+d) 两半配对、负角折回 [0,2π)、bin 中心重构。"""
    x = np.asarray(keys, dtype=np.float64)
    b_n, h_n, l_n, d_n = x.shape
    dh, lb = d_n // 2, l_n // group_size
    phi = np.zeros((b_n, h_n, lb, group_size, dh))
    radii, theta, rho = np.zeros_like(phi), np.zeros_like(phi, np.int64), np.zeros_like(phi, np.int64)
    recon = np.zeros((b_n, h_n, l_n, d_n))
    for b, h, g, d in [(b, h, g, d) for b in range(b_n) for h in range(h_n) for g in range(lb) for d in range(dh)]:
        angs, rr = [], []
        for t in range(group_size):
            a0, a1 = x[b, h, g * group_size + t, d], x[b, h, g * group_size + t, dh + d]
            a = math.atan2(a1, a0)
            angs.append(a + 2 * math.pi if a < 0 else a)
            rr.append(math.hypot(a0, a1))
        tc, ts, tmn = _quant(angs, tbits)
        rc, rs, rmn = _quant(rr, rbits)
        for t in range(group_size):
            l = g * group_size + t
            phi[b, h, g, t, d], radii[b, h, g, t, d] = angs[t], rr[t]
            theta[b, h, g, t, d], rho[b, h, g, t, d] = tc[t], rc[t]
            ang, rad = (tc[t] + 0.5) * ts + tmn, (rc[t] + 0.5) * rs + rmn
            recon[b, h, l, d], recon[b, h, l, dh + d] = rad * math.cos(ang), rad * math.sin(ang)
    return {"phi": phi, "radii": radii, "theta": theta, "rho": rho,
            "indices": rho * 2 ** tbits + theta, "recon": recon}


def _verify(solve, keys, group_size, rbits, tbits):
    before, got = keys.clone(), solve(keys, group_size, rbits, tbits)
    assert isinstance(got, dict) and set(got) == set(KEYS), f"键不对：{sorted(got)}"
    assert torch.equal(keys, before), "输入被就地修改了"
    b_n, h_n, l_n, d_n = keys.shape
    dh, lb = d_n // 2, l_n // group_size
    want = _oracle(keys.numpy(), group_size, rbits, tbits)
    for key in ("phi", "radii", "theta", "rho", "indices"):
        assert got[key].shape == (b_n, h_n, lb, group_size, dh), f"{key} 形状不对"
    assert got["recon"].shape == (b_n, h_n, l_n, d_n), "recon 形状不对"
    for key in ("theta", "rho", "indices"):
        assert got[key].dtype == torch.int64, f"{key} 应为 int64"
        assert torch.equal(got[key], torch.from_numpy(want[key])), f"{key} 与 oracle 不一致"
    for key in ("phi", "radii", "recon"):
        assert got[key].dtype == keys.dtype, f"{key} 的 dtype 应跟随输入"
        torch.testing.assert_close(got[key], torch.tensor(want[key], dtype=keys.dtype),
                                   rtol=TOL[keys.dtype], atol=TOL[keys.dtype], msg=f"{key} 不对")
    return got


def check_example(solve):
    """题面示例：角度 0.1/0.9/1.6/2.0、半径 1/2/3/4，group_size=4、rbits=tbits=2。"""
    ang, rad = [0.1, 0.9, 1.6, 2.0], [1.0, 2.0, 3.0, 4.0]
    keys = torch.tensor([[[[r * math.cos(a), 1.0, r * math.sin(a), 1.0] for a, r in zip(ang, rad)]]], dtype=torch.float64)
    got = _verify(solve, keys, 4, 2, 2)
    ts, tmn, rs, rmn = (2.0 - 0.1) / 4, 0.1, (4.0 - 1.0) / 4, 1.0
    assert got["theta"][0, 0, 0, :, 0].tolist() == [0, 1, 3, 3], "0.8/0.475=1.68 向下取整成 1"
    assert got["rho"][0, 0, 0, :, 0].tolist() == [0, 1, 2, 3]
    assert got["indices"][0, 0, 0, :, 0].tolist() == [0, 5, 11, 15], "(rho<<2)+theta"
    assert got["theta"][0, 0, 0, :, 1].tolist() == [0] * 4 == got["rho"][0, 0, 0, :, 1].tolist()
    assert ang[2] > tmn + 3 * ts and ang[2] < tmn + 4 * ts, "1.6 落在最后一个 bin 里"
    ct, cr, p4 = tmn + 3.5 * ts, rmn + 2.5 * rs, math.atan2(1.0, 1.0)
    torch.testing.assert_close(got["phi"][0, 0, 0, :, 1], torch.full((4,), p4, dtype=torch.float64),
                               rtol=0, atol=1e-12, msg="角度常数组的 φ 就是 π/4")
    exp = torch.tensor([cr * math.cos(ct), 1.0, cr * math.sin(ct), 1.0], dtype=torch.float64)
    torch.testing.assert_close(got["recon"][0, 0, 2], exp, rtol=0, atol=1e-12,
                               msg="重构取 bin 中心：角 1.7625、半径 2.875")


def check_pairing_is_halves_not_interleaved(solve):
    """第 d 对是 (d, D//2+d)：交错配对会把 atan2(5,2) 变成 atan2(2,1)。"""
    keys = torch.tensor([[[[1.0, 2.0, 0.0, 5.0], [1.0, 2.0, 1.0, 5.0]]]], dtype=torch.float64)
    got = _verify(solve, keys, 2, 3, 3)
    torch.testing.assert_close(got["phi"][0, 0, 0, 0], torch.tensor([0.0, math.atan2(5, 2)], dtype=torch.float64))
    torch.testing.assert_close(got["radii"][0, 0, 0, 0], torch.tensor([1.0, math.hypot(2, 5)], dtype=torch.float64))
    exp = torch.tensor([2.0, 5.0], dtype=torch.float64)
    torch.testing.assert_close(got["recon"][0, 0, 0, [1, 3]], exp, rtol=0, atol=1e-12,
                               msg="d=1 那一整组恒定，精确重构回 (2,5)")


def check_negative_angles_wrap_and_change_the_min(solve):
    """-0.1 折成 2π-0.1 后组内最小值变成 0.0 那个角，整个分箱平移。"""
    ang = [3.0, 2 * math.pi - 0.1, 0.0, math.pi]
    row = [[r * math.cos(a), r * math.sin(a)] for a, r in zip(ang, [1.0, 2.0, 3.0, 4.0])]
    got = _verify(solve, torch.tensor([row, row], dtype=torch.float64).reshape(1, 2, 4, 2), 4, 2, 2)
    assert float(got["phi"].min()) >= 0.0 and float(got["phi"].max()) < 2 * math.pi
    torch.testing.assert_close(got["phi"][0, 0, 0, :, 0], torch.tensor(ang, dtype=torch.float64),
                               rtol=0, atol=1e-12, msg="负角已折回 [0, 2π)")
    assert got["theta"][0, 0, 0, :, 0].tolist() == [1, 3, 0, 2], "折回后 0.0 才是组最小"
    assert got["phi"][0, 0, 0, 2, 0] == 0.0, "atan2(0,3)=0 不会被折成 2π"
    assert int(got["rho"][0, 0, 0, 3, 0]) == 3, "半径最大元顶到 2**rbits-1"


def check_max_lands_on_top_code_and_centre_is_inside_bin(solve):
    """分母是 2**bits 而非 2**bits-1：最大元码字恰为 2**bits-1，重构角落在组跨度内。"""
    ang, rad = [0.1, 0.2, 1.6, 2.0], [1.0, 2.0, 3.0, 4.0]
    keys = torch.tensor([[[[r * math.cos(a), r * math.sin(a)] for a, r in zip(ang, rad)]]], dtype=torch.float64)
    for tbits in (2, 3, 5):
        got = _verify(solve, keys, 4, 2, tbits)
        phi, theta = got["phi"][0, 0, 0, :, 0], got["theta"][0, 0, 0, :, 0]
        assert int(theta[int(phi.argmax())]) == 2 ** tbits - 1, "最大角顶到 2**tbits-1"
        assert int(theta[int(phi.argmin())]) == 0
        lo, step = float(phi.min()), (float(phi.max()) - float(phi.min())) / 2 ** tbits
        hat = lo + (2 ** tbits - 0.5) * step
        rec = math.atan2(float(got["recon"][0, 0, 3, 1]), float(got["recon"][0, 0, 3, 0]))
        assert abs(rec - hat) < 1e-12, "重构角是 bin 中心"
        assert float(phi.max()) - step < hat < float(phi.max()), "中心严格在组内、不在 bin 边界外"
        assert abs(float(got["recon"][0, 0, 3].norm()) - 3.625) < 1e-9, "半径也取 bin 中心"


def check_constant_group_reconstructs_exactly(solve):
    """整组相同的对：scale==0 → 码字 0，重构按原角原半径精确回到输入。"""
    tok = [0.7, -2.0, -1.3, 0.4]
    keys = torch.tensor([[[tok] * 4]], dtype=torch.float64)
    got = _verify(solve, keys, 4, 4, 3)
    assert got["theta"][0, 0, 0].tolist() == [[0, 0]] * 4 == got["rho"][0, 0, 0].tolist()
    assert got["indices"][0, 0, 0].tolist() == [[0, 0]] * 4
    torch.testing.assert_close(got["phi"][0, 0, 0, 0], torch.tensor(
        [math.atan2(-1.3, 0.7) + 2 * math.pi, math.atan2(0.4, -2.0)], dtype=torch.float64),
        rtol=0, atol=1e-12)
    torch.testing.assert_close(got["recon"], keys, rtol=0, atol=1e-12, msg="常数组精确重构")


def check_random_grid_and_dtype_against_oracle(solve):
    """多 (B,H,L,D)、多位宽（含 rbits≠tbits）、两种 dtype 的对拍；float32 入 float32 出。"""
    rng = np.random.default_rng(20261030)
    grid = [(2, 3, 8, 6, 4, 4, 4), (1, 2, 12, 4, 4, 3, 5), (3, 1, 8, 8, 8, 5, 2), (1, 2, 16, 2, 16, 1, 1),
            (2, 2, 12, 10, 4, 6, 7), (1, 1, 4, 4, 4, 4, 4)]
    for i, (b_n, h_n, l_n, d_n, gs, rbits, tbits) in enumerate(grid):
        keys = torch.tensor(rng.normal(size=(b_n, h_n, l_n, d_n)) * 3.0, dtype=torch.float64)
        got = _verify(solve, keys.float() if i % 2 else keys, gs, rbits, tbits)
        assert int(got["theta"].max()) <= 2 ** tbits - 1 and int(got["rho"].max()) <= 2 ** rbits - 1
        assert int(got["indices"].max()) < 2 ** (rbits + tbits)
    lo = _verify(solve, keys.float(), 4, 4, 4)
    hi = _verify(solve, keys, 4, 4, 4)
    assert lo["phi"].dtype == torch.float32 and lo["recon"].dtype == torch.float32
    assert lo["theta"].tolist() == hi["theta"].tolist(), "内部 float64，码字与 dtype 无关"
    torch.testing.assert_close(lo["recon"].double(), hi["recon"], rtol=1e-6, atol=1e-4)
    _verify(solve, torch.zeros(1, 2, 4, 4, dtype=torch.float64), 4, 4, 4)


def check_value_errors(solve):
    """维度、dtype、奇偶、整除与参数类型不合法都抛 ValueError。"""
    keys = torch.randn(1, 2, 8, 6, dtype=torch.float64)
    _verify(solve, keys, 4, 4, 4)
    bad = [(keys[0], 4, 4, 4), (keys.to(torch.int64), 4, 4, 4), (keys.to(torch.bfloat16), 4, 4, 4),
           (torch.randn(1, 1, 4, 5, dtype=torch.float64), 4, 4, 4),
           (torch.randn(1, 1, 6, 4, dtype=torch.float64), 4, 4, 4),
           (torch.zeros(1, 1, 0, 4, dtype=torch.float64), 4, 4, 4),
           (keys, 0, 4, 4), (keys, -4, 4, 4), (keys, 4.0, 4, 4), (keys, True, 4, 4), (keys, 3, 4, 4),
           (keys, 4, 0, 4), (keys, 4, 4.0, 4), (keys, 4, True, 4), (keys, 4, 4, 0),
           (keys, 4, 4, -1), (keys, 4, 4, 4.0), (keys, 4, 4, False)]
    for i, (k, gs, rb, tb) in enumerate(bad):
        try:
            solve(k, gs, rb, tb)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 组非法输入没抛 ValueError：{gs},{rb},{tb}")


check_example.example = True
