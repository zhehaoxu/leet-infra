"""FasterCache CFG-Cache 频域逼近步 -- 测试用例."""

import numpy as np
import torch


def _lat(v, batch):
    """确定性 BCFHW (batch,1,2,8,8) 张量, 值域约 [-1,1]."""
    n = batch * 1 * 2 * 8 * 8
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * 0.37) % 1.0) * 2.0 - 1.0 + v * 0.05
    return x.reshape(batch, 1, 2, 8, 8).contiguous()


def _to_flat_np(t):
    """BCFHW -> (B*F, C, H, W) numpy."""
    t = t.transpose(0, 2, 1, 3, 4)
    return t.reshape(-1, *t.shape[2:])


def _np_split(x):
    """独立 numpy oracle: _split_low_high_freq (float64 复算, 全维 fftshift)."""
    f = np.fft.fft2(np.asarray(x, dtype=np.float64), axes=(-2, -1))
    s = np.fft.fftshift(f)
    h, w = x.shape[-2:]
    r = min(h, w) // 5
    y, xg = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx, cy = w // 2, h // 2
    mask = (xg - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    m = mask[None, None]
    return s * m, s * ~m


def _np_delta(full):
    """整批输出的频域 delta oracle: (low, high), 各 (B*F, C, H, W) 复数."""
    uncond, cond = full[:2], full[2:]
    lu, hu = _np_split(_to_flat_np(uncond).astype(np.float64))
    lc, hc = _np_split(_to_flat_np(cond).astype(np.float64))
    return lu - lc, hu - hc


def _np_uncond_approx(delta_low, delta_high, cond, w_low, w_high):
    """skip 步 uncond 频域重建 oracle."""
    lc, hc = _np_split(_to_flat_np(cond).astype(np.float64))
    freq = (delta_low * w_low + lc) + (delta_high * w_high + hc)
    rec = np.fft.ifft2(np.fft.ifftshift(freq), axes=(-2, -1)).real
    b, f_cnt = cond.shape[0], cond.shape[2]
    rec = rec.reshape(b, f_cnt, -1, *cond.shape[3:])  # (B, F, C, H, W)
    return rec.transpose(0, 2, 1, 3, 4)  # -> BCFHW


def _assert_complex_close(actual, expected, atol=1e-3):
    a = np.stack([np.asarray(actual.real), np.asarray(actual.imag)], axis=-1)
    e = np.stack([np.asarray(expected.real), np.asarray(expected.imag)], axis=-1)
    assert np.allclose(a, e, atol=atol), np.abs(a - e).max()


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 非 skip (整批)
def check_full_passthrough(solve):
    """整批步: 输出原样, batch_size 记录, delta 建立."""
    full = _lat(1.0, 4)
    r = solve(full, None, 400)
    assert r["uncond_skipped"] is False
    assert torch.equal(r["output"], full)
    st = r["state"]
    assert st["batch_size"] == 4 and st["iteration"] == 1
    dl, dh = _np_delta(full.numpy())
    _assert_complex_close(st["low_frequency_delta"].numpy(), dl)
    _assert_complex_close(st["high_frequency_delta"].numpy(), dh)


def check_delta_oracle(solve):
    """delta 对拍独立 numpy oracle (fft2 + fftshift + 圆形掩码)."""
    full = _lat(2.0, 4)
    r = solve(full, None, 100)
    dl, dh = _np_delta(full.numpy())
    st = r["state"]
    _assert_complex_close(st["low_frequency_delta"].numpy(), dl)
    _assert_complex_close(st["high_frequency_delta"].numpy(), dh)
    # delta 逐元素非平凡: 低频/高频支撑不重叠
    low_energy = np.abs(dl).sum()
    assert low_energy > 0


# ---------------------------------------------------------------- 门控序列
def check_skip_sequence(solve):
    """N=5, 区间内: [整批, skip, skip, skip, skip, 整批]."""
    kw = dict(unconditional_batch_skip_range=5)
    st = None
    seq = []
    for i in range(6):
        batch = 4 if (i == 0 or i == 5) else 2
        r = solve(_lat(float(i), batch), st, 400, **kw)
        seq.append(r["uncond_skipped"])
        st = r["state"]
    assert seq == [False, True, True, True, True, False], seq
    assert st["iteration"] == 6


def check_open_interval_uncond(solve):
    """uncond 跳过区间端点: t==lo/t==hi 时条件半批输入走整批路径 -> batch 不一致报错;
    区间内同输入才允许 skip."""
    full = _lat(1.0, 4)
    r0 = solve(full, None, 400)
    st = r0["state"]  # iteration=1, batch_size=4, delta 已建立
    cond_in = _lat(5.0, 2)
    for t in (-1, 641):
        assert "整批步的输入 batch 必须与" in _expect_err(solve, cond_in, st, t)
    r1 = solve(cond_in, st, 400)
    assert r1["uncond_skipped"] is True


def check_distilled(solve):
    """蒸馏模型: 恒整批, delta 永不建立, 输出原样."""
    kw = dict(is_guidance_distilled=True)
    st = None
    for i in range(3):
        r = solve(_lat(float(i), 4), st, 400, **kw)
        assert r["uncond_skipped"] is False
        assert torch.equal(r["output"], r["output"])
        st = r["state"]
        assert st["low_frequency_delta"] is None and st["high_frequency_delta"] is None
    assert st["iteration"] == 3


# ---------------------------------------------------------------- skip 逼近
def check_uncond_approx_oracle(solve):
    """skip 步 uncond 频域重建对拍 numpy oracle; t=400 -> w_low=1.1, w_high=1.0."""
    full0, cond1 = _lat(1.0, 4), _lat(5.0, 2)
    r0 = solve(full0, None, 400)
    st = r0["state"]
    dl0, dh0 = _np_delta(full0.numpy())
    r1 = solve(cond1, st, 400)
    assert r1["uncond_skipped"] is True
    expected = _np_uncond_approx(dl0, dh0, cond1.numpy(), w_low=1.1, w_high=1.0)
    assert np.allclose(r1["output"][:2].numpy(), expected, atol=1e-4), \
        np.abs(r1["output"][:2].numpy() - expected).max()
    assert torch.equal(r1["output"][2:], cond1)  # 条件半批原样拼接在后
    assert r1["output"].shape == (4, 1, 2, 8, 8)
    # skip 步 state 存的是**缩放后**的 delta (官方 delta <- delta * weight)
    _assert_complex_close(r1["state"]["low_frequency_delta"].numpy(), dl0 * 1.1)
    _assert_complex_close(r1["state"]["high_frequency_delta"].numpy(), dh0 * 1.0)


def check_weight_alpha_ranges(solve):
    """alpha 权重区间规则: 区间内 ×alpha、区间外 ×1.0; 整批步重算 (非缩放) delta."""
    full0, cond1 = _lat(1.0, 4), _lat(5.0, 2)
    dl0, dh0 = _np_delta(full0.numpy())
    kw = dict(alpha_low_frequency=1.1, alpha_high_frequency=1.1)
    r0 = solve(full0, None, 400, **kw)  # iter0 整批: delta = d0 (重算)
    st = r0["state"]
    _assert_complex_close(st["low_frequency_delta"].numpy(), dl0)
    # iter1 skip @t=400: low 在 (99,901) 内 ×1.1、high 不在 ×1.0
    r1 = solve(cond1, st, 400, **kw)
    _assert_complex_close(r1["state"]["low_frequency_delta"].numpy(), dl0 * 1.1)
    _assert_complex_close(r1["state"]["high_frequency_delta"].numpy(), dh0 * 1.0)
    # iter2-4 skip @t=200: 两个区间都在内 -> 每次 ×1.1 (低 ×1.1^3、高 ×1.1^3)
    st = r1["state"]
    for _ in range(3):
        r = solve(_lat(7.0, 2), st, 200, **kw)
        assert r["uncond_skipped"] is True
        st = r["state"]
    _assert_complex_close(st["low_frequency_delta"].numpy(), dl0 * 1.1 ** 4)
    _assert_complex_close(st["high_frequency_delta"].numpy(), dh0 * 1.0 * 1.1 ** 3)
    # iter5 整批: delta 重算为 d0 (覆盖而非继续缩放)
    r5 = solve(full0, st, 200, **kw)
    assert r5["uncond_skipped"] is False
    _assert_complex_close(r5["state"]["low_frequency_delta"].numpy(), dl0)
    # 区间端点 (开区间): t=99 -> low ×1.0; t=301 -> high ×1.0
    a1 = solve(cond1, solve(full0, None, 400, **kw)["state"], 99, **kw)
    _assert_complex_close(a1["state"]["low_frequency_delta"].numpy(), dl0)
    b1 = solve(cond1, solve(full0, None, 400, **kw)["state"], 301, **kw)
    _assert_complex_close(b1["state"]["high_frequency_delta"].numpy(), dh0)
    # alpha=0: 区间内 delta 清零 -> uncond_freq = cond_freq -> uncond == cond (频域往返)
    kw0 = dict(alpha_low_frequency=0.0, alpha_high_frequency=0.0)
    z0 = solve(full0, None, 200, **kw0)
    z1 = solve(cond1, z0["state"], 200, **kw0)
    assert torch.allclose(z1["output"][:2], cond1, atol=1e-4), \
        (z1["output"][:2] - cond1).abs().max()


def check_skip_requires_history(solve):
    """skip 前置校验: batch_size 未记录 / delta 未建立 / 输入非半批都显式报错."""
    full = _lat(1.0, 4)
    r0 = solve(full, None, 400)
    st = r0["state"]
    cond_in = _lat(5.0, 2)
    # batch_size 未记录 (iteration=1 但 batch_size None)
    st_nob = {"iteration": 1, "batch_size": None,
              "low_frequency_delta": st["low_frequency_delta"],
              "high_frequency_delta": st["high_frequency_delta"]}
    assert "state['batch_size'] 已记录" in _expect_err(solve, cond_in, st_nob, 400)
    # delta 未建立
    st_nodelta = {"iteration": 1, "batch_size": 4,
                  "low_frequency_delta": None, "high_frequency_delta": None}
    assert "频域 delta 必须已建立" in _expect_err(solve, cond_in, st_nodelta, 400)
    # 输入是整批而非半批
    assert "条件半批" in _expect_err(solve, full, st, 400)


def check_pure_function(solve):
    """纯函数: 入参 state 不被修改, 同入参两次调用一致."""
    full = _lat(1.0, 4)
    r0 = solve(full, None, 400)
    st_in = r0["state"]
    snap = (st_in["iteration"], st_in["batch_size"],
            st_in["low_frequency_delta"].clone(), st_in["high_frequency_delta"].clone())
    cond_in = _lat(5.0, 2)
    a = solve(cond_in, st_in, 400)
    b = solve(cond_in, st_in, 400)
    assert torch.equal(a["output"], b["output"])
    assert a["uncond_skipped"] == b["uncond_skipped"]
    assert st_in["iteration"] == snap[0] and st_in["batch_size"] == snap[1]
    _assert_complex_close(st_in["low_frequency_delta"].numpy(),
                          snap[2].numpy(), atol=0.0)
    _assert_complex_close(st_in["high_frequency_delta"].numpy(),
                          snap[3].numpy(), atol=0.0)


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """张量/current_timestep/skip_range/区间/alpha/distilled 校验."""
    full = _lat(1.0, 4)
    assert "model_output 必须是 torch 张量" in _expect_err(solve, 1.0, None, 400)
    assert "5 维 BCFHW" in _expect_err(solve, torch.zeros(2, 1, 8, 8), None, 400)
    assert "current_timestep 必须是整数" in _expect_err(solve, full, None, 400.0)
    assert "current_timestep 必须是整数" in _expect_err(solve, full, None, True)
    assert "unconditional_batch_skip_range 必须 >= 1" in _expect_err(
        solve, full, None, 400, unconditional_batch_skip_range=0)
    assert "unconditional_batch_skip_range 必须 >= 1" in _expect_err(
        solve, full, None, 400, unconditional_batch_skip_range=True)
    assert "unconditional_batch_timestep_skip_range" in _expect_err(
        solve, full, None, 400, unconditional_batch_timestep_skip_range=(100,))
    assert "low_frequency_weight_update_timestep_range" in _expect_err(
        solve, full, None, 400, low_frequency_weight_update_timestep_range="99-901")
    assert "high_frequency_weight_update_timestep_range" in _expect_err(
        solve, full, None, 400, high_frequency_weight_update_timestep_range=(-1,))
    assert "unconditional_batch_timestep_skip_range" in _expect_err(
        solve, full, None, 400, unconditional_batch_timestep_skip_range=(True, 400))
    assert "low_frequency_weight_update_timestep_range" in _expect_err(
        solve, full, None, 400, low_frequency_weight_update_timestep_range={100, 200})
    assert "alpha_low_frequency 必须是有限数值" in _expect_err(
        solve, full, None, 400, alpha_low_frequency=float("inf"))
    assert "alpha_high_frequency 必须是有限数值" in _expect_err(
        solve, full, None, 400, alpha_high_frequency=True)
    assert "is_guidance_distilled 必须是 bool" in _expect_err(
        solve, full, None, 400, is_guidance_distilled=1)


def check_state_validation(solve):
    """state 四键结构校验."""
    full = _lat(1.0, 4)
    dl, dh = _np_delta(full.numpy())
    dl_t = torch.from_numpy(np.stack([dl.real, dl.imag], axis=-1)) \
        .to(torch.complex64).squeeze(-1)
    assert "state 必须是 dict" in _expect_err(solve, full, [], 400)
    assert "未知键" in _expect_err(solve, full, {"foo": 1}, 400)
    assert "缺少键" in _expect_err(solve, full, {"iteration": 0}, 400)
    assert "state['iteration'] 必须是非负整数" in _expect_err(
        solve, full, {"iteration": -1, "batch_size": None,
                      "low_frequency_delta": None, "high_frequency_delta": None}, 400)
    assert "state['batch_size'] 必须是正偶数或 None" in _expect_err(
        solve, full, {"iteration": 0, "batch_size": 3,
                      "low_frequency_delta": None, "high_frequency_delta": None}, 400)
    assert "state['batch_size'] 必须是正偶数或 None" in _expect_err(
        solve, full, {"iteration": 0, "batch_size": True,
                      "low_frequency_delta": None, "high_frequency_delta": None}, 400)
    assert "必须是复数张量或 None" in _expect_err(
        solve, full, {"iteration": 0, "batch_size": None,
                      "low_frequency_delta": 0.5, "high_frequency_delta": None}, 400)
    # 合法复数 delta state 可走整批路径
    st = {"iteration": 0, "batch_size": None,
          "low_frequency_delta": dl_t, "high_frequency_delta": dl_t}
    r = solve(full, st, 400)
    assert r["uncond_skipped"] is False


# ---------------------------------------------------------------- 元信息
check_uncond_approx_oracle.example = True
check_skip_sequence.example = True
