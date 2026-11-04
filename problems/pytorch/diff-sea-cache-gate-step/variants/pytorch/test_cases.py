"""SeaCache 频域指示器门控步 -- 测试用例."""

import numpy as np
import torch


def _lat(c, t, h, w, delta=0.0):
    """确定性 (C,T,H,W) 视觉隐张量, 值域约 [-1,1]; delta 整体平移."""
    n = c * t * h * w
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * 0.37) % 1.0) * 2.0 - 1.0 + delta
    return x.reshape(c, t, h, w).contiguous()


def _g(v):
    return torch.tensor([[v, v * 0.5 - 0.3]], dtype=torch.float32)


def _ind_np(latent, sigma, power_exp=3.0):
    """独立 numpy oracle 指示器: (C,T,H,W) -> movedim(0,-1) -> filter -> (T,H,W,C)."""
    return _np_filter(latent.movedim(0, -1).numpy(), sigma, power_exp)


def _np_filter(x_thwc, sigma, power_exp):
    """独立 numpy oracle: 官方 _apply_sea_filter (float64 复算)."""
    x = np.asarray(x_thwc, dtype=np.float64)
    spectrum = np.fft.fftn(x, axes=(0, 1, 2))
    sigma = max(1e-6, min(1.0 - 1e-6, float(sigma)))
    s, n = 1.0 - sigma, sigma
    gain = None
    for axis in range(3):
        freqs = np.fft.fftfreq(x.shape[axis]).astype(np.float32).astype(np.float64)
        cp = 1.0 / (np.abs(freqs) ** power_exp + 1e-16)
        ag = s * cp / (s * s * cp + n * n + 1e-16)
        shape = [1, 1, 1, 1]
        shape[axis] = ag.shape[0]
        g = ag.reshape(shape)
        gain = g if gain is None else gain * g
    m = gain.mean()
    if np.isfinite(m) and m > 0:
        gain = gain / m
    return np.fft.ifftn(spectrum * gain, axes=(0, 1, 2)).real


def _np_distance(cur, prev):
    num = np.abs(cur.astype(np.float64) - prev.astype(np.float64)).mean()
    den = np.abs(prev.astype(np.float64)).mean() + 1e-16
    return float(num / den)


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


def _fresh_state(**over):
    st = {"history": [], "previous_indicator": None, "accumulated_distance": 0.0,
          "consecutive_cached": 0, "last_step": None}
    st.update(over)
    return st


# ---------------------------------------------------------------- sea_filter 指示器
def check_indicator_identity_1d(solve):
    """T=H=W=1 时 fftfreq(1)=[0] -> clean_power≈1e16, 归一化后增益≈1: 指示器≈恒等."""
    latent = _lat(2, 1, 1, 1)
    r = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 4)
    ind = r["state"]["previous_indicator"]
    assert ind.shape == (1, 1, 1, 2), ind.shape
    assert torch.allclose(ind, latent.movedim(0, -1), atol=1e-4), (ind, latent)


def check_indicator_oracle(solve):
    """指示器对拍独立 numpy oracle; sigma=0/1 走 clamp 端点."""
    latent = _lat(2, 2, 4, 4)
    expected = _ind_np(latent, 0.3)
    r = solve(latent, _g(1.0), _g(1.2), None, 0, 0.3, 6)
    ind = r["state"]["previous_indicator"].numpy()
    assert np.allclose(ind, expected, atol=1e-4), np.abs(ind - expected).max()
    for sigma in (0.0, 1.0):  # sigma=0 -> clamp 1e-6; sigma=1 -> clamp 1-1e-6
        expected_c = _ind_np(latent, sigma)
        r2 = solve(latent, _g(1.0), _g(1.2), None, 0, sigma, 6)
        assert np.allclose(r2["state"]["previous_indicator"].numpy(), expected_c,
                           atol=1e-4)


def check_sigma_change_distance(solve):
    """同一 latent 换 sigma: 指示器变化 -> distance > 0 (sigma 参与滤波)."""
    latent = _lat(2, 2, 4, 4)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.2, 5, threshold=10.0)
    st = r0["state"]
    assert r0["computed"] is True  # retention
    r1 = solve(latent, _g(1.0), _g(1.2), st, 1, 0.8, 5, threshold=10.0)
    assert r1["distance"] is not None and r1["distance"] > 1e-4, r1["distance"]
    assert r1["computed"] is False  # distance < 10 -> skip
    assert r1["state"]["accumulated_distance"] == r1["distance"]


# ---------------------------------------------------------------- 门控与状态机
def check_first_step_retention(solve):
    """retention 窗内强制 compute: distance=None, history 记录, consecutive=0."""
    latent = _lat(1, 2, 2, 2)
    r = solve(latent, _g(1.0), _g(1.4), None, 0, 0.4, 4,
              retention_steps=1, cache_end_steps=1)
    assert r["computed"] is True and r["distance"] is None
    assert r["accumulated_distance"] == 0.0
    st = r["state"]
    assert st["last_step"] == 0 and st["consecutive_cached"] == 0
    assert len(st["history"]) == 1 and st["history"][0]["step"] == 0
    assert torch.equal(st["history"][0]["residual"], _g(1.4) - _g(1.0))


def check_boundary_threshold_zero(solve):
    """candidate >= threshold 的 >= 语义: 阈值 0 + 零距离 (同 latent 同 sigma)
    -> candidate 恰为 0.0, 官方 compute; > 变异会错误 skip."""
    latent = _lat(2, 2, 4, 4)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.5, 4,
               threshold=0.0, retention_steps=1, cache_end_steps=0)
    assert r0["computed"] is True
    r1 = solve(latent, _g(1.0), _g(1.2), r0["state"], 1, 0.5, 4,
               threshold=0.0, retention_steps=1, cache_end_steps=0)
    assert r1["distance"] == 0.0, r1["distance"]  # 逐位相同 -> 零距离
    assert r1["computed"] is True  # 0.0 >= 0.0
    assert r1["accumulated_distance"] == 0.0


def check_accumulated_skip_then_recover(solve):
    """累积预算: 首个小距离 skip (accumulated=distance), 累加越界 compute+归零."""
    l0 = _lat(2, 2, 4, 4)
    l1 = _lat(2, 2, 4, 4, delta=0.02)   # 小变化 -> 小 distance
    l2 = _lat(2, 2, 4, 4, delta=0.5)    # 大变化 -> 大 distance
    thr = 0.3
    d1 = _np_distance(_ind_np(l1, 0.4), _ind_np(l0, 0.4))
    d2 = _np_distance(_ind_np(l2, 0.4), _ind_np(l1, 0.4))
    assert d1 < thr * 0.5 and d2 > thr, (d1, d2)  # 余量远离边界

    r0 = solve(l0, _g(1.0), _g(1.2), None, 0, 0.4, 6, threshold=thr,
               retention_steps=1, cache_end_steps=0)
    st = r0["state"]
    r1 = solve(l1, _g(2.0), _g(2.5), st, 1, 0.4, 6,
               threshold=thr, retention_steps=1, cache_end_steps=0)
    assert r1["computed"] is False
    assert abs(r1["accumulated_distance"] - d1) < 1e-5
    # 无外推 (单条 history): output = gen_input + 最近残差 (step0 的)
    assert torch.allclose(r1["output"], _g(2.0) + (_g(1.2) - _g(1.0)), atol=1e-6)
    st = r1["state"]
    r2 = solve(l2, _g(3.0), _g(3.5), st, 2, 0.4, 6,
               threshold=thr, retention_steps=1, cache_end_steps=0)
    assert r2["computed"] is True  # d1 + d2 >= thr
    assert r2["accumulated_distance"] == 0.0  # compute 步归零
    assert torch.equal(r2["output"], _g(3.5))
    assert r2["state"]["history"][-1]["step"] == 2


def check_distance_value_rel(solve):
    """distance 数值 (rel 1e-3): 分母必须是 previous 的 absmean (换成 current
    会在大 delta 下偏 ~9%)."""
    l0 = _lat(2, 2, 4, 4)
    l1 = _lat(2, 2, 4, 4, delta=0.3)
    d1 = _np_distance(_ind_np(l1, 0.4), _ind_np(l0, 0.4))
    r0 = solve(l0, _g(1.0), _g(1.2), None, 0, 0.4, 6, threshold=2.0,
               retention_steps=1, cache_end_steps=0)
    r1 = solve(l1, _g(1.0), _g(1.2), r0["state"], 1, 0.4, 6, threshold=2.0,
               retention_steps=1, cache_end_steps=0)
    assert r1["distance"] is not None
    assert abs(r1["distance"] - d1) < 1e-3 * d1, (r1["distance"], d1)
    assert r1["computed"] is False  # 0.65 < 2.0 -> skip


def check_non_adjacent_reset(solve):
    """步非相邻 (≠上次+1, 前跳/回跳都算): 整条轨迹重置."""
    latent = _lat(1, 2, 2, 2)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 8)
    st = r0["state"]
    st["accumulated_distance"] = 0.123  # 模拟已累积
    r1 = solve(latent, _g(2.0), _g(2.2), st, 3, 0.4, 8)  # 3 != 0+1 (前跳)
    assert r1["computed"] is True and r1["distance"] is None  # 重置后首观测
    assert len(r1["state"]["history"]) == 1
    assert r1["state"]["history"][0]["step"] == 3
    assert r1["state"]["last_step"] == 3
    # 回跳: last_step=3 -> 步 1 (1 != 4) 同样重置
    r2 = solve(latent, _g(3.0), _g(3.2), r1["state"], 1, 0.4, 8)
    assert r2["computed"] is True and r2["distance"] is None
    assert r2["state"]["history"][0]["step"] == 1


def check_max_consecutive(solve):
    """连续缓存达上限强制 compute; max=0 不设限."""
    latent = _lat(1, 2, 2, 2)
    kw = dict(threshold=0.5, retention_steps=1, cache_end_steps=0,
              max_consecutive_cached=2)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 8, **kw)
    st = r0["state"]
    seq = []
    for s in range(1, 5):
        r = solve(latent, _g(1.0), _g(1.2), st, s, 0.4, 8, **kw)
        seq.append(r["computed"])
        st = r["state"]
    # 零距离: 步 1,2 skip (consecutive 1,2), 步 3 达上限强制 compute, 步 4 再 skip
    assert seq == [False, False, True, False], seq
    # max=0: 不设限, 全程 skip
    kw0 = dict(kw, max_consecutive_cached=0)
    r0b = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 8, **kw0)
    st = r0b["state"]
    for s in range(1, 5):
        r = solve(latent, _g(1.0), _g(1.2), st, s, 0.4, 8, **kw0)
        assert r["computed"] is False
        st = r["state"]


def check_cache_end_window(solve):
    """cache_end 窗: step >= N - cache_end_steps 强制 compute (步须相邻推进)."""
    latent = _lat(1, 2, 2, 2)
    kw = dict(threshold=0.5, retention_steps=1, cache_end_steps=1,
              max_consecutive_cached=0)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 4, **kw)
    st = r0["state"]
    r1 = solve(latent, _g(1.0), _g(1.2), st, 1, 0.4, 4, **kw)
    assert r1["computed"] is False  # 步 1 在窗外的可跳区
    r2 = solve(latent, _g(1.0), _g(1.2), r1["state"], 2, 0.4, 4, **kw)
    assert r2["computed"] is False  # 步 2 仍在窗外 (2 < 4-1)
    r3 = solve(latent, _g(1.0), _g(1.2), r2["state"], 3, 0.4, 4, **kw)
    assert r3["computed"] is True and r3["distance"] is None  # 3 >= 4-1


def check_cache_hit_without_history(solve):
    """门控判 skip 但 history 为空 -> fail-open compute + accumulated 归零."""
    latent = _lat(1, 2, 2, 2)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 6)
    st = _fresh_state(previous_indicator=r0["state"]["previous_indicator"],
                      last_step=0)  # history 空
    r1 = solve(latent, _g(1.0), _g(1.2), st, 1, 0.4, 6, threshold=0.5)
    assert r1["distance"] == 0.0  # 门控本判 skip
    assert r1["computed"] is True  # fail-open
    assert r1["accumulated_distance"] == 0.0
    assert len(r1["state"]["history"]) == 1


def check_history_shape_mismatch(solve):
    """history 残差形状/dtype 与 gen_input 不符 -> 清空 history fail-open compute."""
    latent = _lat(1, 2, 2, 2)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 6)
    st = dict(r0["state"])
    st["history"] = [{"step": 0, "residual": torch.zeros(5, 5, dtype=torch.float32)}]
    r1 = solve(latent, _g(1.0), _g(1.2), st, 1, 0.4, 6, threshold=0.5)
    assert r1["computed"] is True  # fail-open
    assert len(r1["state"]["history"]) == 1  # 旧 history 已清, 只剩本步记录
    assert r1["state"]["history"][0]["step"] == 1
    # dtype 不符同理
    st2 = dict(r0["state"])
    st2["history"] = [{"step": 0, "residual": (_g(1.2) - _g(1.0)).to(torch.float64)}]
    r2 = solve(latent, _g(1.0), _g(1.2), st2, 1, 0.4, 6, threshold=0.5)
    assert r2["computed"] is True


# ---------------------------------------------------------------- 残差复用
def check_residual_extrapolation(solve):
    """residual_order=1: 两条不同步记录 -> 线性外推缓存残差."""
    l0 = _lat(1, 2, 2, 2)
    kw = dict(threshold=0.5, retention_steps=2, cache_end_steps=0,
              residual_order=1, max_consecutive_cached=0)
    # 步 0,1 被 retention 强制 compute, 残差 R0/R1
    in0, out0 = _g(1.0), _g(1.4)
    in1, out1 = _g(2.0), _g(2.9)
    r0 = solve(l0, in0, out0, None, 0, 0.4, 6, **kw)
    r1 = solve(l0, in1, out1, r0["state"], 1, 0.4, 6, **kw)
    assert r0["computed"] is True and r1["computed"] is True
    # 步 2 同 latent 同 sigma -> 零距离 -> skip, 外推 scale=(2-1)/(1-0)=1
    in2 = _g(3.0)
    r2 = solve(l0, in2, _g(9.9), r1["state"], 2, 0.4, 6, **kw)
    assert r2["computed"] is False
    res0 = out0 - in0
    res1 = out1 - in1
    expected = in2 + (res1 + (res1 - res0) * 1.0)
    assert torch.allclose(r2["output"], expected, atol=1e-6), (r2["output"], expected)
    assert r2["state"]["consecutive_cached"] == 1
    assert len(r2["state"]["history"]) == 2  # skip 步不追加


def check_residual_extrapolation_scale(solve):
    """外推 step_scale=(step−latest)/(latest−prev) 在历史步非相邻时 ≠ 1:
    步 0,2 计算 (retention=1 + 大距离), 步 3 skip -> scale=0.5."""
    l0 = _lat(1, 2, 2, 2)
    l2 = _lat(1, 2, 2, 2, delta=0.5)  # 步 2 大距离 -> compute
    l3 = _lat(1, 2, 2, 2, delta=0.5)  # 与 l2 相同 -> 步 3 零距离 skip
    kw = dict(threshold=0.5, retention_steps=1, cache_end_steps=0,
              residual_order=1, max_consecutive_cached=0)
    in0, out0 = _g(1.0), _g(1.4)
    in2, out2 = _g(2.0), _g(2.5)
    r0 = solve(l0, in0, out0, None, 0, 0.4, 6, **kw)
    assert r0["computed"] is True
    # 步 1 与步 0 同指示器 -> 零距离 skip (维持步邻接)
    r1 = solve(l0, _g(1.5), _g(1.9), r0["state"], 1, 0.4, 6, **kw)
    assert r1["computed"] is False
    d2 = _np_distance(_ind_np(l2, 0.4), _ind_np(l0, 0.4))
    assert d2 > 0.5, d2  # 余量: 步 2 越界 compute
    r2 = solve(l2, in2, out2, r1["state"], 2, 0.4, 6, **kw)
    assert r2["computed"] is True
    # 步 3 与步 2 同指示器 -> 零距离 skip; history = [步0, 步2]
    r3 = solve(l3, _g(3.0), _g(9.9), r2["state"], 3, 0.4, 6, **kw)
    assert r3["computed"] is False
    res0 = out0 - in0
    res2 = out2 - in2
    scale = (3 - 2) / (2 - 0)  # 0.5
    expected = _g(3.0) + (res2 + (res2 - res0) * scale)
    assert torch.allclose(r3["output"], expected, atol=1e-6), (r3["output"], expected)


def check_indicator_shape_mismatch(solve):
    """previous_indicator 形状不符 -> distance=inf -> 强制 compute."""
    latent = _lat(1, 2, 2, 2)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 6)
    st = dict(r0["state"])
    st["previous_indicator"] = torch.zeros(3, 3, 3, 2, dtype=torch.float32)
    r1 = solve(latent, _g(1.0), _g(1.2), st, 1, 0.4, 6, threshold=0.5)
    assert r1["computed"] is True
    assert r1["distance"] == float("inf")
    assert r1["accumulated_distance"] == 0.0


def check_residual_order_zero(solve):
    """residual_order=0: 直接复用最近残差, 不外推."""
    l0 = _lat(1, 2, 2, 2)
    kw = dict(threshold=0.5, retention_steps=2, cache_end_steps=0,
              residual_order=0, max_consecutive_cached=0)
    in0, out0 = _g(1.0), _g(1.4)
    in1, out1 = _g(2.0), _g(2.9)
    r0 = solve(l0, in0, out0, None, 0, 0.4, 6, **kw)
    r1 = solve(l0, in1, out1, r0["state"], 1, 0.4, 6, **kw)
    r2 = solve(l0, _g(3.0), _g(9.9), r1["state"], 2, 0.4, 6, **kw)
    assert r2["computed"] is False
    assert torch.allclose(r2["output"], _g(3.0) + (out1 - in1), atol=1e-6)


def check_full_loop(solve):
    """6 步全循环: 独立 oracle 状态机对拍 computed 序列."""
    lats = [_lat(2, 2, 4, 4), _lat(2, 2, 4, 4, delta=0.02), _lat(2, 2, 4, 4, delta=0.5),
            _lat(2, 2, 4, 4, delta=0.5), _lat(2, 2, 4, 4, delta=0.03),
            _lat(2, 2, 4, 4, delta=0.6)]
    sigma, thr, n = 0.4, 0.25, len(lats)
    inds = [_ind_np(l, sigma) for l in lats]

    # 余量自检: 距离远离 0.25 边界, float32/float64 差异不会翻转决策
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        d = _np_distance(inds[b], inds[a])
        assert d < thr * 0.7 or d > thr * 1.3, (a, b, d)

    # oracle 状态机 (独立转写官方 gate 语义)
    history_len, prev, acc, consec, last = 0, None, 0.0, 0, None
    expected = []
    for s in range(n):
        if last is not None and s != last + 1:
            history_len, prev, acc, consec = 0, None, 0.0, 0
        forced = s < 1 or s >= n - 1 or prev is None or consec >= 2
        comp = True
        cand = 0.0
        if not forced:
            cand = acc + _np_distance(inds[s], prev)
            comp = cand >= thr
        acc = 0.0 if comp else cand
        if comp:
            history_len = min(history_len + 1, 2)
            consec = 0
        else:
            consec += 1
        prev = inds[s]
        last = s
        expected.append(comp)

    st = None
    seq = []
    for s in range(n):
        gen_in = _g(float(s) + 1.0)
        gen_out = gen_in + _g(0.3)
        r = solve(lats[s], gen_in, gen_out, st, s, sigma, n, threshold=thr,
                  retention_steps=1, cache_end_steps=1, max_consecutive_cached=2)
        seq.append(r["computed"])
        st = r["state"]
    assert seq == expected, (seq, expected)
    assert seq[0] is True and seq[-1] is True  # retention + cache_end
    assert st["last_step"] == n - 1


def check_pure_function(solve):
    """纯函数: 入参 state 不被修改, 同入参两次调用一致."""
    latent = _lat(1, 2, 2, 2)
    r0 = solve(latent, _g(1.0), _g(1.2), None, 0, 0.4, 6)
    st_in = r0["state"]
    snap_hist = [(e["step"], e["residual"].clone()) for e in st_in["history"]]
    snap_ind = st_in["previous_indicator"].clone()
    a = solve(latent, _g(1.0), _g(1.2), st_in, 1, 0.4, 6)
    b = solve(latent, _g(1.0), _g(1.2), st_in, 1, 0.4, 6)
    assert torch.equal(a["output"], b["output"])
    assert a["computed"] == b["computed"] and a["distance"] == b["distance"]
    assert len(st_in["history"]) == len(snap_hist)
    assert all(s == e["step"] and torch.equal(rr, e["residual"])
               for (s, rr), e in zip(snap_hist, st_in["history"]))
    assert torch.equal(st_in["previous_indicator"], snap_ind)
    assert st_in["last_step"] == 0  # 入参 last_step 未被改


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """张量/形状/step/N/sigma/threshold/residual_order/retention/cache_end/
    max_consecutive/power_exp 校验."""
    latent = _lat(1, 2, 2, 2)
    kw = dict(threshold=0.25, residual_order=1, retention_steps=1,
              cache_end_steps=1, max_consecutive_cached=2, power_exp=3.0)
    assert "vision_latent 必须是 torch 张量" in _expect_err(
        solve, 1.0, _g(1.0), _g(1.2), None, 0, 0.4, 6, **kw)
    assert "gen_input 必须是 torch 张量" in _expect_err(
        solve, latent, None, _g(1.2), None, 0, 0.4, 6, **kw)
    assert "gen_output 必须是 torch 张量" in _expect_err(
        solve, latent, _g(1.0), None, None, 0, 0.4, 6, **kw)
    assert "4 维" in _expect_err(
        solve, _g(1.0), _g(1.0), _g(1.2), None, 0, 0.4, 6, **kw)
    assert "形状必须一致" in _expect_err(
        solve, latent, _g(1.0), torch.zeros((2, 2), dtype=torch.float32),
        None, 0, 0.4, 6, **kw)
    assert "step_index 必须是非负整数" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, -1, 0.4, 6, **kw)
    assert "step_index 必须是非负整数" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, True, 0.4, 6, **kw)
    assert "num_inference_steps 必须 >= 1" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 0, **kw)
    assert "小于 num_inference_steps" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 6, 0.4, 6, **kw)
    assert "sigma 必须落在 [0, 1]" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 1.5, 6, **kw)
    assert "sigma 必须落在 [0, 1]" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, -0.1, 6, **kw)
    assert "threshold 必须是非负有限数值" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 6, threshold=-0.1)
    assert "residual_order 必须是 0 或 1" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 6, residual_order=2)
    assert "retention_steps 必须是非负整数" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 6, retention_steps=-1)
    assert "cache_end_steps 必须是非负整数" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 6, cache_end_steps=-1)
    assert "max_consecutive_cached 必须是非负整数" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 6, max_consecutive_cached=True)
    assert "power_exp 必须是正的有限数值" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), None, 0, 0.4, 6, power_exp=0.0)


def check_state_validation(solve):
    """state 五键结构校验."""
    latent = _lat(1, 2, 2, 2)
    kw = dict(threshold=0.25)
    assert "state 必须是 dict" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), [], 0, 0.4, 6, **kw)
    assert "未知键" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), {"foo": 1}, 0, 0.4, 6, **kw)
    assert "缺少键" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), {"history": []}, 0, 0.4, 6, **kw)
    assert "state['history'] 必须是 list" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), _fresh_state(history="x"), 0, 0.4, 6, **kw)
    assert "history 条目" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), _fresh_state(history=[{"step": 0}]),
        0, 0.4, 6, **kw)
    assert "step 必须是非负整数" in _expect_err(
        solve, latent, _g(1.0), _g(1.2),
        _fresh_state(history=[{"step": True, "residual": _g(0.1)}]), 0, 0.4, 6, **kw)
    assert "residual 必须是 torch 张量" in _expect_err(
        solve, latent, _g(1.0), _g(1.2),
        _fresh_state(history=[{"step": 0, "residual": 0.5}]), 0, 0.4, 6, **kw)
    assert "必须是 torch 张量或 None" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), _fresh_state(previous_indicator=0.5),
        0, 0.4, 6, **kw)
    assert "accumulated_distance" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), _fresh_state(accumulated_distance=-0.1),
        0, 0.4, 6, **kw)
    assert "consecutive_cached" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), _fresh_state(consecutive_cached=True),
        0, 0.4, 6, **kw)
    assert "last_step" in _expect_err(
        solve, latent, _g(1.0), _g(1.2), _fresh_state(last_step=-1),
        0, 0.4, 6, **kw)


# ---------------------------------------------------------------- 元信息
check_indicator_oracle.example = True
check_full_loop.example = True
