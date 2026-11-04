"""MagCache 累积误差门控步 -- 测试用例."""

import numpy as np
import torch


def _t(v):
    return torch.tensor([[float(v), v * 0.5 - 0.3]], dtype=torch.float64)


def _close(actual, expected, tol=1e-9):
    a = np.asarray(torch.detach(actual).to(torch.float64).numpy(), dtype=np.float64).reshape(-1)
    e = np.broadcast_to(np.asarray(expected, dtype=np.float64).reshape(-1), a.shape)
    return bool(np.all(np.abs(a - e) <= tol))


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


def _gen_out(step):
    v = 0.7 * step - 0.05 * step * step
    return _t(v)


# 独立 oracle: 门控状态机 -------------------------------------------------
def _oracle_loop(n_steps, ratios, threshold=0.06, retention_ratio=0.2, max_skip=3):
    """返回 [(step, computed, err_after)], ratios 为 list (长度 == n)."""
    retention = int(retention_ratio * n_steps + 0.5)
    res = None
    ratio, steps_c, err = 1.0, 0, 0.0
    out = []
    for s in range(n_steps):
        computed = True
        if s >= retention:
            ratio *= ratios[s]
            steps_c += 1
            err += abs(1.0 - ratio)
            if res is not None and err <= threshold and steps_c <= max_skip:
                computed = False
            else:
                ratio, steps_c, err = 1.0, 0, 0.0
        out.append((s, computed, err))
        if computed:
            res = "updated"  # 占位: compute 步更新残差
    return out


# ---------------------------------------------------------------- 全循环对拍
def check_full_loop(solve):
    """N=10, retention 0.2 -> retention_step=2; ratios 从 0.98 递减; 对拍序列."""
    n = 10
    ratios = torch.tensor([0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.945, 0.94, 0.93],
                          dtype=torch.float64)
    st = None
    seq = []
    for s in range(n):
        r = solve(_gen_out(s), _gen_out(s) + _t(0.1), st, s, n, ratios,
                  threshold=0.06, retention_ratio=0.2, max_skip_steps=3)
        seq.append(r["computed"])
        st = r["state"]
    expected = _oracle_loop(n, ratios.tolist())
    assert seq == [e[1] for e in expected], (seq, [e[1] for e in expected])
    # 首两步 retention: 强制 compute
    assert seq[0] is True and seq[1] is True
    # compute 步 residual 更新、skip 步 output = hidden + residual
    # (逐数值由下面专用用例覆盖)


def check_retention_rounding(solve):
    """retention_step = int(ratio*N + 0.5) 四舍五入: 0.2*8=1.6 -> 2; 0.25*10=2.5 -> 3."""
    ratios = torch.ones(10, dtype=torch.float64) * 1.5  # 大 ratio -> err 快速累积
    # N=8, retention_ratio=0.2 -> int(1.6+0.5)=2 -> step 2 起可 skip 判定
    st = None
    r = solve(_t(1.0), _t(1.5), st, 1, 8, ratios[:8].clone(),
              threshold=100.0, retention_ratio=0.2, max_skip_steps=10)
    assert r["computed"] is True and r["state"]["accumulated_steps"] == 0  # retention 内不累积
    # N=10, retention_ratio=0.25 -> int(2.5+0.5)=3 -> step 2 仍 retention (不累积)
    r2 = solve(_t(1.0), _t(1.5), None, 2, 10, ratios,
               threshold=100.0, retention_ratio=0.25, max_skip_steps=10)
    assert r2["computed"] is True and r2["state"]["accumulated_steps"] == 0
    # step 3 起: 累积开始; threshold=100 下 err=0.5<=100 且残差已建立 -> skip,
    # skip 步保留累积器 (ratio=1.5, err=0.5)
    r3 = solve(_t(1.0), _t(1.5), r2["state"], 3, 10, ratios,
               threshold=100.0, retention_ratio=0.25, max_skip_steps=10)
    assert r3["computed"] is False
    assert r3["state"]["accumulated_ratio"] == 1.5
    assert r3["state"]["accumulated_steps"] == 1
    assert abs(r3["state"]["accumulated_err"] - abs(1.0 - 1.5)) < 1e-12
    # 反截断 (int(r*N+0.5) != int(r*N) 需 frac(r*N)>=0.5; 0.2*8=1.6 -> 2 vs 1):
    # 带非 None 残差的步 1, retention 内必须零累积 (若被截断成 retention=1,
    # step 1 会累积 ratio=1.5/err=0.5 并走 skip 分支保留累积器)
    st_pre = {"previous_residual": _t(0.2), "accumulated_ratio": 1.0,
              "accumulated_steps": 0, "accumulated_err": 0.0}
    r4 = solve(_t(1.0), _t(1.5), st_pre, 1, 8, ratios[:8].clone(),
               threshold=100.0, retention_ratio=0.2, max_skip_steps=10)
    assert r4["computed"] is True
    assert r4["state"]["accumulated_steps"] == 0
    assert r4["state"]["accumulated_err"] == 0.0


def check_accumulated_error(solve):
    """累积误差手算: 连续 skip 时 err = Σ|1-Πscale| (不是单步 |1-scale|)."""
    n = 10
    ratios = torch.tensor([1.0] * 2 + [0.98] * 8, dtype=torch.float64)
    # retention_step = int(0.2*10+0.5) = 2
    st = None
    # step 2: compute (previous_residual None) -> 重置? 判定失败 -> 重置 + compute
    r2 = solve(_t(1.0), _t(1.2), st, 2, n, ratios, threshold=0.06)
    assert r2["computed"] is True
    assert r2["state"]["accumulated_ratio"] == 1.0 and r2["state"]["accumulated_err"] == 0.0
    st = r2["state"]
    # step 3: ratio = 0.98, err = |1-0.98| = 0.02 <= 0.06, res 非 None -> skip
    r3 = solve(_t(1.0), _t(1.2), st, 3, n, ratios, threshold=0.06)
    assert r3["computed"] is False
    assert abs(r3["state"]["accumulated_err"] - 0.02) < 1e-12
    assert _close(r3["output"], (_t(1.0) + (_t(1.2) - _t(1.0))).to(torch.float64).numpy(), tol=0.0)
    st = r3["state"]
    # step 4: ratio = 0.98^2 = 0.9604, err = 0.02 + |1-0.9604| = 0.02 + 0.0396 = 0.0596 <= 0.06 -> skip
    r4 = solve(_t(1.0), _t(1.2), st, 4, n, ratios, threshold=0.06)
    assert r4["computed"] is False
    assert abs(r4["state"]["accumulated_err"] - (0.02 + abs(1.0 - 0.98 ** 2))) < 1e-12
    st = r4["state"]
    # step 5: ratio = 0.98^3, err = 0.0596 + |1-0.941192| = 0.0596 + 0.0588 = 0.1184 > 0.06 -> compute + 重置
    r5 = solve(_t(1.0), _t(1.2), st, 5, n, ratios, threshold=0.06)
    assert r5["computed"] is True
    assert r5["state"]["accumulated_ratio"] == 1.0
    assert r5["state"]["accumulated_steps"] == 0
    assert r5["state"]["accumulated_err"] == 0.0


def check_max_skip_steps(solve):
    """连续 skip 到 accumulated_steps > max_skip_steps -> 强制 compute + 重置."""
    n = 10
    ratios = torch.tensor([1.0] * 2 + [1.0] * 8, dtype=torch.float64)  # scale=1 -> err 恒 0
    st = None
    r2 = solve(_t(1.0), _t(1.2), st, 2, n, ratios, threshold=0.06, max_skip_steps=3)
    st = r2["state"]  # compute, res 建立, 重置
    seq = []
    for s in range(3, 9):
        r = solve(_t(1.0), _t(1.2), st, s, n, ratios, threshold=0.06, max_skip_steps=3)
        seq.append(r["computed"])
        st = r["state"]
    # steps 累积: 1,2,3 skip (<=3), 第 4 次 (steps=4) 强制 compute + 重置, 之后 1,2 skip...
    assert seq == [False, False, False, True, False, False], seq
    # compute 步重置后 steps=0 (state 里), 下一步 skip 时 steps=1
    assert st["accumulated_steps"] == 2


def check_stale_residual(solve):
    """skip 步 output 用上次 compute 的残差; compute 步更新残差."""
    n = 10
    ratios = torch.tensor([1.0] * 10, dtype=torch.float64)
    st = None
    r2 = solve(_t(1.0), _t(1.2), st, 2, n, ratios, threshold=0.06)
    st = r2["state"]
    assert _close(st["previous_residual"], (_t(1.2) - _t(1.0)).to(torch.float64).numpy(), tol=0.0)
    # step 3 skip: output = hidden + 旧残差 (与 step 3 的 tail_output 无关)
    r3 = solve(_t(1.0), _t(99.0), st, 3, n, ratios, threshold=0.06)
    assert r3["computed"] is False
    assert _close(r3["output"], _t(1.2).to(torch.float64).numpy(), tol=0.0)
    assert _close(r3["state"]["previous_residual"],
                  (_t(1.2) - _t(1.0)).to(torch.float64).numpy(), tol=0.0)
    # step 4 compute (tail 换新): residual 更新
    r4 = solve(_t(1.0), _t(1.4), r3["state"], 4, n, ratios, threshold=0.06,
               max_skip_steps=1)  # steps=2 > 1 -> 强制 compute
    assert r4["computed"] is True
    # compute 步 output = tail_output (完整执行结果), 不是 hidden_states
    assert _close(r4["output"], _t(1.4).to(torch.float64).numpy(), tol=0.0)
    assert _close(r4["state"]["previous_residual"],
                  (_t(1.4) - _t(1.0)).to(torch.float64).numpy(), tol=0.0)


def check_pure_function(solve):
    """纯函数: 同入参两次调用一致, 入参 state 不被修改."""
    st_in = {"previous_residual": _t(0.2), "accumulated_ratio": 1.0,
             "accumulated_steps": 0, "accumulated_err": 0.0}
    ratios = torch.tensor([1.0] * 10, dtype=torch.float64)
    a = solve(_t(1.0), _t(1.2), st_in, 2, 10, ratios, threshold=0.06)
    b = solve(_t(1.0), _t(1.2), st_in, 2, 10, ratios, threshold=0.06)
    assert torch.equal(a["output"], b["output"]) and a["computed"] == b["computed"]
    assert torch.equal(st_in["previous_residual"], _t(0.2))
    assert st_in["accumulated_steps"] == 0


# ---------------------------------------------------------------- nearest_interp
def check_nearest_interp(solve):
    """nearest_interp 通过 solve 的自动插值间接验证: err = |1 - 插值后 ratios[step]|."""
    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    res = _t(0.2)
    def run(step, n):
        st = {"previous_residual": res, "accumulated_ratio": 1.0,
              "accumulated_steps": 0, "accumulated_err": 0.0}
        r = solve(_t(1.0), _t(1.2), st, step, n, src,
                  threshold=100.0, retention_ratio=0.0, max_skip_steps=100)
        return r["state"]["accumulated_err"]
    # N=6: scale=4/5=0.8, idx=round([0,.8,1.6,2.4,3.2,4])=[0,1,2,2,3,4]
    # -> 插值 ratios=[1,2,3,3,4,5]; err(step)=|1-scale|
    assert run(1, 6) == 1.0, run(1, 6)   # idx 1 -> scale 2
    assert run(2, 6) == 2.0, run(2, 6)   # idx round(1.6)=2 -> scale 3
    assert run(3, 6) == 2.0              # idx round(2.4)=2 -> scale 3
    assert run(5, 6) == 4.0              # idx 4 -> scale 5
    # N=1: target==1 -> 取 src[-1:] -> scale 5 -> err 4
    assert run(0, 1) == 4.0
    # torch.round 的银行家舍入: src 长度 3、N=5 -> scale=0.5
    # grid*scale = [0, .5, 1, 1.5, 2]; round(0.5)=0 (banker) -> step1 scale=1 -> err 0
    src3 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    st = {"previous_residual": res, "accumulated_ratio": 1.0,
          "accumulated_steps": 0, "accumulated_err": 0.0}
    r = solve(_t(1.0), _t(1.2), st, 1, 5, src3,
              threshold=100.0, retention_ratio=0.0, max_skip_steps=100)
    assert r["state"]["accumulated_err"] == 0.0, r["state"]["accumulated_err"]
    # list[float] 入参自动转张量
    r2 = solve(_t(1.0), _t(1.2), st, 2, 5, [1.0, 2.0, 3.0],
               threshold=100.0, retention_ratio=0.0, max_skip_steps=100)
    assert r2["state"]["accumulated_err"] == 1.0  # idx round(1.0)=1 -> scale 2


def check_err_boundary(solve):
    """err 恰等于 threshold -> skip (官方 <=); 2 的幂精确构造."""
    n = 10
    ratios = torch.tensor([1.0] * 10, dtype=torch.float64)
    st = None
    r0 = solve(_t(1.0), _t(1.2), st, 0, n, ratios, threshold=0.5,
               retention_ratio=0.0, max_skip_steps=100)  # res None -> compute + 重置
    st = r0["state"]
    # step 1: scale=1.0 -> ratio 1.0 -> err 0 -> skip
    r1 = solve(_t(1.0), _t(1.2), st, 1, n, ratios, threshold=0.5,
               retention_ratio=0.0, max_skip_steps=100)
    st = r1["state"]
    # 构造 err 恰为 0.5: state ratio=1.0, scale=0.5 -> ratio 0.5 -> err = 0.5 恰等
    st2 = {"previous_residual": _t(0.2), "accumulated_ratio": 1.0,
           "accumulated_steps": 0, "accumulated_err": 0.0}
    ratios_h = torch.tensor([0.5] * 10, dtype=torch.float64)
    r2 = solve(_t(1.0), _t(1.2), st2, 1, n, ratios_h, threshold=0.5,
               retention_ratio=0.0, max_skip_steps=100)
    assert r2["computed"] is False and abs(r2["accumulated_err"] - 0.5) < 1e-12


def check_scale_oob_fallback(solve):
    """step >= len(ratios) (插值后仍越界: step >= N) -> scale=1.0 回退, err 恒 0."""
    n = 4
    ratios = torch.tensor([1.0] * 4, dtype=torch.float64)
    st = {"previous_residual": _t(0.2), "accumulated_ratio": 1.0,
          "accumulated_steps": 0, "accumulated_err": 0.0}
    r = solve(_t(1.0), _t(1.2), st, 6, n, ratios, threshold=0.06,
              retention_ratio=0.0, max_skip_steps=100)
    assert r["computed"] is False
    assert r["state"]["accumulated_ratio"] == 1.0
    assert r["state"]["accumulated_err"] == 0.0
    assert r["state"]["accumulated_steps"] == 1


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """张量/形状/step/N/threshold/retention/max_skip/ratios 校验."""
    ratios = torch.tensor([1.0] * 10, dtype=torch.float64)
    kw = dict(threshold=0.06, retention_ratio=0.2, max_skip_steps=3)
    assert "hidden_states 必须是 torch 张量" in _expect_err(
        solve, 1.0, _t(1.2), None, 2, 10, ratios, **kw)
    assert "tail_output 必须是 torch 张量" in _expect_err(
        solve, _t(1.0), None, None, 2, 10, ratios, **kw)
    assert "形状必须一致" in _expect_err(
        solve, _t(1.0), torch.zeros((2, 2), dtype=torch.float64), None, 2, 10, ratios, **kw)
    assert "step 必须是非负整数" in _expect_err(
        solve, _t(1.0), _t(1.2), None, -1, 10, ratios, **kw)
    assert "step 必须是非负整数" in _expect_err(
        solve, _t(1.0), _t(1.2), None, True, 10, ratios, **kw)
    assert "num_inference_steps" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 0, ratios, **kw)
    assert "threshold 必须是非负数值" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 10, ratios, threshold=-0.1,
        retention_ratio=0.2, max_skip_steps=3)
    assert "retention_ratio" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 10, ratios, retention_ratio=1.5,
        threshold=0.06, max_skip_steps=3)
    assert "retention_ratio" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 10, ratios, retention_ratio=-0.1,
        threshold=0.06, max_skip_steps=3)
    assert "max_skip_steps" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 10, ratios, max_skip_steps=0,
        threshold=0.06, retention_ratio=0.2)
    assert "mag_ratios 必须显式提供" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 10, None, **kw)
    assert "mag_ratios" in _expect_err(
        solve, _t(1.0), _t(1.2), None, 2, 10, torch.tensor([1, 2, 3], dtype=torch.int64), **kw)


def check_state_validation(solve):
    """state 结构校验."""
    ratios = torch.tensor([1.0] * 10, dtype=torch.float64)
    kw = dict(threshold=0.06)
    assert "state 必须是 dict" in _expect_err(solve, _t(1.0), _t(1.2), [], 2, 10, ratios, **kw)
    assert "未知键" in _expect_err(solve, _t(1.0), _t(1.2), {"foo": 1}, 2, 10, ratios, **kw)
    assert "缺少键" in _expect_err(
        solve, _t(1.0), _t(1.2), {"previous_residual": 0.5}, 2, 10, ratios, **kw)
    full = lambda **over: {"previous_residual": None, "accumulated_ratio": 1.0,
                           "accumulated_steps": 0, "accumulated_err": 0.0, **over}
    assert "必须是数值" in _expect_err(
        solve, _t(1.0), _t(1.2), full(accumulated_ratio="x"), 2, 10, ratios, **kw)
    assert "非负整数" in _expect_err(
        solve, _t(1.0), _t(1.2), full(accumulated_steps=True), 2, 10, ratios, **kw)
    assert "非负整数" in _expect_err(
        solve, _t(1.0), _t(1.2), full(accumulated_steps=-1), 2, 10, ratios, **kw)
    assert "必须是 torch 张量或 None" in _expect_err(
        solve, _t(1.0), _t(1.2), full(previous_residual=0.5), 2, 10, ratios, **kw)
    # ratio=1 err=0 的合法 state: skip 判定 res None -> compute
    st = {"previous_residual": None, "accumulated_ratio": 1.0,
          "accumulated_steps": 0, "accumulated_err": 0.0}
    r = solve(_t(1.0), _t(1.2), st, 2, 10, ratios, **kw)
    assert r["computed"] is True


# ---------------------------------------------------------------- 元信息
check_full_loop.example = True
check_accumulated_error.example = True
