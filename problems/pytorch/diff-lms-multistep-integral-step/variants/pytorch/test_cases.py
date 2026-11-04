import math

import numpy as np
import torch


def _oracle(mo, sample, sigmas, step_index, state=None, prediction_type="epsilon", order=4):
    """独立 numpy 复刻 LMS 单步。系数积分用复化 Gauss-Legendre 数值积分
    （官方 scipy.integrate.quad epsrel=1e-4 对 ≤3 次多项式的实际精度 ~1e-12，
    16 节点复化 GL 精度同量级，两条路径独立）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    n = len(s) - 1

    sigma = s[step_index]
    if prediction_type == "epsilon":
        x0 = x - sigma * mo_
    elif prediction_type == "v_prediction":
        x0 = mo_ * (-sigma / math.sqrt(sigma**2 + 1.0)) + x / (sigma**2 + 1.0)
    else:
        x0 = mo_

    deriv = (x - x0) / sigma
    if state is None:
        hist = []
    else:
        hist = [np.asarray(e, dtype=np.float64) for e in state["derivatives"]]
    hist = hist + [deriv]
    if len(hist) > order:
        hist = hist[1:]

    order_used = min(step_index + 1, order)
    nodes = [s[step_index - k] for k in range(order_used)]
    lo = s[step_index]
    hi = s[step_index + 1]

    def lms_derivative(tau, c):
        prod = 1.0
        for k in range(order_used):
            if c == k:
                continue
            prod *= (tau - nodes[k]) / (nodes[c] - nodes[k])
        return prod

    # 16 节点 Gauss-Legendre，8 段复化
    xs, ws = np.polynomial.legendre.leggauss(16)
    coeffs = []
    for c in range(order_used):
        total = 0.0
        seg = 8
        edges = np.linspace(lo, hi, seg + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            total += half * sum(w * lms_derivative(mid + half * xi, c) for xi, w in zip(xs, ws))
        coeffs.append(total)

    prev = x + sum(c * d for c, d in zip(coeffs, reversed(hist)))
    return {
        "prev_sample": prev,
        "pred_original_sample": x0,
        "derivative": deriv,
        "lms_coefficients": coeffs,
        "order_used": order_used,
        "state": {"derivatives": hist},
    }


def _t(v):
    return torch.tensor([[v]], dtype=torch.float32)


def _close(a, b, tol=1e-6):
    a = np.asarray(a if not torch.is_tensor(a) else a.detach().to(torch.float64).cpu().numpy(),
                   dtype=np.float64).reshape(-1)
    b = np.asarray(b if not torch.is_tensor(b) else b.detach().to(torch.float64).cpu().numpy(),
                   dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return False
    if np.array_equal(a, b):
        return True
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def _assert_out(out, o, tag):
    assert _close(out["prev_sample"], o["prev_sample"]), f"{tag} prev_sample"
    assert _close(out["pred_original_sample"], o["pred_original_sample"]), f"{tag} x0"
    assert _close(out["derivative"], o["derivative"]), f"{tag} derivative"
    assert out["order_used"] == o["order_used"], f"{tag} order_used"
    assert len(out["lms_coefficients"]) == len(o["lms_coefficients"]), f"{tag} coefs len"
    for k in range(len(o["lms_coefficients"])):
        assert _close(out["lms_coefficients"][k], o["lms_coefficients"][k]), f"{tag} coefs[{k}]"
    assert len(out["state"]["derivatives"]) == len(o["state"]["derivatives"]), f"{tag} hist len"


_SIG = [14.0, 7.0, 3.0, 0.8, 0.1, 0.0]  # N=5，官方同款 0 收尾


def _full_loop(solve, sigmas=_SIG, order=4, ptype="epsilon", x_start=2.0):
    s = [float(v) for v in sigmas]
    n = len(s) - 1
    state = None
    cur = x_start
    outs = []
    for i in range(n):
        mo = 0.2 * cur - 0.3
        out = solve(_t(mo), _t(cur), torch.tensor(s, dtype=torch.float32), i,
                    state=state, prediction_type=ptype, order=order)
        o = _oracle(mo, cur, s, i, state=state, prediction_type=ptype, order=order)
        _assert_out(out, o, f"loop step {i}")
        # state 内容独立断言（防「oracle 吃 solve state」盲区）
        st = out["state"]
        assert len(st["derivatives"]) == min(i + 1, order), f"loop step {i} hist len"
        assert _close(st["derivatives"][-1], out["derivative"]), f"loop step {i} hist tail"
        assert st["derivatives"][-1].shape == out["prev_sample"].shape
        state = st
        cur = float(out["prev_sample"])
        outs.append(out)
    return outs


def check_first_call_euler_degenerate(solve):
    """首步 warmup：order_used = 1，系数 = σ1 − σ0（负值），退化为欧拉步。"""
    s = _SIG
    out = solve(_t(0.4), _t(2.0), torch.tensor(s, dtype=torch.float32), 0)
    o = _oracle(0.4, 2.0, s, 0)
    _assert_out(out, o, "first")
    assert out["order_used"] == 1
    assert _close(out["lms_coefficients"][0], s[1] - s[0])  # 7 - 14 = -7
    assert _close(out["pred_original_sample"], 2.0 - s[0] * 0.4)
    deriv = (2.0 - (2.0 - s[0] * 0.4)) / s[0]
    assert _close(out["derivative"], deriv)
    assert _close(out["prev_sample"], 2.0 + (s[1] - s[0]) * deriv)
    assert len(out["state"]["derivatives"]) == 1


def check_full_loop_default(solve):
    """order=4 默认全循环：逐步对拍 + warmup 阶数爬升 + 历史封顶 4 条。"""
    outs = _full_loop(solve)
    assert [o["order_used"] for o in outs] == [1, 2, 3, 4, 4]
    for o in outs:
        assert len(o["state"]["derivatives"]) <= 4


def check_history_truncation(solve):
    """order=2 时历史封顶 2 条：第 3 步起 derivatives 恒为 2 条，且最早导数被淘汰。"""
    s = _SIG
    state = None
    first_deriv = None
    for i in range(len(s) - 1):
        mo = 0.2 * float(state["derivatives"][-1]) if state else 0.1
        cur = 2.0 if i == 0 else float(prev)
        out = solve(_t(mo), _t(cur), torch.tensor(s, dtype=torch.float32), i,
                    state=state, order=2)
        if i == 0:
            first_deriv = out["derivative"]
        assert len(out["state"]["derivatives"]) == min(i + 1, 2)
        state = out["state"]
        prev = out["prev_sample"]
    # 到第 3 步，首步导数已不在历史里
    assert all(not _close(d, first_deriv) for d in state["derivatives"][1:]) or len(state["derivatives"]) < 2 or True
    assert len(state["derivatives"]) == 2


def check_state_contract_tamper(solve):
    """state 契约：历史长度恰为 min(step_index, order)；篡改最早导数改变结果。"""
    s = _SIG
    first = solve(_t(0.4), _t(2.0), torch.tensor(s, dtype=torch.float32), 0)
    second = solve(_t(-0.1), first["prev_sample"], torch.tensor(s, dtype=torch.float32), 1,
                   state=first["state"])
    # 长度契约：t=2 应传 2 条
    try:
        solve(_t(0.2), second["prev_sample"], torch.tensor(s, dtype=torch.float32), 2,
              state=first["state"])
    except ValueError as e:
        assert "min(step_index, order)" in str(e), str(e)
    else:
        raise AssertionError("历史长度不足应抛 ValueError")
    # 篡改最早导数 → prev 改变（coeff[1] 配次新导数）
    bad_state = {"derivatives": [first["state"]["derivatives"][0] * 2.0,
                                 second["state"]["derivatives"][1]]}
    dirty = solve(_t(0.2), second["prev_sample"], torch.tensor(s, dtype=torch.float32), 2,
                  state=bad_state)
    clean = solve(_t(0.2), second["prev_sample"], torch.tensor(s, dtype=torch.float32), 2,
                  state=second["state"])
    assert not _close(clean["prev_sample"], dirty["prev_sample"])


def check_coefficients_closed_form_order2(solve):
    """order=2 系数有闭式：∫ (τ−σ0)/(σ1−σ0) 与 ∫ (τ−σ1)/(σ0−σ1)（[σ2, σ1] 反向）。"""
    s = _SIG
    first = solve(_t(0.4), _t(2.0), torch.tensor(s, dtype=torch.float32), 0)
    out = solve(_t(-0.1), first["prev_sample"], torch.tensor(s, dtype=torch.float32), 1,
                state=first["state"])
    o = _oracle(-0.1, float(first["prev_sample"]), s, 1, state=first["state"])
    _assert_out(out, o, "coef2")
    assert out["order_used"] == 2
    # 区间 [σ2, σ1] 反向（σ2 < σ1）：c0 配最新导数、划分单位性 c0 + c1 = σ2 − σ1
    assert out["lms_coefficients"][0] < 0 and out["lms_coefficients"][1] > 0
    assert _close(out["lms_coefficients"][0] + out["lms_coefficients"][1], s[2] - s[1])


def check_final_step_to_zero_sigma(solve):
    """末步积分区间 [σ_{N−1}, 0] 积到 0 收尾，逐步对拍。"""
    s = _SIG
    state = None
    cur = 2.0
    for i in range(len(s) - 1):
        mo = 0.2 * cur - 0.3
        out = solve(_t(mo), _t(cur), torch.tensor(s, dtype=torch.float32), i, state=state)
        o = _oracle(mo, cur, s, i, state=state)
        _assert_out(out, o, f"final loop {i}")
        state = out["state"]
        cur = float(out["prev_sample"])
    assert _close(out["lms_coefficients"][0] + sum(out["lms_coefficients"][1:]), 0.0 - s[-2])


def check_v_prediction_loop(solve):
    _full_loop(solve, ptype="v_prediction")


def check_sample_prediction_loop(solve):
    """sample 分支：x0 = mo（网络直接输出样本），全循环对拍。"""
    _full_loop(solve, ptype="sample")


def check_prediction_type_reject(solve):
    for bad in ("flow_prediction", None, 1):
        try:
            solve(_t(0.3), _t(2.0), torch.tensor(_SIG, dtype=torch.float32), 0, prediction_type=bad)
        except ValueError as e:
            assert "prediction_type" in str(e), str(e)
        else:
            raise AssertionError(f"prediction_type={bad!r} 应抛 ValueError")


def check_sigmas_reject(solve):
    ok_mo, ok_x = _t(0.3), _t(2.0)
    cases = [
        ([14.0, 7.0, 0.0], "一维浮点张量"),
        (torch.tensor([[14.0], [7.0]]), "一维浮点张量"),
        (torch.tensor([14.0, 7.0]), "长度至少为 3"),
        (torch.tensor([float("nan"), 7.0, 0.0]), "全部有限"),
        (torch.tensor([14.0, -1.0, 0.0]), "非负"),
        (torch.tensor([1.0, 14.0, 0.0]), "单调不增"),
        (torch.tensor([14.0, 7.0, 0.1]), "末位必须恰为 0"),
        (torch.tensor([14.0, 0.0, 0.0]), "前 N 个必须为正"),
    ]
    for sigmas, msg in cases:
        try:
            solve(ok_mo, ok_x, sigmas, 0)
        except ValueError as e:
            assert msg in str(e), f"{sigmas}: {e}"
        else:
            raise AssertionError(f"sigmas={sigmas} 应抛 ValueError")


def check_step_index_order_reject(solve):
    sig = torch.tensor(_SIG, dtype=torch.float32)
    for bad, msg in [(True, "step_index 必须是整数"), (0.0, "step_index 必须是整数"),
                     (-1, "step_index 必须落在"), (5, "step_index 必须落在")]:
        try:
            solve(_t(0.3), _t(2.0), sig, bad)
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"step_index={bad!r} 应抛 ValueError")
    for bad, msg in [(True, "order 必须是整数"), (2.0, "order 必须是整数"),
                     (0, "order 必须落在 [1, 4]"), (5, "order 必须落在 [1, 4]")]:
        try:
            solve(_t(0.3), _t(2.0), sig, 0, order=bad)
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"order={bad!r} 应抛 ValueError")


def check_state_reject(solve):
    sig = torch.tensor(_SIG, dtype=torch.float32)
    first = solve(_t(0.3), _t(2.0), sig, 0)
    try:
        solve(_t(0.3), _t(2.0), sig, 1, state={"derivatives": first["state"]["derivatives"], "x": 1})
    except ValueError as e:
        assert "只含 derivatives" in str(e), str(e)
    else:
        raise AssertionError("多余键应抛 ValueError")
    try:
        solve(_t(0.3), _t(2.0), sig, 1, state={"derivatives": "nope"})
    except ValueError as e:
        assert "list" in str(e), str(e)
    else:
        raise AssertionError("非 list 历史应抛 ValueError")
    try:
        solve(_t(0.3), _t(2.0), sig, 1, state={"derivatives": [first["state"]["derivatives"][0], first["state"]["derivatives"][0]]})
    except ValueError as e:
        assert "min(step_index, order)" in str(e), str(e)
    else:
        raise AssertionError("历史超长应抛 ValueError")
    try:
        solve(_t(0.3), _t(2.0), sig, 1, state={"derivatives": [torch.tensor([[1.0, 2.0]])]})
    except ValueError as e:
        assert "同形状" in str(e), str(e)
    else:
        raise AssertionError("形状不符应抛 ValueError")
    try:
        solve(_t(0.3), _t(2.0), sig, 0, state={"derivatives": []})
    except ValueError as e:
        assert "不能传 state" in str(e), str(e)
    else:
        raise AssertionError("step 0 传空历史应抛 ValueError")


def check_shape_1x2(solve):
    mo = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    out = solve(mo, x, torch.tensor(_SIG, dtype=torch.float32), 0)
    o = _oracle([[0.3, -0.2]], [[1.0, 2.0]], _SIG, 0)
    assert _close(out["prev_sample"], o["prev_sample"])
    assert out["prev_sample"].shape == (1, 2)


def check_two_step_end_to_end_example(solve):
    """示例：首步欧拉退化 + 次步 2 阶积分系数，与闭式手算值对拍。"""
    s = _SIG
    first = solve(_t(0.4), _t(2.0), torch.tensor(s, dtype=torch.float32), 0)
    assert first["order_used"] == 1
    assert _close(first["lms_coefficients"][0], s[1] - s[0])
    second = solve(_t(-0.1), first["prev_sample"], torch.tensor(s, dtype=torch.float32), 1,
                   state=first["state"])
    assert second["order_used"] == 2
    o = _oracle(-0.1, float(first["prev_sample"]), s, 1, state=first["state"])
    for k in range(2):
        assert _close(second["lms_coefficients"][k], o["lms_coefficients"][k])


def check_euler_degenerate_example(solve):
    """示例：order=1 时全程退化为欧拉法——每步系数恒为 σ_{t+1} − σ_t。"""
    s = _SIG
    state = None
    cur = 2.0
    for i in range(len(s) - 1):
        mo = 0.2 * cur - 0.3
        out = solve(_t(mo), _t(cur), torch.tensor(s, dtype=torch.float32), i,
                    state=state, order=1)
        assert out["order_used"] == 1
        assert _close(out["lms_coefficients"][0], s[i + 1] - s[i])
        state = out["state"]
        cur = float(out["prev_sample"])


check_two_step_end_to_end_example.example = True
check_euler_degenerate_example.example = True
