import math

import numpy as np
import torch


def _alpha_sigma_t(sigma):
    a = 1.0 / np.sqrt(sigma**2 + 1.0)
    return a, sigma * a


def _lam(sigma):
    a, s = _alpha_sigma_t(sigma)
    return np.log(a) - np.log(s)


def _exp_pos(order, a, b, tau):
    """∫ exp(x(1+tau^2)) x^order dx，diffusers get_coefficients_exponential_positive
    的换元式（predict_x0 分支），独立 numpy 重写。"""
    c = 1.0 + tau**2
    be = c * b
    ae = c * a
    decay = np.exp(-(be - ae))
    if order == 0:
        return np.exp(be) * (1.0 - decay) / c
    if order == 1:
        return np.exp(be) * ((be - 1.0) - (ae - 1.0) * decay) / c**2
    if order == 2:
        return np.exp(be) * ((be**2 - 2.0 * be + 2.0) - (ae**2 - 2.0 * ae + 2.0) * decay) / c**3
    return (
        np.exp(be)
        * ((be**3 - 3.0 * be**2 + 6.0 * be - 6.0) - (ae**3 - 3.0 * ae**2 + 6.0 * ae - 6.0) * decay)
        / c**4
    )


def _lag(deg, lam):
    if deg == 0:
        return [[1.0]]
    if deg == 1:
        d = lam[0] - lam[1]
        return [[1.0 / d, -lam[1] / d], [1.0 / (lam[1] - lam[0]), -lam[0] / (lam[1] - lam[0])]]
    if deg == 2:
        d1 = (lam[0] - lam[1]) * (lam[0] - lam[2])
        d2 = (lam[1] - lam[0]) * (lam[1] - lam[2])
        d3 = (lam[2] - lam[0]) * (lam[2] - lam[1])
        return [
            [1.0 / d1, (-lam[1] - lam[2]) / d1, lam[1] * lam[2] / d1],
            [1.0 / d2, (-lam[0] - lam[2]) / d2, lam[0] * lam[2] / d2],
            [1.0 / d3, (-lam[0] - lam[1]) / d3, lam[0] * lam[1] / d3],
        ]
    d1 = (lam[0] - lam[1]) * (lam[0] - lam[2]) * (lam[0] - lam[3])
    d2 = (lam[1] - lam[0]) * (lam[1] - lam[2]) * (lam[1] - lam[3])
    d3 = (lam[2] - lam[0]) * (lam[2] - lam[1]) * (lam[2] - lam[3])
    d4 = (lam[3] - lam[0]) * (lam[3] - lam[1]) * (lam[3] - lam[2])
    return [
        [1.0 / d1, (-lam[1] - lam[2] - lam[3]) / d1,
         (lam[1] * lam[2] + lam[1] * lam[3] + lam[2] * lam[3]) / d1,
         (-lam[1] * lam[2] * lam[3]) / d1],
        [1.0 / d2, (-lam[0] - lam[2] - lam[3]) / d2,
         (lam[0] * lam[2] + lam[0] * lam[3] + lam[2] * lam[3]) / d2,
         (-lam[0] * lam[2] * lam[3]) / d2],
        [1.0 / d3, (-lam[0] - lam[1] - lam[3]) / d3,
         (lam[0] * lam[1] + lam[0] * lam[3] + lam[1] * lam[3]) / d3,
         (-lam[0] * lam[1] * lam[2]) / d3],
        [1.0 / d4, (-lam[0] - lam[1] - lam[2]) / d4,
         (lam[0] * lam[1] + lam[0] * lam[2] + lam[1] * lam[2]) / d4,
         (-lam[0] * lam[1] * lam[2]) / d4],
    ]


def _coefs(order, a, b, nodes, tau):
    lag = _lag(order - 1, nodes)
    return [
        sum(lag[i][j] * _exp_pos(order - 1 - j, a, b, tau) for j in range(order))
        for i in range(order)
    ]


def _oracle(
    mo,
    sample,
    sigmas,
    step_index,
    state=None,
    noise=0.0,
    tau=0.0,
    tau_corrector=None,
    predictor_order=2,
    corrector_order=2,
    lower_order_final=True,
    prediction_type="epsilon",
):
    """独立 numpy 重写 SA-Solver 单调用状态机（公式从 diffusers
    scheduling_sasolver.py 的 step L1262-1362 / stochastic_adams_bashforth_update
    L999-1070 / stochastic_adams_moulton_update L1139-1206 数学定义直接推出）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    n = len(s) - 1

    def conv(o, xx, sigma):
        a, st = _alpha_sigma_t(sigma)
        if prediction_type == "epsilon":
            return (xx - st * o) / a
        return a * xx - st * o

    x0 = conv(mo_, x, s[step_index])
    m = max(predictor_order, corrector_order - 1)
    if state is None:
        hist = [None] * m
        last_sample = last_noise = None
        lo = 0
        use_corr = False
        saved_tc = None
    else:
        hist = [None if e is None else np.asarray(e, dtype=np.float64)
                for e in state["model_outputs"]]
        last_sample = np.asarray(state["last_sample"], dtype=np.float64)
        last_noise = np.asarray(state["last_noise"], dtype=np.float64)
        lo = state["lower_order_nums"]
        saved_tc = state["this_corrector_order"]
        use_corr = True

    corrected = None
    corr_coefs = None
    if use_corr:
        tc = saved_tc
        lt = _lam(s[step_index])
        ls0 = _lam(s[step_index - 1])
        h = lt - ls0
        nodes = [_lam(s[step_index - k]) for k in range(tc)]
        co = _coefs(tc, ls0, lt, nodes, tau_corrector)
        if tc == 2:
            d = (
                np.exp((1 + tau_corrector**2) * lt)
                * (
                    h / 2.0
                    - (h * (1 + tau_corrector**2) - 1.0 + np.exp((1 + tau_corrector**2) * (-h)))
                    / ((1 + tau_corrector**2) ** 2 * h)
                )
            )
            co[0] += d
            co[1] -= d
        corr_coefs = co
        _, st = _alpha_sigma_t(s[step_index])
        _, ss0 = _alpha_sigma_t(s[step_index - 1])
        mp = hist + [x0]
        g = np.zeros_like(x)
        for k in range(tc):
            g = g + (1 + tau_corrector**2) * st * np.exp(-(tau_corrector**2) * lt) * co[k] * mp[-(k + 1)]
        npart = st * np.sqrt(1.0 - np.exp(-2.0 * tau_corrector**2 * h)) * last_noise
        corrected = np.exp(-(tau_corrector**2) * h) * (st / ss0) * last_sample + g + npart

    hist = hist[1:] + [x0]
    if lower_order_final:
        tp = min(predictor_order, n - step_index)
        tcn = min(corrector_order, n - step_index + 1)
    else:
        tp = predictor_order
        tcn = corrector_order
    tp = min(tp, lo + 1)
    tcn = min(tcn, lo + 2)

    lt = _lam(s[step_index + 1])
    ls0 = _lam(s[step_index])
    h = lt - ls0
    nodes = [_lam(s[step_index - k]) for k in range(tp)]
    co = _coefs(tp, ls0, lt, nodes, tau)
    if tp == 2:
        d = (
            np.exp((1 + tau**2) * lt)
            * (
                h**2 / 2.0
                - (h * (1 + tau**2) - 1.0 + np.exp((1 + tau**2) * (-h))) / (1 + tau**2) ** 2
            )
            / (ls0 - _lam(s[step_index - 1]))
        )
        co[0] += d
        co[1] -= d
    _, st = _alpha_sigma_t(s[step_index + 1])
    _, ss0 = _alpha_sigma_t(s[step_index])
    g = np.zeros_like(x)
    for k in range(tp):
        g = g + (1 + tau**2) * st * np.exp(-(tau**2) * lt) * co[k] * hist[-(k + 1)]
    npart = st * np.sqrt(1.0 - np.exp(-2.0 * tau**2 * h)) * np.asarray(noise, dtype=np.float64)
    base = corrected if use_corr else x
    prev = np.exp(-(tau**2) * h) * (st / ss0) * base + g + npart

    new_lo = lo + 1 if lo < m else lo
    o_state = {
        "model_outputs": hist,
        "last_sample": base,
        "last_noise": np.asarray(noise, dtype=np.float64),
        "lower_order_nums": new_lo,
        "this_corrector_order": tcn,
    }
    return {
        "prev_sample": prev,
        "pred_original_sample": x0,
        "corrected_sample": corrected,
        "used_corrector": use_corr,
        "this_predictor_order": tp,
        "this_corrector_order": tcn,
        "predictor_coefficients": co,
        "corrector_coefficients": corr_coefs,
        "state": o_state,
    }


def _t(v):
    return torch.tensor([[v]], dtype=torch.float32)


def _as_np(v):
    if torch.is_tensor(v):
        return v.detach().to(torch.float64).cpu().numpy()
    return np.asarray(v, dtype=np.float64)


def _close(a, b, tol=1e-6):
    a = np.asarray(_as_np(a), dtype=np.float64).reshape(-1)
    b = np.asarray(_as_np(b), dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    if np.array_equal(a, b):
        return True
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def _assert_out_matches(out, o, tag):
    assert _close(out["prev_sample"], o["prev_sample"]), f"{tag} prev_sample"
    assert _close(out["pred_original_sample"], o["pred_original_sample"]), f"{tag} x0"
    assert out["used_corrector"] == o["used_corrector"], f"{tag} used_corrector"
    assert out["this_predictor_order"] == o["this_predictor_order"], f"{tag} tp"
    assert out["this_corrector_order"] == o["this_corrector_order"], f"{tag} tc"
    if o["corrected_sample"] is None:
        assert out["corrected_sample"] is None, f"{tag} corrected 应为 None"
    else:
        assert out["corrected_sample"] is not None, f"{tag} corrected 不应为 None"
        assert _close(out["corrected_sample"], o["corrected_sample"]), f"{tag} corrected"
    assert len(out["predictor_coefficients"]) == o["this_predictor_order"], f"{tag} coefs len"
    for k in range(o["this_predictor_order"]):
        assert _close(out["predictor_coefficients"][k], o["predictor_coefficients"][k]), (
            f"{tag} predictor coefs[{k}]"
        )
    if o["corrector_coefficients"] is None:
        assert out["corrector_coefficients"] is None, f"{tag} corrector coefs 应为 None"
    else:
        assert len(out["corrector_coefficients"]) == len(o["corrector_coefficients"])
        for k in range(len(o["corrector_coefficients"])):
            assert _close(out["corrector_coefficients"][k], o["corrector_coefficients"][k]), (
                f"{tag} corrector coefs[{k}]"
            )


_SIG = [8.0, 4.0, 2.0, 0.5, 0.2]  # N=4，全正（官方各 schedule 均无 0 收尾）


def _full_loop(solve, sigmas=_SIG, orders=(2, 2), lof=True, ptype="epsilon",
               taus=None, tau_cs=None, x0_start=3.0):
    """跑完整 N 步采样循环：伪模型 eps = 0.25x + 0.5，确定性伪噪声序列，
    每步与 oracle 对拍（state 用 solve 返回的 torch 张量原样透传）。"""
    s = [float(v) for v in sigmas]
    n = len(s) - 1
    m_order, c_order = orders
    state = None
    cur = x0_start
    outs = []
    for i in range(n):
        mo = 0.25 * cur + 0.5
        noise = 0.11 * (i + 1) - 0.3
        tau = taus[i] if taus is not None else 0.0
        tau_c = tau_cs[i] if tau_cs is not None else None
        out = solve(_t(mo), _t(cur), torch.tensor(s, dtype=torch.float32), i,
                    state=state, noise=_t(noise), tau=tau, tau_corrector=tau_c,
                    predictor_order=m_order, corrector_order=c_order,
                    lower_order_final=lof, prediction_type=ptype)
        o = _oracle(mo, cur, s, i, state=state, noise=noise, tau=tau, tau_corrector=tau_c,
                    predictor_order=m_order, corrector_order=c_order,
                    lower_order_final=lof, prediction_type=ptype)
        _assert_out_matches(out, o, f"loop step {i}")
        st = out["state"]
        assert set(st.keys()) == {
            "model_outputs", "last_sample", "last_noise", "lower_order_nums",
            "this_corrector_order",
        }
        assert len(st["model_outputs"]) == max(m_order, c_order - 1)
        assert st["lower_order_nums"] == o["state"]["lower_order_nums"]
        # 独立验证 state 内容（oracle 不吃 solve 的 state，防「错 state 对拍恒绿」盲区）：
        # last_sample = 本步 corrector 输出（无 corrector 时 = 本步输入样本）、
        # last_noise = 本步入参噪声、model_outputs 末位 = 本步 x0。
        assert _close(st["last_sample"],
                      out["corrected_sample"] if out["used_corrector"] else cur), (
            f"loop step {i} state.last_sample")
        assert _close(st["last_noise"], noise), f"loop step {i} state.last_noise"
        assert _close(st["model_outputs"][-1], out["pred_original_sample"]), (
            f"loop step {i} state.model_outputs[-1]")
        state = st
        cur = float(out["prev_sample"])
        outs.append(out)
    return outs


def check_first_call_warmup(solve):
    """首调：无 corrector、predictor warmup 到 1 阶、state 存本轮样本与噪声。"""
    out = solve(_t(0.3), _t(2.0), torch.tensor(_SIG, dtype=torch.float32), 0,
                noise=_t(-0.4), tau=0.0)
    o = _oracle(0.3, 2.0, _SIG, 0, noise=-0.4)
    _assert_out_matches(out, o, "first")
    assert out["used_corrector"] is False
    assert out["corrected_sample"] is None
    assert out["corrector_coefficients"] is None
    assert out["this_predictor_order"] == 1
    assert out["this_corrector_order"] == 2
    st = out["state"]
    assert st["lower_order_nums"] == 1
    assert st["model_outputs"][0] is None
    assert _close(st["model_outputs"][1], o["state"]["model_outputs"][1])
    assert _close(st["last_sample"], 2.0)
    assert _close(st["last_noise"], -0.4)


def check_corrector_step1(solve):
    """step 1：corrector 首次生效（用上一步暂存的样本/噪声），predictor 升到 2 阶。"""
    first = solve(_t(0.3), _t(2.0), torch.tensor(_SIG, dtype=torch.float32), 0,
                  noise=_t(-0.4))
    out = solve(_t(-0.2), first["prev_sample"], torch.tensor(_SIG, dtype=torch.float32), 1,
                state=first["state"], noise=_t(0.6), tau=0.5, tau_corrector=0.0)
    o = _oracle(-0.2, float(first["prev_sample"]), _SIG, 1,
                state=first["state"], noise=0.6, tau=0.5, tau_corrector=0.0)
    _assert_out_matches(out, o, "step1")
    assert out["used_corrector"] is True
    assert out["this_predictor_order"] == 2
    assert out["this_corrector_order"] == 2
    # 2 阶修正项让 coefs[0] + coefs[1] 不再等于平凡和
    assert out["predictor_coefficients"][0] != out["predictor_coefficients"][1]


def check_full_loop_default(solve):
    """默认 2/2 阶 + lower_order_final：完整 4 步循环逐步对拍；末步 predictor 降到 1 阶。"""
    outs = _full_loop(solve, taus=[0.0, 1.0, 1.0, 0.0], tau_cs=[None, 0.0, 1.0, 1.0])
    assert outs[0]["this_predictor_order"] == 1
    assert outs[1]["this_predictor_order"] == 2
    assert outs[-1]["this_predictor_order"] == 1  # 末步必降 1 阶
    assert outs[-1]["this_corrector_order"] == 2


def check_full_loop_tau0_ode(solve):
    """tau = 0 的 ODE 路径：噪声完全不起作用（两套不同噪声结果逐位一致）。"""
    s = _SIG
    runs = []
    for noise_base in (0.7, -2.0):
        state = None
        cur = 3.0
        for i in range(len(s) - 1):
            mo = 0.25 * cur + 0.5
            out = solve(_t(mo), _t(cur), torch.tensor(s, dtype=torch.float32), i,
                        state=state, noise=_t(noise_base * (i + 1)), tau=0.0,
                        tau_corrector=0.0 if state is not None else None)
            state = out["state"]
            cur = float(out["prev_sample"])
        runs.append(cur)
    assert runs[0] == runs[1]


def check_full_loop_orders34(solve):
    """3/3 与 4/4 阶：覆盖 Lagrange 2/3 次与 3 阶指数积分的完整循环。"""
    _full_loop(solve, orders=(3, 3), taus=[0.0, 1.0, 1.0, 0.0],
               tau_cs=[None, 0.0, 1.0, 1.0])
    _full_loop(solve, orders=(4, 4), taus=[1.0, 0.0, 1.0, 1.0],
               tau_cs=[None, 1.0, 0.0, 1.0])


def check_lower_order_final_false(solve):
    """lower_order_final = False：末步不再降阶（保持 2 阶），逐步对拍。"""
    outs = _full_loop(solve, lof=False, taus=[0.0, 1.0, 0.0, 1.0],
                      tau_cs=[None, 0.0, 1.0, 0.0])
    assert outs[-1]["this_predictor_order"] == 2
    assert outs[-1]["this_corrector_order"] == 2


def check_v_prediction_loop(solve):
    """v_prediction 全循环对拍（convert 走 alpha·sample − sigma_t·mo 分支）。"""
    _full_loop(solve, ptype="v_prediction", taus=[0.0, 1.0, 1.0, 0.0],
               tau_cs=[None, 0.0, 1.0, 1.0])


def check_midchain_fresh_start(solve):
    """中段冷启动（image-to-image 同款）：state=None 但 step_index > 0 时
    不跑 corrector，predictor warmup 从 1 阶重新开始。"""
    out = solve(_t(0.4), _t(1.0), torch.tensor(_SIG, dtype=torch.float32), 2,
                noise=_t(0.3), tau=0.0)
    o = _oracle(0.4, 1.0, _SIG, 2, noise=0.3)
    _assert_out_matches(out, o, "midchain")
    assert out["used_corrector"] is False
    assert out["this_predictor_order"] == 1


def check_duplicate_sigma_tail_identity(solve):
    """karras 式末位相邻相等：末步 h = 0 → 系数与噪声项全为 0，prev 恒等于
    corrector 输出（或原样本），全循环对拍不炸。"""
    s = [8.0, 4.0, 1.0, 0.5, 0.5]
    outs = _full_loop(solve, sigmas=s, taus=[1.0, 1.0, 1.0, 0.0],
                      tau_cs=[None, 1.0, 1.0, 1.0])
    last = outs[-1]
    assert _close(last["prev_sample"], last["corrected_sample"])
    assert _close(last["predictor_coefficients"][0], 0.0)


def check_state_history_shape(solve):
    """非标量 (1,2) 形状端到端一步：形状校验与逐元素对拍。"""
    mo = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    noise = torch.tensor([[0.5, 0.1]], dtype=torch.float32)
    out = solve(mo, x, torch.tensor(_SIG, dtype=torch.float32), 0, noise=noise)
    o = _oracle([[0.3, -0.2]], [[1.0, 2.0]], _SIG, 0, noise=[[0.5, 0.1]])
    _assert_out_matches(out, o, "shape")
    assert out["prev_sample"].shape == (1, 2)


def check_state_contract_last_noise_matters(solve):
    """state 契约：corrector 真的消费上一步的 last_noise 与暂存阶数——
    篡改任一项都会改变 corrected/prev。"""
    first = solve(_t(0.3), _t(2.0), torch.tensor(_SIG, dtype=torch.float32), 0,
                  noise=_t(-0.4))
    clean = solve(_t(-0.2), first["prev_sample"], torch.tensor(_SIG, dtype=torch.float32), 1,
                  state=first["state"], noise=_t(0.6), tau=0.5, tau_corrector=1.0)
    bad_state = dict(first["state"])
    bad_state["last_noise"] = first["state"]["last_noise"] * 3.0
    dirty = solve(_t(-0.2), first["prev_sample"], torch.tensor(_SIG, dtype=torch.float32), 1,
                  state=bad_state, noise=_t(0.6), tau=0.5, tau_corrector=1.0)
    assert not _close(clean["corrected_sample"], dirty["corrected_sample"])
    assert not _close(clean["prev_sample"], dirty["prev_sample"])
    bad_tc = dict(first["state"])
    bad_tc["this_corrector_order"] = 1
    lower = solve(_t(-0.2), first["prev_sample"], torch.tensor(_SIG, dtype=torch.float32), 1,
                  state=bad_tc, noise=_t(0.6), tau=0.5, tau_corrector=1.0)
    assert not _close(clean["corrected_sample"], lower["corrected_sample"])


def check_prediction_type_reject(solve):
    for bad in ("sample", "flow_prediction", None, 0):
        try:
            solve(_t(0.3), _t(2.0), torch.tensor(_SIG, dtype=torch.float32), 0,
                  noise=_t(0.1), prediction_type=bad)
        except ValueError as e:
            assert "prediction_type" in str(e), str(e)
        else:
            raise AssertionError(f"prediction_type={bad!r} 应抛 ValueError")


def check_sigmas_reject(solve):
    ok_mo, ok_x, ok_noise = _t(0.3), _t(2.0), _t(0.1)
    cases = [
        ([8.0, 4.0, 2.0, 0.5, 0.2], "一维浮点张量"),   # list 不是张量
        (torch.tensor([[8.0], [4.0]]), "一维浮点张量"),
        (torch.tensor([8.0, 4.0]), "长度至少为 3"),
        (torch.tensor([float("nan"), 4.0, 2.0, 0.5, 0.2]), "全部有限"),
        (torch.tensor([8.0, 0.5, 0.0]), "全为正"),
        (torch.tensor([1.0, 4.0, 2.0]), "单调不增"),
        (torch.tensor([8.0, -0.5, -1.0]), "全为正"),
    ]
    for sigmas, msg in cases:
        try:
            solve(ok_mo, ok_x, sigmas, 0, noise=ok_noise)
        except ValueError as e:
            assert msg in str(e), f"{sigmas}: {e}"
        else:
            raise AssertionError(f"sigmas={sigmas} 应抛 ValueError")


def check_step_index_reject(solve):
    ok = (torch.tensor(_SIG, dtype=torch.float32),)
    for bad, msg in [(True, "step_index 必须是整数"), (1.0, "step_index 必须是整数"),
                     (-1, "step_index 必须落在"), (4, "step_index 必须落在")]:
        try:
            solve(_t(0.3), _t(2.0), *ok, bad, noise=_t(0.1))
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"step_index={bad!r} 应抛 ValueError")


def check_orders_flags_reject(solve):
    base = dict(noise=_t(0.1))
    for kw, msg in [
        (dict(predictor_order=0), "predictor_order 必须落在 [1, 4]"),
        (dict(predictor_order=5), "predictor_order 必须落在 [1, 4]"),
        (dict(predictor_order=True), "predictor_order 必须是整数"),
        (dict(predictor_order=2.0), "predictor_order 必须是整数"),
        (dict(corrector_order=0), "corrector_order 必须落在 [1, 4]"),
        (dict(corrector_order=5), "corrector_order 必须落在 [1, 4]"),
        (dict(corrector_order=1.5), "corrector_order 必须是整数"),
        (dict(lower_order_final=1), "lower_order_final 必须是布尔值"),
    ]:
        try:
            solve(_t(0.3), _t(2.0), torch.tensor(_SIG, dtype=torch.float32), 0, **base, **kw)
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"{kw} 应抛 ValueError")


def check_noise_tau_reject(solve):
    sig = torch.tensor(_SIG, dtype=torch.float32)
    try:
        solve(_t(0.3), _t(2.0), sig, 0)
    except ValueError as e:
        assert "必须显式给 noise" in str(e), str(e)
    else:
        raise AssertionError("缺 noise 应抛 ValueError")
    try:
        solve(_t(0.3), _t(2.0), sig, 0, noise=torch.tensor([[0.1, 0.2]], dtype=torch.float32))
    except ValueError as e:
        assert "noise 形状" in str(e), str(e)
    else:
        raise AssertionError("noise 形状不符应抛 ValueError")
    for bad, msg in [(-0.1, "tau 必须非负"), (True, "tau 必须是数值"),
                     (float("nan"), "tau 必须有限"), ("1", "tau 必须是数值")]:
        try:
            solve(_t(0.3), _t(2.0), sig, 0, noise=_t(0.1), tau=bad)
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"tau={bad!r} 应抛 ValueError")


def check_state_reject(solve):
    sig = torch.tensor(_SIG, dtype=torch.float32)
    first = solve(_t(0.3), _t(2.0), sig, 0, noise=_t(-0.4))
    good_state = first["state"]
    noise = _t(0.6)
    mo, x = _t(-0.2), first["prev_sample"]

    def expect_err(**kw):
        try:
            solve(mo, x, sig, 1, noise=noise, tau=0.0, tau_corrector=0.0, **kw)
        except ValueError as e:
            return str(e)
        raise AssertionError(f"{kw} 应抛 ValueError")

    assert "state 必须是只含" in expect_err(state={**good_state, "extra": 1})
    assert "state 必须是只含" in expect_err(state={"model_outputs": good_state["model_outputs"]})
    assert "长度 2" in expect_err(state={**good_state, "model_outputs": [None, None, None]})
    bad_tail = {**good_state, "model_outputs": [None, None]}
    assert "末 lower_order_nums 条必须是真实输出" in expect_err(state=bad_tail)
    bad_head = {**good_state, "model_outputs": [first["pred_original_sample"], None]}
    assert "移位空位必须为 None" in expect_err(state=bad_head)
    assert "必须落在" in expect_err(state={**good_state, "lower_order_nums": 3})
    assert "必须落在" in expect_err(state={**good_state, "lower_order_nums": -1})
    assert "必须是整数" in expect_err(state={**good_state, "lower_order_nums": True})
    assert "必须落在 [1, 2]" in expect_err(
        state={**good_state, "this_corrector_order": 3})
    assert "同形状" in expect_err(state={**good_state, "last_sample": torch.tensor([[1.0, 2.0]])})
    assert "同形状" in expect_err(state={**good_state, "last_noise": torch.tensor([[1.0, 2.0]])})
    try:
        solve(mo, x, sig, 0, state=good_state, noise=noise, tau=0.0, tau_corrector=0.0)
    except ValueError as e:
        assert "不能传 state" in str(e), str(e)
    else:
        raise AssertionError("step 0 传 state 应抛 ValueError")
    try:
        solve(mo, x, sig, 1, state=good_state, noise=noise, tau=0.0)
    except ValueError as e:
        assert "tau_corrector" in str(e), str(e)
    else:
        raise AssertionError("缺 tau_corrector 应抛 ValueError")


def check_two_step_end_to_end_example(solve):
    """示例：前两步端到端——首调 warmup 1 阶无 corrector，次调 corrector 生效。"""
    sig = torch.tensor(_SIG, dtype=torch.float32)
    first = solve(_t(0.3), _t(2.0), sig, 0, noise=_t(-0.4))
    assert first["used_corrector"] is False and first["this_predictor_order"] == 1
    second = solve(_t(-0.2), first["prev_sample"], sig, 1, state=first["state"],
                   noise=_t(0.6), tau=0.5, tau_corrector=0.0)
    assert second["used_corrector"] is True and second["this_predictor_order"] == 2
    assert _close(second["prev_sample"],
                  _oracle(-0.2, float(first["prev_sample"]), _SIG, 1,
                          state=first["state"], noise=0.6, tau=0.5, tau_corrector=0.0)["prev_sample"])


def check_final_step_drop_example(solve):
    """示例：3 步表 [8, 4, 1, 0.2] 全循环，末步 predictor 必降到 1 阶。"""
    s = [8.0, 4.0, 1.0, 0.2]
    state = None
    cur = 1.5
    for i in range(3):
        out = solve(_t(0.25 * cur + 0.5), _t(cur), torch.tensor(s, dtype=torch.float32), i,
                    state=state, noise=_t(0.2 - 0.1 * i), tau=0.0,
                    tau_corrector=0.0 if state is not None else None)
        state = out["state"]
        cur = float(out["prev_sample"])
    assert out["this_predictor_order"] == 1


check_two_step_end_to_end_example.example = True
check_final_step_drop_example.example = True
