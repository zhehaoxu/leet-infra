import math

import numpy as np
import torch


def _lam(sigma):
    """λ = −log σ（α ≡ 1）；σ = 0 → +∞（torch 语义）。"""
    if sigma == 0.0:
        return np.inf
    return -np.log(sigma)


def _exp_core(n, smin=0.3, smax=500.0):
    """官方 _compute_exponential_sigmas（独立 numpy 重写）：
    exp(linspace(log σmin, log σmax, n)).flip → [σmax … σmin]。"""
    return np.exp(np.linspace(math.log(smin), math.log(smax), n))[::-1].copy()


def _karras_core(n, smin=0.3, smax=500.0, rho=7.0):
    """官方 _compute_karras_sigmas（独立 numpy 重写）。"""
    ramp = np.linspace(0.0, 1.0, n)
    mir = smin ** (1.0 / rho)
    mar = smax ** (1.0 / rho)
    return (mar + ramp * (mir - mar)) ** rho


def _oracle(
    mo,
    sample,
    sigmas,
    step_index,
    state=None,
    noise=0.0,
    prediction_type="v_prediction",
    solver_order=2,
    solver_type="midpoint",
    lower_order_final=True,
    euler_at_final=False,
    final_sigmas_type="zero",
    sigma_data=1.0,
):
    """独立 numpy 重写 CosineDPMSolverMultistep 单调用状态机（公式从 diffusers
    scheduling_cosine_dpmsolver_multistep.py 的 step L604-686 /
    dpm_solver_first_order_update L448-487 /
    multistep_dpm_solver_second_order_update L489-548 /
    precondition_outputs L199-233 数学定义直接推出）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    n = len(s) - 1

    if state is None:
        hist = [None] * solver_order
        lo = 0
    else:
        hist = [
            None if e is None else np.asarray(e, dtype=np.float64)
            for e in state["model_outputs"]
        ]
        lo = state["lower_order_nums"]

    sigma_now = s[step_index]
    d2 = sigma_data**2
    c_skip = d2 / (sigma_now**2 + d2)
    den = np.sqrt(sigma_now**2 + d2)
    if prediction_type == "epsilon":
        c_out = sigma_now * sigma_data / den
    elif prediction_type == "v_prediction":
        c_out = -sigma_now * sigma_data / den
    else:
        raise AssertionError("oracle 只支持 epsilon / v_prediction")
    x0 = c_skip * x + c_out * mo_

    hist = hist[1:] + [x0]

    lof_flag = (step_index == n - 1) and (
        euler_at_final or (lower_order_final and n < 15) or final_sigmas_type == "zero"
    )
    use_first = solver_order == 1 or lo < 1 or lof_flag

    st = s[step_index + 1]
    ss0 = s[step_index]
    h = _lam(st) - _lam(ss0)
    noise_ = np.asarray(noise, dtype=np.float64)

    r0 = None
    d1c = None
    if use_first:
        decay = 1.0 - np.exp(-2.0 * h)
        prev = (st / ss0) * np.exp(-h) * x + decay * x0 + st * np.sqrt(decay) * noise_
        order = 1
    else:
        ss1 = s[step_index - 1]
        h0 = _lam(ss0) - _lam(ss1)
        r0 = h0 / h
        d1 = (1.0 / r0) * (x0 - hist[-2])
        decay = 1.0 - np.exp(-2.0 * h)
        d1c = 0.5 * decay if solver_type == "midpoint" else decay / (-2.0 * h) + 1.0
        prev = (st / ss0) * np.exp(-h) * x + decay * x0 + d1c * d1 + st * np.sqrt(decay) * noise_
        order = 2

    new_lo = lo + 1 if lo < solver_order else lo
    return {
        "prev_sample": prev,
        "pred_original_sample": x0,
        "used_first_order": use_first,
        "this_order": order,
        "r0": r0,
        "d1_coefficient": d1c,
        "state": {"model_outputs": hist, "lower_order_nums": new_lo},
    }


def _t(v):
    return torch.tensor([[v]], dtype=torch.float64)


def _sig_t(s):
    return torch.tensor([float(v) for v in s], dtype=torch.float64)


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
    assert out["used_first_order"] == o["used_first_order"], f"{tag} used_first_order"
    assert out["this_order"] == o["this_order"], f"{tag} this_order"
    if o["r0"] is None:
        assert out["r0"] is None, f"{tag} r0 应为 None"
    else:
        assert out["r0"] is not None and _close(out["r0"], o["r0"]), f"{tag} r0"
    if o["d1_coefficient"] is None:
        assert out["d1_coefficient"] is None, f"{tag} d1c 应为 None"
    else:
        assert out["d1_coefficient"] is not None and _close(
            out["d1_coefficient"], o["d1_coefficient"]
        ), f"{tag} d1c"
    st = out["state"]
    assert set(st.keys()) == {"model_outputs", "lower_order_nums"}, f"{tag} state keys"
    assert st["lower_order_nums"] == o["state"]["lower_order_nums"], f"{tag} lo"
    o_hist = o["state"]["model_outputs"]
    assert len(st["model_outputs"]) == len(o_hist), f"{tag} history len"
    for k in range(len(o_hist)):
        if o_hist[k] is None:
            assert st["model_outputs"][k] is None, f"{tag} history[{k}] 应为 None"
        else:
            assert _close(st["model_outputs"][k], o_hist[k]), f"{tag} history[{k}]"


def _expected_first(i, n, solver_order, lower_order_final, euler_at_final, final_sigmas_type):
    """独立重推一阶条件（不用 oracle 的返回值）。"""
    if solver_order == 1 or i == 0:
        return True
    if i == n - 1 and (
        euler_at_final or (lower_order_final and n < 15) or final_sigmas_type == "zero"
    ):
        return True
    return False


def _full_loop(
    solve,
    sigmas,
    solver_type="midpoint",
    prediction_type="v_prediction",
    solver_order=2,
    lower_order_final=True,
    euler_at_final=False,
    final_sigmas_type="zero",
    sigma_data=1.0,
    x0_start=3.0,
):
    """跑完整 N 步采样循环：伪模型 mo = 0.25x + 0.5，确定性伪噪声序列，
    每步与 oracle 对拍（state 用 solve 返回的 torch 张量原样透传），
    并做不依赖 oracle 的独立断言（lo 计数、一阶条件、末位 history = x0）。"""
    s = [float(v) for v in sigmas]
    n = len(s) - 1
    state = None
    cur = x0_start
    outs = []
    for i in range(n):
        mo = 0.25 * cur + 0.5
        noise = 0.11 * (i + 1) - 0.3
        out = solve(
            _t(mo), _t(cur), _sig_t(s), i, state=state, noise=_t(noise),
            prediction_type=prediction_type, solver_order=solver_order,
            solver_type=solver_type, lower_order_final=lower_order_final,
            euler_at_final=euler_at_final, final_sigmas_type=final_sigmas_type,
            sigma_data=sigma_data,
        )
        o = _oracle(
            mo, cur, s, i, state=state, noise=noise,
            prediction_type=prediction_type, solver_order=solver_order,
            solver_type=solver_type, lower_order_final=lower_order_final,
            euler_at_final=euler_at_final, final_sigmas_type=final_sigmas_type,
            sigma_data=sigma_data,
        )
        _assert_out_matches(out, o, f"loop step {i}")
        # 独立断言（防「错 state 对拍恒绿」盲区）
        assert out["used_first_order"] == _expected_first(
            i, n, solver_order, lower_order_final, euler_at_final, final_sigmas_type
        ), f"loop step {i} 一阶条件"
        assert out["state"]["lower_order_nums"] == min(i + 1, solver_order), (
            f"loop step {i} lower_order_nums"
        )
        assert _close(
            out["state"]["model_outputs"][-1], out["pred_original_sample"]
        ), f"loop step {i} state.model_outputs[-1]"
        state = out["state"]
        cur = float(out["prev_sample"])
        outs.append(out)
    return outs


_S6 = np.concatenate([_exp_core(6), [0.0]])  # exponential，zero 收尾，N=6


def check_first_call_defaults(solve):
    """首调（默认 v_prediction + zero 收尾）：1 阶、r0/d1c 为 None、
    state lo=1 且 history 头 None；系数用闭式独立验证。"""
    out = solve(_t(0.3), _t(2.0), _sig_t(_S6), 0, noise=_t(-0.4))
    o = _oracle(0.3, 2.0, _S6, 0, noise=-0.4)
    _assert_out_matches(out, o, "first")
    assert out["used_first_order"] is True and out["this_order"] == 1
    assert out["r0"] is None and out["d1_coefficient"] is None
    # 闭式独立验证：c_sample = (σt/σs)²、decay = 1−(σt/σs)²
    ss, st_ = float(_S6[0]), float(_S6[1])
    ratio2 = (st_ / ss) ** 2
    x0_indep = (1.0 / (ss**2 + 1.0)) * 2.0 + (-ss / math.sqrt(ss**2 + 1.0)) * 0.3
    expected = ratio2 * 2.0 + (1.0 - ratio2) * x0_indep + st_ * math.sqrt(1.0 - ratio2) * (-0.4)
    assert _close(out["prev_sample"], expected), "首调 prev 闭式"
    assert _close(out["pred_original_sample"], x0_indep), "首调 x0 闭式"
    st = out["state"]
    assert st["lower_order_nums"] == 1
    assert st["model_outputs"][0] is None
    assert _close(st["model_outputs"][1], out["pred_original_sample"])


def check_epsilon_vs_v_prediction(solve):
    """prediction_type：epsilon 的 c_out 为正、v_prediction 为负（x0 反号量级）。"""
    s = _S6
    out_e = solve(_t(0.3), _t(2.0), _sig_t(s), 0, noise=_t(-0.4), prediction_type="epsilon")
    out_v = solve(_t(0.3), _t(2.0), _sig_t(s), 0, noise=_t(-0.4), prediction_type="v_prediction")
    _assert_out_matches(out_e, _oracle(0.3, 2.0, s, 0, noise=-0.4, prediction_type="epsilon"), "eps")
    _assert_out_matches(out_v, _oracle(0.3, 2.0, s, 0, noise=-0.4, prediction_type="v_prediction"), "v")
    assert float(out_e["pred_original_sample"]) > 0.0
    assert float(out_v["pred_original_sample"]) < 0.0
    # epsilon 分支的 c_out 也依赖 sigma_data（d=2 时因子不同）
    out_e2 = solve(_t(0.3), _t(2.0), _sig_t(s), 0, noise=_t(-0.4),
                   prediction_type="epsilon", sigma_data=2.0)
    _assert_out_matches(out_e2, _oracle(0.3, 2.0, s, 0, noise=-0.4,
                                        prediction_type="epsilon", sigma_data=2.0), "eps-d2")


def check_second_order_step1_midpoint(solve):
    """step 1：二阶 midpoint 首次生效；r0 > 0、d1c = 0.5·(1−(σt/σs0)²) 闭式验证。"""
    first = solve(_t(0.3), _t(2.0), _sig_t(_S6), 0, noise=_t(-0.4))
    cur1 = float(first["prev_sample"])
    second = solve(_t(-0.2), first["prev_sample"], _sig_t(_S6), 1,
                   state=first["state"], noise=_t(0.6))
    o = _oracle(-0.2, cur1, _S6, 1, state=first["state"], noise=0.6)
    _assert_out_matches(second, o, "second")
    assert second["used_first_order"] is False and second["this_order"] == 2
    assert second["r0"] is not None and float(second["r0"]) > 0.0
    ss0, st_ = float(_S6[1]), float(_S6[2])
    ratio2 = (st_ / ss0) ** 2
    assert _close(second["d1_coefficient"], 0.5 * (1.0 - ratio2)), "midpoint d1c 闭式"
    st = second["state"]
    assert st["lower_order_nums"] == 2
    assert _close(st["model_outputs"][0], first["pred_original_sample"])
    assert _close(st["model_outputs"][1], second["pred_original_sample"])


def check_heun_step1(solve):
    """solver_type=heun：d1c = (1−(σt/σs0)²)/(−2h) + 1，且落在 (0, 1)。"""
    first = solve(_t(0.3), _t(2.0), _sig_t(_S6), 0, noise=_t(-0.4), solver_type="heun")
    cur1 = float(first["prev_sample"])
    second = solve(_t(-0.2), first["prev_sample"], _sig_t(_S6), 1,
                   state=first["state"], noise=_t(0.6), solver_type="heun")
    o = _oracle(-0.2, cur1, _S6, 1, state=first["state"], noise=0.6, solver_type="heun")
    _assert_out_matches(second, o, "heun second")
    assert second["this_order"] == 2
    ss0, st_ = float(_S6[1]), float(_S6[2])
    ratio2 = (st_ / ss0) ** 2
    h = math.log(ss0 / st_)
    assert _close(second["d1_coefficient"], (1.0 - ratio2) / (-2.0 * h) + 1.0), "heun d1c 闭式"
    d1c = float(second["d1_coefficient"])
    assert 0.0 < d1c < 1.0


def check_full_loop_default(solve):
    """示例：N=6 exponential + zero 收尾全循环（midpoint + v_prediction 默认）；
    末步 σt=0、h=+∞ → prev 恒等 x0（独立断言）。"""
    outs = _full_loop(solve, _S6)
    last = outs[-1]
    assert last["used_first_order"] is True
    assert _close(last["prev_sample"], last["pred_original_sample"]), "末步 prev == x0"
    assert outs[0]["this_order"] == 1
    for out in outs[1:-1]:
        assert out["this_order"] == 2


check_first_call_defaults.example = True
check_full_loop_default.example = True


def check_full_loop_heun_sigma_min(solve):
    """N=10 exponential + sigma_min 收尾 + heun：末步 h=0 → prev 恒等 sample
    （一阶恒等步），倒数第二步仍是二阶。"""
    sig = _exp_core(10)
    sig = np.concatenate([sig, [float(sig[-1])]])  # 末位精确复制 → 末步 h 精确为 0
    outs = _full_loop(solve, sig, solver_type="heun", final_sigmas_type="sigma_min")
    assert outs[-1]["used_first_order"] is True
    assert _close(outs[-1]["prev_sample"], outs[-2]["prev_sample"]), "末步恒等（h=0）"
    assert outs[-2]["this_order"] == 2


def check_euler_at_final(solve):
    """N=15 + sigma_min 收尾 + euler_at_final=True + lower_order_final=False：
    末步仍降 1 阶（euler 分支独立生效）。"""
    core = _exp_core(15)
    sig = np.concatenate([core, [float(core[-1])]])
    outs = _full_loop(solve, sig, solver_type="heun", lower_order_final=False,
                      euler_at_final=True, final_sigmas_type="sigma_min")
    assert outs[-1]["used_first_order"] is True
    assert outs[-2]["this_order"] == 2


def check_zero_ending_long(solve):
    """N=15 + zero 收尾 + lower_order_final=False + euler=False：末步仍降 1 阶
    （final_sigmas_type=="zero" 分支独立生效），prev 恒等 x0。"""
    sig = np.concatenate([_exp_core(15), [0.0]])
    outs = _full_loop(solve, sig, lower_order_final=False, euler_at_final=False,
                      final_sigmas_type="zero")
    assert outs[-1]["used_first_order"] is True
    assert _close(outs[-1]["prev_sample"], outs[-1]["pred_original_sample"])
    assert outs[-2]["this_order"] == 2


def check_solver_order1(solve):
    """solver_order=1：全程 1 阶、history 长度 1、lo 封顶 1。"""
    sig = np.concatenate([_exp_core(4), [0.0]])
    outs = _full_loop(solve, sig, solver_order=1)
    for out in outs:
        assert out["used_first_order"] is True and out["this_order"] == 1
        assert out["r0"] is None and out["d1_coefficient"] is None
        assert len(out["state"]["model_outputs"]) == 1


def check_karras_schedule(solve):
    """karras σ 表（EDM ρ=7）+ 默认配置全循环。"""
    sig = np.concatenate([_karras_core(6), [0.0]])
    _full_loop(solve, sig)


def check_sigma_data(solve):
    """sigma_data=2.0：c_skip/c_out 闭式独立验证（EDM 预条件随 d 变化）。"""
    d = 2.0
    ss = float(_S6[0])
    out = solve(_t(0.3), _t(2.0), _sig_t(_S6), 0, noise=_t(-0.4), sigma_data=d)
    o = _oracle(0.3, 2.0, _S6, 0, noise=-0.4, sigma_data=d)
    _assert_out_matches(out, o, "sigma_data")
    c_skip = d**2 / (ss**2 + d**2)
    c_out = -ss * d / math.sqrt(ss**2 + d**2)
    assert _close(out["pred_original_sample"], c_skip * 2.0 + c_out * 0.3)


def check_mid_schedule_fresh_start(solve):
    """官方 begin_index 场景：中途 state=None 全新开始（step 2）→ 1 阶；
    step 3 二阶时 m1 正是 step 2 的 x0（None 头被移出）。"""
    first = solve(_t(0.3), _t(2.0), _sig_t(_S6), 2, noise=_t(-0.4))
    assert first["used_first_order"] is True
    assert first["state"]["lower_order_nums"] == 1
    assert first["state"]["model_outputs"][0] is None
    cur = float(first["prev_sample"])
    second = solve(_t(-0.2), first["prev_sample"], _sig_t(_S6), 3,
                   state=first["state"], noise=_t(0.6))
    o = _oracle(-0.2, cur, _S6, 3, state=first["state"], noise=0.6)
    _assert_out_matches(second, o, "mid-schedule second")
    assert second["this_order"] == 2
    assert _close(second["state"]["model_outputs"][0], first["pred_original_sample"])


def check_second_order_zero_h_rejected(solve):
    """sigma_min 收尾 + lower_order_final=False + euler=False：末步 h=0 时
    二阶更新官方为 0/0 未定义，本题拒绝。"""
    core = _exp_core(10)
    sig = np.concatenate([core, [float(core[-1])]])
    fake_state = {
        "model_outputs": [_t(0.5), _t(0.7)],
        "lower_order_nums": 2,
    }
    try:
        solve(_t(0.3), _t(1.0), _sig_t(sig), 9, state=fake_state, noise=_t(0.2),
              lower_order_final=False, euler_at_final=False, final_sigmas_type="sigma_min")
    except ValueError as e:
        assert "相邻 λ 严格递减" in str(e), str(e)
    else:
        raise AssertionError("h=0 的二阶更新应抛 ValueError")


def check_state_reject(solve):
    """state 契约：键集、长度、lo 范围、None 头/真实尾、形状、step 0 拒 state。"""
    first = solve(_t(0.3), _t(2.0), _sig_t(_S6), 0, noise=_t(-0.4))
    good_state = first["state"]
    mo, x, sig, noise = _t(0.3), _t(2.0), _sig_t(_S6), _t(-0.4)

    def expect_err(**kw):
        try:
            solve(mo, x, sig, 1, noise=noise, **kw)
        except ValueError as e:
            return str(e)
        raise AssertionError(f"{kw} 应抛 ValueError")

    assert "state 必须是只含" in expect_err(state={**good_state, "extra": 1})
    assert "state 必须是只含" in expect_err(state={"model_outputs": good_state["model_outputs"]})
    assert "长度 2" in expect_err(state={**good_state, "model_outputs": [None, None, None]})
    assert "长度 2" in expect_err(state={"model_outputs": [None], "lower_order_nums": 1})
    assert "末 lower_order_nums 条必须是真实输出" in expect_err(
        state={**good_state, "model_outputs": [None, None]})
    bad_head = {**good_state, "model_outputs": [first["pred_original_sample"],
                                                first["pred_original_sample"]]}
    assert "移位空位必须为 None" in expect_err(state=bad_head)
    assert "必须落在 [1, min" in expect_err(state={**good_state, "lower_order_nums": 0})
    assert "必须落在" in expect_err(state={**good_state, "lower_order_nums": 3})
    assert "必须是整数" in expect_err(state={**good_state, "lower_order_nums": True})
    # lo=2 超出 [1, min(2, step_index=1)]（step 1 时最多只能有 1 次先前调用）
    assert "必须落在" in expect_err(
        state={"model_outputs": [_t(0.1), _t(0.2)], "lower_order_nums": 2})
    assert "同形状" in expect_err(
        state={**good_state, "model_outputs": [None, torch.tensor([[1.0, 2.0]], dtype=torch.float64)]})
    # step 0 给 state：lo ≥ 1 > step_index，被 lo 范围检查拒绝
    try:
        solve(mo, x, sig, 0, state=good_state, noise=noise)
    except ValueError as e:
        assert "必须落在 [1, min" in str(e), str(e)
    else:
        raise AssertionError("step 0 传 state 应抛 ValueError")


def check_validation_rejects(solve):
    """入参校验全景：张量/形状/σ 表/step_index/配置项/prediction_type。"""
    sig = _sig_t(_S6)
    mo, x, noise = _t(0.3), _t(2.0), _t(-0.4)

    def expect_err(*a, **kw):
        try:
            solve(*a, **kw)
        except ValueError as e:
            return str(e)
        raise AssertionError(f"{kw} 应抛 ValueError")

    assert "必须是 torch 张量" in expect_err(0.3, x, sig, 0, noise=noise)
    assert "形状必须一致" in expect_err(mo, torch.tensor([[1.0, 2.0]], dtype=torch.float64), sig, 0, noise=noise)
    assert "必须显式给 noise" in expect_err(mo, x, sig, 0, noise=None)
    assert "noise 形状" in expect_err(mo, x, sig, 0, noise=torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert "长度至少为 3" in expect_err(mo, x, _sig_t([4.0, 0.0]), 0, noise=noise)
    assert "全部有限" in expect_err(mo, x, _sig_t([4.0, 2.0, float("nan"), 0.0]), 0, noise=noise)
    assert "不能为负" in expect_err(mo, x, _sig_t([4.0, 2.0, -0.5]), 0, noise=noise)
    assert "除末位外必须全为正" in expect_err(mo, x, _sig_t([0.0, 2.0, 1.0]), 0, noise=noise)
    assert "单调不增" in expect_err(mo, x, _sig_t([1.0, 2.0, 3.0]), 0, noise=noise)
    assert "必须落在 [0," in expect_err(mo, x, sig, -1, noise=noise)
    assert "必须落在 [0," in expect_err(mo, x, sig, 6, noise=noise)
    assert "必须是整数" in expect_err(mo, x, sig, True, noise=noise)
    assert "必须是 1 或 2" in expect_err(mo, x, sig, 0, noise=noise, solver_order=3)
    assert "必须是整数" in expect_err(mo, x, sig, 0, noise=noise, solver_order=True)
    assert "is not implemented" in expect_err(mo, x, sig, 0, noise=noise, solver_type="logrho")
    assert "final_sigmas_type" in expect_err(mo, x, sig, 0, noise=noise, final_sigmas_type="zero_")
    assert "布尔值" in expect_err(mo, x, sig, 0, noise=noise, lower_order_final=1)
    assert "布尔值" in expect_err(mo, x, sig, 0, noise=noise, euler_at_final=1)
    assert "必须为正有限" in expect_err(mo, x, sig, 0, noise=noise, sigma_data=0)
    assert "必须为正有限" in expect_err(mo, x, sig, 0, noise=noise, sigma_data=-1.0)
    assert "必须是数值" in expect_err(mo, x, sig, 0, noise=noise, sigma_data=True)
    assert "is not supported" in expect_err(mo, x, sig, 0, noise=noise, prediction_type="sample")
