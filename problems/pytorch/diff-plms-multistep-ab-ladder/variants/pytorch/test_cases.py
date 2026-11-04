import math

import numpy as np
import torch


def _oracle(mo, sample, alphas, timestep, step_ratio, counter, state=None,
            prediction_type="epsilon"):
    """独立 numpy 复刻 PLMS 单步（公式从 scheduling_pndm.py 的 step_plms
    L324-392 与 _get_prev_sample L409-458 数学定义直接推出）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    a = np.asarray(alphas, dtype=np.float64)

    if counter == 0:
        ets = []
        cur_sample = None
    else:
        ets = [np.asarray(e, dtype=np.float64) for e in state["ets"]]
        cur_sample = None if state["cur_sample"] is None else np.asarray(state["cur_sample"], dtype=np.float64)

    if counter != 1:
        ets = ets[-3:]
        ets = ets + [mo_]

    if len(ets) == 1 and counter == 0:
        combined = mo_
        cur_sample = x
    elif len(ets) == 1 and counter == 1:
        combined = (mo_ + ets[-1]) / 2
        x = cur_sample
        cur_sample = None
    elif len(ets) == 2:
        combined = (3 * ets[-1] - ets[-2]) / 2
    elif len(ets) == 3:
        combined = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        combined = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    if counter == 1:
        prev_timestep = timestep
        timestep = timestep + step_ratio
    else:
        prev_timestep = timestep - step_ratio

    alpha_t = a[timestep]
    alpha_prev = a[prev_timestep] if prev_timestep >= 0 else a[0]
    beta_t = 1.0 - alpha_t
    beta_prev = 1.0 - alpha_prev

    eps = combined
    if prediction_type == "v_prediction":
        eps = math.sqrt(alpha_t) * combined + math.sqrt(beta_t) * x

    sample_coeff = math.sqrt(alpha_prev / alpha_t)
    denom = alpha_t * math.sqrt(beta_prev) + math.sqrt(alpha_t * beta_t * alpha_prev)
    prev = sample_coeff * x - (alpha_prev - alpha_t) * eps / denom

    return {
        "prev_sample": prev,
        "model_output_combined": combined,
        "prev_timestep": prev_timestep,
        "state": {"ets": ets, "cur_sample": cur_sample},
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
    assert _close(out["model_output_combined"], o["model_output_combined"]), f"{tag} combined"
    assert out["prev_timestep"] == o["prev_timestep"], f"{tag} prev_timestep"
    assert len(out["state"]["ets"]) == len(o["state"]["ets"]), f"{tag} ets len"
    for k in range(len(o["state"]["ets"])):
        assert _close(out["state"]["ets"][k], o["state"]["ets"][k]), f"{tag} ets[{k}]"
    if o["state"]["cur_sample"] is None:
        assert out["state"]["cur_sample"] is None, f"{tag} cur_sample 应为 None"
    else:
        assert _close(out["state"]["cur_sample"], o["state"]["cur_sample"]), f"{tag} cur_sample"


# 官方 linear β 表（T=1000）的真实 ᾱ：β_t 从 1e-4 线性升到 0.02，ᾱ(0) = 1−β_0 ≈ 0.9999
_BETAS = np.linspace(1e-4, 0.02, 1000)
_ALPHAS = np.cumprod(1.0 - _BETAS)
# SD 的 skip_prk_steps 流水线（N=5, T=1000）：
# _timesteps = linspace(0, 999, 5).round() = [0, 250, 500, 750, 999]
# 表 = concat([:-1], [-2:-1], [-1:])[::-1] = [999, 500, 500, 250, 0]，δ = 200
_TIMESTEPS = [999, 500, 500, 250, 0]
_DELTA = 200
# 调用后 ets 长度（官方记账：c0 append、c1 不 append、c≥2 截断[-3:]+append）
_ETS_LEN_AFTER = [1, 1, 2, 3, 4]


def _full_loop(solve, timesteps=_TIMESTEPS, delta=_DELTA, alphas=None,
               ptype="epsilon", x_start=2.0):
    alphas = _ALPHAS if alphas is None else alphas
    state = None
    cur = x_start
    last_mo = None
    outs = []
    for counter, t in enumerate(timesteps):
        mo = 0.3 * math.sqrt(float(alphas[t])) + 0.05 * counter  # 伪模型（依赖噪声水平）
        out = solve(_t(mo), _t(cur), torch.tensor(alphas, dtype=torch.float32), t,
                    delta, counter, state=state, prediction_type=ptype)
        o = _oracle(mo, cur, alphas, t, delta, counter, state=state, prediction_type=ptype)
        _assert_out(out, o, f"loop counter {counter}")
        # state 内容独立断言（oracle 不吃 solve 的 state）
        st = out["state"]
        assert len(st["ets"]) == _ETS_LEN_AFTER[counter], f"loop counter {counter} ets len"
        if counter == 1:
            assert _close(st["ets"][-1], last_mo), "counter 1 不 append，ets 尾仍是上一轮输出"
        else:
            assert _close(st["ets"][-1], mo), f"loop counter {counter} ets tail 应为本轮原始输出"
        if counter == 0:
            assert _close(st["cur_sample"], x_start), "counter 0 应暂存本轮样本"
        else:
            assert st["cur_sample"] is None, f"counter {counter} 应清空 cur_sample"
        state = st
        cur = float(out["prev_sample"])
        last_mo = mo
        outs.append(out)
    return outs


def check_counter0_identity_and_stash(solve):
    """counter 0：ets=[mo]（组合=恒等）、cur_sample 暂存、prev_timestep = t−δ。"""
    out = solve(_t(0.3), _t(2.0), torch.tensor(_ALPHAS, dtype=torch.float32), 999, 200, 0)
    o = _oracle(0.3, 2.0, _ALPHAS, 999, 200, 0)
    _assert_out(out, o, "c0")
    assert _close(out["model_output_combined"], 0.3)
    assert _close(out["state"]["cur_sample"], 2.0)
    assert out["prev_timestep"] == 799
    assert len(out["state"]["ets"]) == 1


def check_counter1_trapezoid_restart(solve):
    """counter 1：不 append、梯形组合 (mo0+mo1)/2、从 cur_sample 重启、prev = t。"""
    first = solve(_t(0.3), _t(2.0), torch.tensor(_ALPHAS, dtype=torch.float32), 500, 200, 0)
    out = solve(_t(-0.2), first["prev_sample"], torch.tensor(_ALPHAS, dtype=torch.float32),
                500, 200, 1, state=first["state"])
    o = _oracle(-0.2, float(first["prev_sample"]), _ALPHAS, 500, 200, 1, state=first["state"])
    _assert_out(out, o, "c1")
    assert _close(out["model_output_combined"], 0.05)  # (0.3 + (-0.2)) / 2
    assert out["prev_timestep"] == 500  # (timestep, prev) = (700, 500)
    assert out["state"]["cur_sample"] is None
    assert len(out["state"]["ets"]) == 1  # counter 1 不 append


def check_ab_ladder_full_loop(solve):
    """全循环：AB 阶梯逐级爬升（恒等→梯形→2/3/4 阶），逐步对拍。"""
    outs = _full_loop(solve)
    assert [len(o["state"]["ets"]) for o in outs] == _ETS_LEN_AFTER


def check_ets_cap_and_content(solve):
    """ets 封顶：counter ≥ 4 后历史恒 4 条，最早输出被挤出（独立断言）。"""
    state = None
    first_mo = None
    seq = _TIMESTEPS + [250, 500]
    for counter, t in enumerate(seq):
        mo = 0.1 + 0.05 * counter
        out = solve(_t(mo), _t(2.0 if counter == 0 else 1.0),
                    torch.tensor(_ALPHAS, dtype=torch.float32), t, 200, counter, state=state)
        if counter == 0:
            first_mo = mo
        assert len(out["state"]["ets"]) == _ETS_LEN_AFTER[min(counter, 4)]
        state = out["state"]
    assert len(seq) >= 6
    assert all(not _close(e, first_mo) for e in state["ets"])


def check_final_step_table_head(solve):
    """末步 timestep=0 → prev_timestep=−δ < 0 → α_prev 取表头 ᾱ(0)=1−β_0。"""
    out = solve(_t(0.3), _t(2.0), torch.tensor(_ALPHAS, dtype=torch.float32), 0, 200, 0)
    o = _oracle(0.3, 2.0, _ALPHAS, 0, 200, 0)
    _assert_out(out, o, "final")
    assert out["prev_timestep"] == -200
    # 手算：α_prev = ᾱ(0)，α_t = ᾱ(0)（timestep=0）→ sample_coeff = 1、α_prev − α_t = 0
    # → prev = 1·sample − 0 = sample 恒等！
    assert _close(out["prev_sample"], 2.0)


def check_v_prediction_loop(solve):
    _full_loop(solve, ptype="v_prediction")


def check_prediction_type_reject(solve):
    for bad in ("sample", None, 0):
        try:
            solve(_t(0.3), _t(2.0), torch.tensor(_ALPHAS, dtype=torch.float32), 999, 200, 0,
                  prediction_type=bad)
        except ValueError as e:
            assert "prediction_type" in str(e), str(e)
        else:
            raise AssertionError(f"prediction_type={bad!r} 应抛 ValueError")


def check_alphas_reject(solve):
    ok = (_t(0.3), _t(2.0))
    cases = [
        ([0.9, 0.8], "一维浮点张量"),
        (torch.tensor([[0.9], [0.8]]), "一维浮点张量"),
        (torch.tensor([0.9]), "长度至少为 2"),
        (torch.tensor([float("nan"), 0.8]), "全部有限"),
        (torch.tensor([0.9, 0.0]), "落在 (0, 1]"),
        (torch.tensor([0.9, 1.2]), "落在 (0, 1]"),
        (torch.tensor([0.8, 0.9]), "单调不增"),
    ]
    # 相邻相等合法（契约是非增不是严格递减），应正常计算不抛错
    solve(ok[0], ok[1], torch.tensor([0.9, 0.9, 0.5, 0.2, 0.1]), 4, 200, 0)
    for alphas, msg in cases:
        try:
            solve(*ok, alphas, 999, 200, 0)
        except ValueError as e:
            assert msg in str(e), f"{alphas}: {e}"
        else:
            raise AssertionError(f"alphas={alphas} 应抛 ValueError")


def check_scalar_args_reject(solve):
    sig = torch.tensor(_ALPHAS, dtype=torch.float32)
    for kwargs, msg in [
        (dict(timestep=True), "timestep 必须是整数"),
        (dict(timestep=999.0), "timestep 必须是整数"),
        (dict(timestep=-1), "timestep 必须落在"),
        (dict(timestep=1000), "timestep 必须落在"),
        (dict(step_ratio=0), "step_ratio 必须至少为 1"),
        (dict(step_ratio=True), "step_ratio 必须是整数"),
        (dict(step_ratio=2.0), "step_ratio 必须是整数"),
        (dict(counter=True), "counter 必须是整数"),
        (dict(counter=-1), "counter 必须非负"),
        (dict(counter=0.0), "counter 必须是整数"),
    ]:
        try:
            solve(_t(0.3), _t(2.0), sig, kwargs.get("timestep", 999),
                  kwargs.get("step_ratio", 200), kwargs.get("counter", 0))
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"{kwargs} 应抛 ValueError")


def check_state_reject(solve):
    sig = torch.tensor(_ALPHAS, dtype=torch.float32)
    first = solve(_t(0.3), _t(2.0), sig, 500, 200, 0)
    good = first["state"]

    def expect(counter, state, msg, mo=_t(-0.2), x=_t(1.0), t=500):
        try:
            solve(mo, x, sig, t, 200, counter, state=state)
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"counter={counter} state={state} 应抛 ValueError")

    expect(0, good, "必须传 state=None")
    expect(1, {"ets": good["ets"]}, "只含 ets / cur_sample")
    expect(1, {"ets": [], "cur_sample": _t(2.0)}, "长度必须恰为 1")
    expect(2, {"ets": [], "cur_sample": None}, "长度必须恰为 1")
    expect(3, {"ets": [good["ets"][0]], "cur_sample": None}, "长度必须恰为 2")
    expect(4, {"ets": [good["ets"][0]] * 2, "cur_sample": None}, "长度必须恰为 3")
    expect(5, {"ets": [good["ets"][0]] * 5, "cur_sample": None}, "长度必须恰为 4")
    expect(1, {"ets": good["ets"], "cur_sample": None}, "cur_sample 必须存在")
    expect(2, {"ets": good["ets"], "cur_sample": _t(2.0)}, "必须为 None")
    expect(1, {"ets": [torch.tensor([[1.0, 2.0]])], "cur_sample": _t(2.0)}, "同形状")
    expect(1, {"ets": good["ets"], "cur_sample": torch.tensor([[1.0, 2.0]])}, "同形状")
    # counter 1 的 t+δ 越界（t=999 + δ=200 > 表尾 999）
    try:
        solve(_t(-0.2), first["prev_sample"], sig, 999, 200, 1, state=good)
    except ValueError as e:
        assert "越过 alphas_cumprod 表尾" in str(e), str(e)
    else:
        raise AssertionError("t+δ 越界应抛 ValueError")


def check_shape_1x2(solve):
    mo = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    out = solve(mo, x, torch.tensor(_ALPHAS, dtype=torch.float32), 999, 200, 0)
    o = _oracle([[0.3, -0.2]], [[1.0, 2.0]], _ALPHAS, 999, 200, 0)
    assert _close(out["prev_sample"], o["prev_sample"])
    assert out["prev_sample"].shape == (1, 2)


def check_two_step_trapezoid_example(solve):
    """示例：前两步 = 一次梯形步（counter 0 暂存 + counter 1 重启）。"""
    first = solve(_t(0.3), _t(2.0), torch.tensor(_ALPHAS, dtype=torch.float32), 500, 200, 0)
    assert _close(first["model_output_combined"], 0.3)
    assert first["prev_timestep"] == 300
    second = solve(_t(-0.2), first["prev_sample"], torch.tensor(_ALPHAS, dtype=torch.float32),
                   500, 200, 1, state=first["state"])
    assert _close(second["model_output_combined"], 0.05)  # (0.3 + (-0.2)) / 2
    assert second["prev_timestep"] == 500
    o = _oracle(-0.2, float(first["prev_sample"]), _ALPHAS, 500, 200, 1, state=first["state"])
    assert _close(second["prev_sample"], o["prev_sample"])


def check_ab_coefficients_example(solve):
    """示例：counter 2/3 的组合输出 = 2 阶 / 3 阶 Adams-Bashforth 线性组合。"""
    state = None
    for counter, t in enumerate(_TIMESTEPS):
        mo = 0.1 * (counter + 1)
        out = solve(_t(mo), _t(2.0 if counter == 0 else 1.0),
                    torch.tensor(_ALPHAS, dtype=torch.float32), t, 200, counter, state=state)
        if counter == 2:
            # ets 记账后 [m0, m2] → (3·m2 − m0)/2 = (3·0.3 − 0.1)/2
            assert _close(out["model_output_combined"], (3 * 0.3 - 0.1) / 2)
        if counter == 3:
            # ets = [m0, m2, m3] → (23·m3 − 16·m2 + 5·m0)/12
            assert _close(out["model_output_combined"], (23 * 0.4 - 16 * 0.3 + 5 * 0.1) / 12)
        state = out["state"]


check_two_step_trapezoid_example.example = True
check_ab_coefficients_example.example = True
