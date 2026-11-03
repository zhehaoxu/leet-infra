import math

import numpy as np
import torch


def _t(x):
    if torch.is_tensor(x):
        return x.detach().to(torch.float32)
    return torch.tensor(x, dtype=torch.float32)


def _close(a, b, tol=1e-6):
    a = float(a)
    b = float(b)
    if a == b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _expand(v):
    """头 1 条 + 主体 repeat_interleave(2) + 尾 1 条，长度 2N+2。"""
    v = np.asarray(v, dtype=np.float64)
    return np.concatenate([v[:1], np.repeat(v[1:], 2), v[-1:]])


def _oracle(mo, sample, sigmas, i, state=None, noise=None, prediction_type="epsilon"):
    """官方 scheduling_k_dpm_2_ancestral_discrete 的 float64 numpy 复刻。"""
    v = np.asarray(sigmas, dtype=np.float64)
    N = len(v) - 1
    nxt = np.roll(v, -1)
    nxt[-1] = 0.0
    with np.errstate(invalid="ignore"):
        up = np.sqrt(nxt**2 * (v**2 - nxt**2) / v**2)
    with np.errstate(invalid="ignore"):
        down = np.sqrt(nxt**2 - up**2)
    down[-1] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        interp = np.exp(np.log(v) + 0.5 * (np.log(down) - np.log(v)))
    interp[-2:] = 0.0
    sig_e = _expand(v)
    sig_ie = _expand(interp)
    sig_ue = _expand(up)
    sig_de = _expand(down)

    first = state is None
    if first:
        sigma = sig_e[i]
        m = sig_ie[i]
    else:
        sigma = sig_e[i - 1]
        m = sig_ie[i - 1]
    sigma_hat = sigma  # gamma = 0

    if prediction_type == "epsilon":
        sigma_input = sigma_hat if first else m
        x0 = sample - sigma_input * mo
    else:
        sigma_input = sigma_hat if first else m
        x0 = mo * (-sigma_input / math.sqrt(sigma_input**2 + 1.0)) + sample / (sigma_input**2 + 1.0)

    if first:
        deriv = (sample - x0) / sigma_hat
        dt = m - sigma_hat
        prev = sample + deriv * dt
        return {
            "prev": prev,
            "x0": x0,
            "state": {"sample": sample},
            "dt": dt,
            "up": None,
            "down": None,
        }
    stored = state["sample"]
    deriv = (sample - x0) / m
    dt = sig_de[i - 1] - sigma_hat
    prev = stored + deriv * dt + noise * sig_ue[i - 1]
    return {
        "prev": prev,
        "x0": x0,
        "state": None,
        "dt": dt,
        "up": sig_ue[i - 1],
        "down": sig_de[i - 1],
    }


def _sigmas(ns):
    """构造合法基础表：几何递减 + 尾 0。"""
    v = [8.0 / (2.0**k) for k in range(ns)]
    v.append(0.0)
    return v


def check_against_oracle_random(solve):
    """随机 σ 表 / 模型输出 / 噪声：两步一组对拍 float64 oracle。"""
    rng = np.random.default_rng(11)
    for trial in range(6):
        ns = int(rng.integers(2, 5))
        sigmas = _sigmas(ns)
        mo1, mo2 = rng.normal(size=()), rng.normal(size=())
        x1 = rng.normal(size=())
        noise = rng.normal(size=())
        j = int(rng.integers(0, ns - 1))
        i = 2 * j

        out = solve(_t(mo1), _t(x1), torch.tensor(sigmas), i)
        ref = _oracle(mo1, x1, sigmas, i)
        assert _close(out["prev_sample"], ref["prev"]), (trial, "first prev")
        assert _close(out["pred_original_sample"], ref["x0"]), (trial, "first x0")
        assert _close(out["dt"], ref["dt"]), (trial, "first dt")

        out2 = solve(_t(mo2), _t(ref["prev"]), torch.tensor(sigmas), i + 1,
                     state=out["state"], noise=_t(noise))
        ref2 = _oracle(mo2, ref["prev"], sigmas, i + 1, state=ref["state"], noise=noise)
        assert _close(out2["prev_sample"], ref2["prev"]), (trial, "second prev")
        assert _close(out2["pred_original_sample"], ref2["x0"]), (trial, "second x0")
        assert _close(out2["dt"], ref2["dt"]), (trial, "second dt")
        assert _close(out2["sigma_up"], ref2["up"]), (trial, "up")
        assert _close(out2["sigma_down"], ref2["down"]), (trial, "down")
        assert out2["state"] is None


def check_first_call_half_step(solve):
    """首调 = 欧拉半步走到 interpol 中点；noise 被忽略；up/down 不暴露。"""
    sigmas = [4.0, 1.0, 0.0]
    mo, x = 0.3, 2.0
    out = solve(_t(mo), _t(x), torch.tensor(sigmas), 0, noise=_t(9.0))
    # interpol[0] = √(σ0·down0)，down0 = σ_next[0]²/σ0 = 1²/4 = 0.25
    m = math.sqrt(4.0 * 0.25)
    deriv = (x - (x - 4.0 * mo)) / 4.0  # epsilon: x0 = x - 4mo, deriv = mo
    dt = m - 4.0
    assert _close(out["prev_sample"], x + deriv * dt)
    assert _close(out["dt"], dt)
    assert out["sigma_up"] is None and out["sigma_down"] is None
    assert isinstance(out["state"], dict)
    assert _close(out["state"]["sample"], x)
    # 首调导数分母是 σ 本身
    assert _close(out["derivative"], mo)


def check_second_call_down_step_and_noise(solve):
    """次调 dt = σ_down − σ（非整步长），落地后注入 noise·σ_up，state 清空。"""
    sigmas = [4.0, 1.0, 0.0]
    mo1, x1, mo2, noise = 0.3, 2.0, -0.5, 1.5
    first = solve(_t(mo1), _t(x1), torch.tensor(sigmas), 0)
    out = solve(_t(mo2), _t(first["prev_sample"]), torch.tensor(sigmas), 1,
                state=first["state"], noise=_t(noise))
    down = 1.0**2 / 4.0  # σ_next²/σ
    up = math.sqrt(1.0**2 * (4.0**2 - 1.0**2) / 4.0**2)
    m = math.sqrt(4.0 * down)
    stored = x1
    mid = float(first["prev_sample"])
    x0 = mid - m * mo2
    deriv = (mid - x0) / m
    dt = down - 4.0
    exp_prev = stored + deriv * dt + noise * up
    assert _close(out["prev_sample"], exp_prev), (float(out["prev_sample"]), exp_prev)
    assert _close(out["sigma_up"], up)
    assert _close(out["sigma_down"], down)
    assert _close(out["dt"], dt)
    assert out["state"] is None
    # 次调导数分母是中点 m
    assert _close(out["derivative"], deriv)


def check_stored_sample_departure(solve):
    """prev − noise·up − deriv·dt 必须回到**首调暂存的原样本**，不是中点样本。"""
    sigmas = [4.0, 1.0, 0.0]
    mo1, x1, mo2, noise = 0.3, 2.0, -0.5, 0.8
    first = solve(_t(mo1), _t(x1), torch.tensor(sigmas), 0)
    out = solve(_t(mo2), _t(first["prev_sample"]), torch.tensor(sigmas), 1,
                state=first["state"], noise=_t(noise))
    down = 0.25
    m = math.sqrt(4.0 * down)
    mid = float(first["prev_sample"])
    deriv = mo2  # epsilon 且分母 m：deriv = (mid − (mid − m·mo2))/m = mo2
    dt = down - 4.0
    up = float(out["sigma_up"])
    back = float(out["prev_sample"]) - noise * up - deriv * dt
    assert _close(back, x1), (back, x1)
    assert not _close(back, mid)


def check_down_equals_next_sq_over_sigma(solve):
    """σ_down = σ_next²/σ、σ_up² + σ_down² = σ_next²（方差拆分）；up/down 次调才暴露。"""
    sigmas = [8.0, 2.0, 0.5, 0.0]
    first = solve(_t(0.2), _t(1.0), torch.tensor(sigmas), 2)
    out = solve(_t(-0.1), _t(first["prev_sample"]), torch.tensor(sigmas), 3,
                state=first["state"], noise=_t(0.4))
    # j=1：σ=2，σ_next=0.5
    assert _close(out["sigma_down"], 0.5**2 / 2.0)
    u2 = out["sigma_up"] ** 2
    d2 = out["sigma_down"] ** 2
    assert _close(u2 + d2, 0.5**2)


def check_last_logical_step_lands_on_x0(solve):
    """末逻辑步（j=N−1）只有首调：σ_next=0 → interpol=0、无噪声，一步到 x0。"""
    sigmas = [4.0, 1.0, 0.0]
    mo, x = 0.7, -3.0
    out = solve(_t(mo), _t(x), torch.tensor(sigmas), 2)
    x0 = x - 1.0 * mo
    assert _close(out["prev_sample"], x0)
    assert _close(out["dt"], -1.0)
    assert isinstance(out["state"], dict)


def check_v_prediction_branch(solve):
    """v_prediction：样本项除以 σ_input²+1，σ_input 首调 σ、次调中点。"""
    sigmas = [4.0, 1.0, 0.0]
    mo, x = 0.4, 3.0
    first = solve(_t(mo), _t(x), torch.tensor(sigmas), 0, prediction_type="v_prediction")
    s_in = 4.0
    exp_x0 = mo * (-s_in / math.sqrt(s_in**2 + 1.0)) + x / (s_in**2 + 1.0)
    assert _close(first["pred_original_sample"], exp_x0)
    assert _close(first["derivative"], (x - exp_x0) / 4.0)
    m = 1.0  # √(4·0.25)
    dt = m - 4.0
    assert _close(first["prev_sample"], x + (x - exp_x0) / 4.0 * dt)

    mo2, noise = -0.2, 0.5
    out = solve(_t(mo2), _t(first["prev_sample"]), torch.tensor(sigmas), 1,
                state=first["state"], noise=_t(noise), prediction_type="v_prediction")
    mid = float(first["prev_sample"])
    s2 = m
    exp_x02 = mo2 * (-s2 / math.sqrt(s2**2 + 1.0)) + mid / (s2**2 + 1.0)
    assert _close(out["pred_original_sample"], exp_x02)
    down, up = 0.25, math.sqrt(1.0 * (16.0 - 1.0) / 16.0)
    deriv = (mid - exp_x02) / m
    exp_prev = x + deriv * (down - 4.0) + noise * up
    assert _close(out["prev_sample"], exp_prev)


def check_noise_validation(solve):
    """noise：次调必需、形状一致、浮点；首调忽略。"""
    sigmas = [4.0, 1.0, 0.0]
    first = solve(_t(0.3), _t(2.0), torch.tensor(sigmas), 0)
    bad = [
        (lambda: solve(_t(0.3), _t(1.0), torch.tensor(sigmas), 1, state=first["state"]),
         "次调缺 noise"),
        (lambda: solve(_t(0.3), _t(1.0), torch.tensor(sigmas), 1, state=first["state"],
                       noise=_t([1.0, 2.0])),
         "noise 形状不符"),
        (lambda: solve(_t(0.3), _t(1.0), torch.tensor(sigmas), 1, state=first["state"],
                       noise=torch.tensor(1, dtype=torch.int32)),
         "noise 整型"),
        (lambda: solve(_t(0.3), _t(1.0), torch.tensor(sigmas), 1, state=first["state"],
                       noise=0.5),
         "noise 非张量"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")
    # 首调给 noise 不影响结果
    a = solve(_t(0.3), _t(2.0), torch.tensor(sigmas), 0, noise=_t(5.0))
    b = solve(_t(0.3), _t(2.0), torch.tensor(sigmas), 0)
    assert _close(a["prev_sample"], b["prev_sample"])


def check_prediction_type_validation(solve):
    """未知 prediction_type 一律 ValueError（官方对 "sample" 抛 NotImplementedError）。"""
    sigmas = [4.0, 1.0, 0.0]
    for bad_t in ("sample", "x0", "", None):
        try:
            solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 0, prediction_type=bad_t)
        except ValueError:
            pass
        else:
            raise AssertionError(f"未抛错：{bad_t!r}")
    solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 0, prediction_type="v_prediction")


def check_reject_parity_and_range(solve):
    """奇偶与范围：首调偶数、次调奇数、末逻辑步无次调、越界拒绝。"""
    sigmas = [4.0, 1.0, 0.0]
    st = {"sample": _t(1.0)}
    bad = [
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 1), "首调用奇数索引"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 0, state=st), "次调用偶数索引"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 3, state=st), "末逻辑步次调"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), -1), "负索引"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 5), "首调越界"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 5, state=st), "次调越界"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), True), "bool 索引"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 1.0), "浮点索引"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 4), "首调索引恰为 2N"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([8.0, 4.0, 1.0, 0.0]), 2,
                       state=st, noise=_t(0.5)), "次调索引为偶数（N=3 落在范围内）"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")
    # 末逻辑步次调必须以"范围"错误拒绝
    try:
        solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 3, state=st, noise=_t(0.5))
    except ValueError as e:
        assert "末逻辑步" in str(e), str(e)
    else:
        raise AssertionError("末逻辑步次调未抛错")


def check_state_validation(solve):
    """state：必须 dict 且只含 sample、张量、形状一致。"""
    sigmas = [4.0, 1.0, 0.0]
    bad = [
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 1, state={"sample": 1.0}),
         "sample 非张量"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 1,
                       state={"sample": _t(1.0), "dt": 0.1}),
         "多余键"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 1,
                       state={"sample": _t([1.0, 2.0])}),
         "形状不符"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 1, state=0),
         "state 非容器"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")


def check_sigma_table_validation(solve):
    """sigmas：长度、单调、末位 0、前 N 为正。"""
    bad = [
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([4.0, 0.0]), 0), "长度 2"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([4.0, 1.0, 2.0, 0.0]), 0), "非单调"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([4.0, 1.0, 0.5]), 0), "末位非 0"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([0.0, 1.0, 0.0]), 0), "首元为 0"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([4.0, -1.0, 0.0]), 0), "负 σ"),
        (lambda: solve(_t(0.1), _t(1.0), torch.tensor([4.0, 1.0, 0.5]), 0), "末位非 0"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")


def check_three_step_loop(solve):
    """N=3 完整循环（2N−1 次调用）：状态机交错 + 末步落 x0，全程对拍 oracle。"""
    sigmas = _sigmas(3)
    rng = np.random.default_rng(5)
    mo = {k: float(rng.normal()) for k in range(6)}
    x = float(rng.normal())
    noise = {k: float(rng.normal()) for k in range(3)}
    state = None
    cur = x
    call = 0
    for j in range(3):
        out = solve(_t(mo[call]), _t(cur), torch.tensor(sigmas), 2 * j)
        ref = _oracle(mo[call], cur, sigmas, 2 * j)
        assert _close(out["prev_sample"], ref["prev"]), (j, "first")
        state = out["state"]
        cur = float(out["prev_sample"])
        call += 1
        if j < 2:
            out2 = solve(_t(mo[call]), _t(cur), torch.tensor(sigmas), 2 * j + 1,
                         state=state, noise=_t(noise[j]))
            ref2 = _oracle(mo[call], cur, sigmas, 2 * j + 1,
                           state=ref["state"], noise=noise[j])
            assert _close(out2["prev_sample"], ref2["prev"]), (j, "second")
            assert out2["state"] is None
            cur = float(out2["prev_sample"])
            call += 1
    # 末逻辑步只有首调：共 2N−1 = 5 次调用
    assert call == 5


def check_full_loop_ends_small(solve):
    """端到端：末步 prev 与 x0 预测一致（σ_next=0，无噪声）。"""
    sigmas = [4.0, 1.0, 0.0]
    mos = [0.3, -0.2, 0.5]
    x = 2.0
    first = solve(_t(mos[0]), _t(x), torch.tensor(sigmas), 0)
    second = solve(_t(mos[1]), _t(first["prev_sample"]), torch.tensor(sigmas), 1,
                   state=first["state"], noise=_t(0.3))
    last = solve(_t(mos[2]), _t(second["prev_sample"]), torch.tensor(sigmas), 2)
    exp_x0 = float(second["prev_sample"]) - 1.0 * mos[2]
    assert _close(last["prev_sample"], exp_x0)
    # 末步没有噪声可注入（up=0），dt = −σ2
    assert _close(last["dt"], -1.0)


def run_examples(solve):
    """示例 1：两调用一步（含噪声注入）。示例 2：三逻辑步全循环。"""
    sigmas = [4.0, 1.0, 0.0]
    first = solve(_t(0.3), _t(2.0), torch.tensor(sigmas), 0)
    second = solve(_t(-0.5), _t(first["prev_sample"]), torch.tensor(sigmas), 1,
                   state=first["state"], noise=_t(0.8))
    assert second["state"] is None

    sigmas = [8.0, 4.0, 1.0, 0.0]
    state = None
    cur = 1.5
    mos = [0.2, -0.1, 0.4, -0.3, 0.1]
    noises = [0.2, 0.2, 0.2, 0.2, 0.2]
    call = 0
    for j in range(3):
        out = solve(_t(mos[call]), _t(cur), torch.tensor(sigmas), 2 * j)
        state = out["state"]
        cur = float(out["prev_sample"])
        call += 1
        if j < 2:
            out2 = solve(_t(mos[call]), _t(cur), torch.tensor(sigmas), 2 * j + 1,
                         state=state, noise=_t(noises[j]))
            assert out2["state"] is None
            cur = float(out2["prev_sample"])
            call += 1
    assert call == 5
