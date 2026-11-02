import math

import numpy as np
import torch


def _oracle(mo, sample, base_sigmas, step_index, state=None, prediction_type="epsilon"):
    """独立 numpy 重写 KDPM2 单步（公式从 diffusers scheduling_k_dpm_2_discrete.py
    的 set_timesteps L323-345 与 step L598-663 数学定义直接推出）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    s = np.asarray(base_sigmas, dtype=np.float64)
    n = len(s) - 1

    # 展开表（几何平均中点，官方 log-lerp 的同构式）
    I = np.concatenate([[0.0], np.sqrt(s[:-1] * s[1:])])
    sig_e = np.concatenate([[s[0]], np.repeat(s[1:], 2), [s[-1]]])
    sig_ie = np.concatenate([[I[0]], np.repeat(I[1:], 2), [I[-1]]])

    def x0_fn(sig_in):
        if prediction_type == "epsilon":
            return x - sig_in * mo_
        return mo_ * (-sig_in / np.sqrt(sig_in**2 + 1.0)) + x / (sig_in**2 + 1.0)

    if state is None:
        sigma = sig_e[step_index]
        mid = sig_ie[step_index + 1]
        x0 = x0_fn(sigma)
        deriv = (x - x0) / sigma
        dt = mid - sigma
        prev = x + deriv * dt
        return prev, x0, {"sample": x}, deriv, dt
    sigma = sig_e[step_index - 1]
    mid = sig_ie[step_index]
    sigma_next = sig_e[step_index]
    x0 = x0_fn(mid)
    deriv = (x - x0) / mid
    dt = sigma_next - sigma
    prev = state["sample"] + deriv * dt
    return prev, x0, None, deriv, dt


def _t(v):
    return torch.tensor([[v]], dtype=torch.float32)


def _close(a, b, tol=1e-6):
    a = float(a)
    b = float(b)
    if a == b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _run(solve, mo, sample, sigmas, i, state=None, prediction_type="epsilon"):
    """跑 solve（state 用 dict 原样透传），再算 oracle 期望。返回
    (out, prev, x0, o_st, deriv, dt)；o_st 是 oracle 的 state，仅供对拍参考，
    传回下一调用必须用 out["state"]（torch 张量，模拟 pipeline 的 float32 往返）。"""
    out = solve(_t(mo), _t(sample), torch.tensor(sigmas, dtype=torch.float32), i,
                state=state, prediction_type=prediction_type)
    prev, x0, o_st, deriv, dt = _oracle(mo, sample, sigmas, i, state=state,
                                        prediction_type=prediction_type)
    return out, prev, x0, o_st, deriv, dt


def check_first_call_midpoint(solve):
    """首调：欧拉半步走到 log-σ 几何平均中点，state 只存原样本。"""
    out, prev, x0, st, deriv, dt = _run(solve, 0.1, 1.0, [4.0, 1.0, 0.0], 0)
    # I1 = sqrt(4*1) = 2 → dt = 2-4 = -2
    assert _close(out["dt"], -2.0)
    assert _close(out["dt"], dt)
    assert _close(out["pred_original_sample"], 1.0 - 4.0 * 0.1)
    assert _close(out["derivative"], 0.1)
    assert _close(out["prev_sample"], 1.0 + 0.1 * (-2.0))
    assert _close(out["prev_sample"], prev)
    assert set(out["state"].keys()) == {"sample"}
    assert _close(out["state"]["sample"], 1.0)


def check_second_call_restarts_from_stored(solve):
    """次调：导数在中点上重估、除以中点，dt 是整步长，样本回滚到暂存原样本。"""
    first, _, _, _, _, _ = _run(solve, 0.1, 1.0, [4.0, 1.0, 0.0], 0)
    out, prev, x0, _, deriv, dt = _run(solve, 0.2, float(first["prev_sample"]),
                                       [4.0, 1.0, 0.0], 1, state=first["state"])
    # 中点样本 0.8，I1 = 2：x0 = 0.8 - 2*0.2 = 0.4，deriv = 0.4/2 = 0.2
    assert _close(out["pred_original_sample"], 0.4)
    assert _close(out["derivative"], 0.2)
    # dt = σ_next - σ = 1 - 4 = -3（整步，不是半步）
    assert _close(out["dt"], -3.0)
    assert _close(out["dt"], dt)
    # prev = 暂存样本 1.0 + 0.2*(-3) = 0.4；若从当前中点样本出发会得 0.2
    assert _close(out["prev_sample"], 0.4)
    assert _close(out["prev_sample"], prev)
    assert out["state"] is None


def check_last_step_prediction_only(solve):
    """末逻辑步（i = 2N-2）只有首调：中点表尾部是 0，dt = -σ 一步到零；次调被拒。"""
    out, prev, x0, st, deriv, dt = _run(solve, 0.1, 1.0, [4.0, 1.0, 0.0], 2)
    assert _close(out["dt"], -1.0)
    assert _close(out["dt"], dt)
    # epsilon 下 deriv = mo，prev = x - σ·mo = x0
    assert _close(out["prev_sample"], 0.9)
    assert _close(out["pred_original_sample"], 0.9)
    try:
        solve(_t(0.1), _t(1.0), torch.tensor([4.0, 1.0, 0.0]), 3,
              state={"sample": _t(1.0)})
    except ValueError:
        pass
    else:
        raise AssertionError("末逻辑步不应存在次调")


def check_v_prediction_branch(solve):
    """v_prediction 两阶都用对应 sigma_input（首调 σ、次调中点），样本项除以 σ²+1。"""
    mo, mo2 = 0.5, -0.3
    first_out, prev, x0, _, deriv, dt = _run(solve, mo, 3.0, [4.0, 1.0, 0.0], 0,
                                             prediction_type="v_prediction")
    s_in = 4.0
    exp_x0 = mo * (-s_in / math.sqrt(s_in**2 + 1.0)) + 3.0 / (s_in**2 + 1.0)
    assert _close(first_out["pred_original_sample"], exp_x0)
    assert _close(first_out["pred_original_sample"], x0)
    assert _close(first_out["derivative"], (3.0 - exp_x0) / 4.0)
    mid = 3.0 + first_out["derivative"] * (-2.0)
    assert _close(first_out["prev_sample"], mid)
    assert _close(first_out["prev_sample"], prev)

    out2, prev2, x02, _, deriv2, dt2 = _run(solve, mo2, float(mid), [4.0, 1.0, 0.0], 1,
                                            state=first_out["state"], prediction_type="v_prediction")
    s_in = 2.0
    exp_x02 = mo2 * (-s_in / math.sqrt(s_in**2 + 1.0)) + mid / (s_in**2 + 1.0)
    assert _close(out2["pred_original_sample"], exp_x02)
    assert _close(out2["pred_original_sample"], x02)
    assert _close(out2["derivative"], (mid - exp_x02) / 2.0)
    assert _close(out2["dt"], -3.0)
    assert _close(out2["prev_sample"], 3.0 + out2["derivative"] * (-3.0))
    assert _close(out2["prev_sample"], prev2)
    assert out2["state"] is None


def check_geometric_midpoint(solve):
    """中点是 log-σ 几何平均：dt 首调 = √(σ_j·σ_{j+1}) − σ_j。"""
    for s0, s1, expect_mid in ((9.0, 4.0, 6.0), (1.0, 0.25, 0.5), (3.0, 2.0, math.sqrt(6.0))):
        out, prev, x0, st, deriv, dt = _run(solve, 0.1, 1.0, [s0, s1, 0.0], 0)
        assert _close(out["dt"], expect_mid - s0), (s0, s1, out["dt"])
        assert _close(out["dt"], dt)
    # 非完全平方也精确：√(3·2)
    out, _, _, _, _, dt = _run(solve, 0.1, 1.0, [3.0, 2.0, 0.0], 0)
    assert _close(out["dt"], math.sqrt(6.0) - 3.0)


def check_three_step_loop(solve):
    """N=3 完整轨迹 A0,B0,A1,B1,A2：每步 dt 序列与最终落点对拍独立 oracle。"""
    sigmas = [8.0, 4.0, 1.0, 0.0]
    mo = 0.5
    x = 2.0
    state = None
    expect_dts = [math.sqrt(8.0 * 4.0) - 8.0, 4.0 - 8.0,
                  math.sqrt(4.0 * 1.0) - 4.0, 1.0 - 4.0, -1.0]
    k = 0
    for i in (0, 1, 2, 3, 4):
        out, prev, x0, st, deriv, dt = _run(solve, mo, x, sigmas, i, state=state)
        assert _close(out["dt"], expect_dts[k]), (i, out["dt"], expect_dts[k])
        assert _close(out["prev_sample"], prev), (i, float(out["prev_sample"]), prev)
        assert _close(out["pred_original_sample"], x0)
        state = out["state"]
        x = float(out["prev_sample"])
        k += 1
    # 末步直接落在 x0 预测上：prev = x4 - 1.0*mo ≈ x0
    assert _close(out["prev_sample"], out["pred_original_sample"])
    # 末次调用是首调，state 仍处于"已存样本"状态（官方 self.sample 不清空）
    assert isinstance(state, dict)


def check_duplicate_sigmas_degenerate(solve):
    """σ_{j+1} == σ_j 时中点 == 两端，dt 全为 0，prev 原样返回。"""
    out, prev, x0, st, deriv, dt = _run(solve, 0.7, 1.0, [2.0, 2.0, 0.0], 0)
    assert _close(out["dt"], 0.0)
    assert _close(out["prev_sample"], 1.0)
    out2, prev2, x02, _, deriv2, dt2 = _run(solve, 0.4, float(out["prev_sample"]),
                                            [2.0, 2.0, 0.0], 1, state=out["state"])
    assert _close(out2["dt"], 0.0)
    assert _close(out2["prev_sample"], 1.0)
    assert out2["state"] is None


def check_expanded_table_indices(solve):
    """交错索引：次调读 σ_j（i−1）、中点 I_{j+1}（i）、σ_next = σ_{j+1}（i）。"""
    # N=2, i=1：σ = sigE[0] = 4，中点 = sigIE[1] = I1 = 2，σ_next = sigE[1] = 1
    first, _, _, _, _, _ = _run(solve, 0.0, 5.0, [4.0, 1.0, 0.0], 0)
    # mo=0 → x0 = sample，deriv = 0，mid = 5.0（原样）；次调 mo=0.5
    out, prev, x0, _, deriv, dt = _run(solve, 0.5, 5.0, [4.0, 1.0, 0.0], 1, state=first["state"])
    assert _close(out["pred_original_sample"], 5.0 - 2.0 * 0.5)
    assert _close(out["derivative"], (5.0 - (5.0 - 1.0)) / 2.0)
    assert _close(out["dt"], 1.0 - 4.0)
    assert _close(out["prev_sample"], 5.0 + out["derivative"] * (-3.0))


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
                       state={"sample": _t(1.0)}), "次调索引为偶数（N=3 落在范围内）"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")

    # 末逻辑步次调必须以"范围"错误拒绝，而不是滑到中点为正的检查里
    try:
        solve(_t(0.1), _t(1.0), torch.tensor(sigmas), 3, state=st)
    except ValueError as e:
        assert "step_index 必须落在" in str(e), str(e)
    else:
        raise AssertionError("末逻辑步次调未抛错")


def check_reject_bad_sigmas(solve):
    """基础表校验：维度/类型/长度/单调/末位零/前 N 个为正。"""
    x, mo = _t(1.0), _t(0.1)
    bad = [
        (torch.tensor([[4.0, 1.0, 0.0]]), "二维 sigmas"),
        (torch.tensor([4, 1, 0]), "整型 sigmas"),
        (torch.tensor([4.0, 0.0]), "长度不足"),
        (torch.tensor([1.0, 4.0, 0.0]), "递增"),
        (torch.tensor([4.0, 1.0, 1e-9]), "末位非零"),
        (torch.tensor([4.0, float("nan"), 0.0]), "含 NaN"),
        (torch.tensor([4.0, -1.0, 0.0]), "负 σ"),
        (torch.tensor([4.0, 0.0, 0.0]), "前 N 个含零"),
    ]
    for sig, tag in bad:
        try:
            solve(mo, x, sig, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")


def check_reject_inputs_and_prediction(solve):
    """输入张量与 prediction_type 校验。"""
    sigmas = torch.tensor([4.0, 1.0, 0.0])
    st = {"sample": _t(1.0)}
    bad = [
        (lambda: solve(torch.randn(2), torch.randn(3), sigmas, 0), "形状不一致"),
        (lambda: solve(torch.randint(0, 2, (2, 2)), torch.randn(2, 2), sigmas, 0), "整型输出"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 0,
                       prediction_type="sample"), "上游同样拒绝 sample"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 0,
                       prediction_type="flow"), "未知类型"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 1,
                       state={"sample": torch.randn(3, 3)}), "state 形状不一致"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 1,
                       state={"sample": torch.randn(2, 2), "dt": 1.0}), "state 多键"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 1, state={}), "state 空键"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 1, state=[1]), "state 非 dict"),
        (lambda: solve(torch.randn(2, 2), torch.randn(2, 2), sigmas, 1,
                       state={"sample": torch.randint(0, 2, (2, 2))}), "state 整型"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")


def check_against_oracle_random(solve):
    """随机多组（固定种子）对拍独立 numpy oracle：A/B 两阶全路径。"""
    rng = np.random.default_rng(20261102)
    for trial in range(6):
        n = int(rng.integers(2, 5))
        vals = np.sort(rng.uniform(0.5, 9.0, size=n))[::-1]
        sigmas = np.concatenate([vals, [0.0]])
        mo = rng.uniform(-1.0, 1.0)
        x = rng.uniform(-2.0, 2.0)
        state = None
        for i in range(2 * n - 1):
            mo_i = rng.uniform(-1.0, 1.0)
            out, prev, x0, st, deriv, dt = _run(solve, mo_i, x, list(sigmas), i, state=state)
            assert _close(out["prev_sample"], prev), (trial, i)
            assert _close(out["pred_original_sample"], x0), (trial, i)
            assert _close(out["dt"], dt), (trial, i)
            state = out["state"]
            x = float(out["prev_sample"])
            if i % 2 == 1:
                assert state is None
            else:
                assert isinstance(state, dict)


def check_two_step_end_to_end_example(solve):
    """2 步端到端：数值直接按解析式核对（不只是 oracle 对拍）。"""
    sigmas = [4.0, 1.0, 0.0]
    x, mo = 1.0, 0.1
    a = solve(_t(mo), _t(x), torch.tensor(sigmas), 0)
    assert _close(a["prev_sample"], 0.8)
    b = solve(_t(0.2), a["prev_sample"], torch.tensor(sigmas), 1, state=a["state"])
    assert _close(b["prev_sample"], 0.4)
    c = solve(_t(0.3), b["prev_sample"], torch.tensor(sigmas), 2)
    # 末步：prev = x3 - 1.0*0.3，x3 = 0.4
    assert _close(c["prev_sample"], 0.4 - 0.3)


def run_examples(solve):
    check_two_step_end_to_end_example(solve)
    check_first_call_midpoint(solve)


check_two_step_end_to_end_example.example = True
check_first_call_midpoint.example = True
