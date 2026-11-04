"""TCD gamma 两段式一致性步 -- 测试用例."""

import math

import numpy as np
import torch

_N_TRAIN = 50
_BETA = np.linspace(1e-4, 0.02, _N_TRAIN)
_ABAR = np.cumprod(1.0 - _BETA)  # float64, 单调不增
_TAB5 = [49, 36, 23, 10, 0]  # N=5, 严格递减


def _abar_t():
    return torch.tensor(_ABAR, dtype=torch.float64)


def _t(v, b=1):
    return torch.tensor([[v, v * 0.5 - 0.3]] * b, dtype=torch.float64)


def _tab(*vals):
    return torch.tensor(list(vals), dtype=torch.int64)


def _close(actual, expected, tol=1e-9):
    a = np.asarray(torch.detach(actual).to(torch.float64).numpy(), dtype=np.float64).reshape(-1)
    e = np.broadcast_to(np.asarray(expected, dtype=np.float64).reshape(-1), a.shape)
    return bool(np.all(np.abs(a - e) <= tol))


def _oracle(mo, x, t_idx, prev_t, s_t, noise, eta, ptype):
    """独立 oracle: ᾱ 表为 python list, math.sqrt 实现."""
    at = float(_ABAR[t_idx])
    bt = 1.0 - at
    a_s = float(_ABAR[s_t])
    bs = 1.0 - a_s
    a_prev = float(_ABAR[prev_t])
    x0, eps = [], []
    for row_m, row_x in zip(mo, x):
        r0, re = [], []
        for m, xv in zip(row_m, row_x):
            if ptype == "epsilon":
                r0.append((xv - math.sqrt(bt) * m) / math.sqrt(at))
                re.append(m)
            elif ptype == "sample":
                r0.append(m)
                re.append((xv - math.sqrt(at) * m) / math.sqrt(bt))
            else:  # v_prediction
                r0.append(math.sqrt(at) * xv - math.sqrt(bt) * m)
                re.append(math.sqrt(at) * m + math.sqrt(bt) * xv)
        x0.append(r0)
        eps.append(re)
    xs = [[math.sqrt(a_s) * x0[b][j] + math.sqrt(bs) * eps[b][j] for j in range(len(x0[0]))]
          for b in range(len(x0))]
    applied = eta > 0 and prev_t != -999  # 占位, 由调用方覆盖
    return x0, eps, xs, a_prev, a_s


def _prev_from_xs(xs, noise, a_prev, a_s):
    return [[math.sqrt(a_prev / a_s) * xs[b][j] + math.sqrt(1.0 - a_prev / a_s) * noise[b][j]
             for j in range(len(xs[0]))] for b in range(len(xs))]


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 全循环对拍
def check_full_loop_epsilon(solve):
    """N=5 表、eta=0.3 全循环对拍: 前 4 步加噪、末步不加噪."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    x = 2.0
    for i in range(5):
        mo = 0.3 * x - 0.5
        t_idx = _TAB5[i]
        prev_t = _TAB5[i + 1] if i + 1 < 5 else 0
        s_t = math.floor((1.0 - 0.3) * prev_t)
        need_noise = i != 4  # 末步不加噪
        nz = 0.4 - 0.1 * i
        out = solve(_t(mo), _t(x), abar, tab, i,
                    noise=_t(nz) if need_noise else None, eta=0.3)
        mo_row = [[mo, mo * 0.5 - 0.3]]
        x_row = [[x, x * 0.5 - 0.3]]
        x0, eps, xs, a_prev, a_s = _oracle(mo_row, x_row, t_idx, prev_t, s_t, None, 0.3, "epsilon")
        assert _close(out["pred_original_sample"], x0), f"step {i} x0"
        assert _close(out["pred_epsilon"], eps), f"step {i} eps"
        assert _close(out["pred_noised_sample"], xs), f"step {i} x_s"
        assert out["timestep_s"] == s_t, f"step {i} s"
        assert out["prev_timestep"] == prev_t, f"step {i} prev_t"
        assert out["noise_applied"] is need_noise, f"step {i} gate"
        if need_noise:
            n_row = [[nz, nz * 0.5 - 0.3]]
            expected_prev = _prev_from_xs(xs, n_row, a_prev, a_s)
            assert _close(out["prev_sample"], expected_prev), f"step {i} prev"
        else:
            assert _close(out["prev_sample"], xs, tol=0.0), f"step {i} prev==x_s"
        x = float(out["prev_sample"].reshape(-1)[0])


def check_eta_zero_deterministic(solve):
    """eta=0: timestep_s == prev_t (floor 恒等), prev == x_s, noise 被忽略."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    out = solve(_t(0.5), _t(1.0), abar, tab, 1, noise=_t(9.9), eta=0.0)
    assert out["timestep_s"] == out["prev_timestep"] == 23
    assert out["noise_applied"] is False
    assert torch.equal(out["prev_sample"], out["pred_noised_sample"])


def check_one_step(solve):
    """单步表 (len=1): step 0 即末步, prev_t=0, prev == x_s (官方单步采样)."""
    abar = _abar_t()
    tab = _tab(49)
    out = solve(_t(0.6), _t(1.7), abar, tab, 0, noise=None, eta=0.3)
    assert out["prev_timestep"] == 0
    assert out["noise_applied"] is False
    assert torch.equal(out["prev_sample"], out["pred_noised_sample"])
    s_t = math.floor(0.7 * 0)
    assert out["timestep_s"] == s_t == 0
    # x_s 级别此时 s=t=0: x_s = sqrt(abar[0])*x0 + sqrt(1-abar[0])*eps
    at = float(_ABAR[49])
    mo_row, x_row = [[0.6, 0.0]], [[1.7, 0.55]]
    x0, eps, xs, _, _ = _oracle(mo_row, x_row, 49, 0, 0, None, 0.3, "epsilon")
    assert _close(out["pred_noised_sample"], xs)


# ---------------------------------------------------------------- 参数化
def check_three_parameterizations(solve):
    """epsilon/sample/v_prediction 三种参数化各自对拍 (不对称数值)."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    t_idx, prev_t = 36, 23
    s_t = math.floor(0.7 * prev_t)
    nz = 0.25
    for ptype, mo_v in (("epsilon", 0.45), ("sample", 1.05), ("v_prediction", -0.35)):
        out = solve(_t(mo_v), _t(1.3), abar, tab, 1, noise=_t(nz), eta=0.3,
                    prediction_type=ptype)
        mo_row = [[mo_v, mo_v * 0.5 - 0.3]]
        x_row = [[1.3, 0.35]]
        n_row = [[nz, nz * 0.5 - 0.3]]
        x0, eps, xs, a_prev, a_s = _oracle(mo_row, x_row, t_idx, prev_t, s_t, None, 0.3, ptype)
        assert _close(out["pred_original_sample"], x0), ptype
        assert _close(out["pred_epsilon"], eps), ptype
        assert _close(out["pred_noised_sample"], xs), ptype
        assert _close(out["prev_sample"], _prev_from_xs(xs, n_row, a_prev, a_s)), ptype


def check_parameterization_consistency(solve):
    """合成互验: x = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps, 三种 mo 给出同一 x0/x_s (逐列)."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    t_idx, at = 49, float(_ABAR[49])
    bt = 1.0 - at
    x0_col = [1.4, -0.7]   # 两列不同的 (x0, eps) 对
    eps_col = [0.6, 0.2]
    x_col = [math.sqrt(at) * x0_col[j] + math.sqrt(bt) * eps_col[j] for j in range(2)]
    mo_v = [math.sqrt(at) * eps_col[j] - math.sqrt(bt) * x0_col[j] for j in range(2)]
    outs = []
    for ptype, mo_row in (("epsilon", eps_col), ("sample", x0_col), ("v_prediction", mo_v)):
        mo = torch.tensor([mo_row], dtype=torch.float64)
        xv = torch.tensor([x_col], dtype=torch.float64)
        outs.append(solve(mo, xv, abar, tab, 0, noise=_t(0.2), eta=0.3,
                          prediction_type=ptype))
    ref = outs[0]["pred_original_sample"]
    for o in outs[1:]:
        assert torch.allclose(o["pred_original_sample"], ref, atol=1e-12), ptype
        assert torch.allclose(o["pred_noised_sample"], outs[0]["pred_noised_sample"], atol=1e-12)
    assert _close(ref, x0_col, tol=1e-12)


def check_floor_gamma(solve):
    """timestep_s = floor((1-eta)*prev_t): 与 round 区分的案例."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    # prev_t=10 (step 2): eta=0.25 -> 7.5 -> floor 7 (round 会给 8)
    out = solve(_t(0.4), _t(1.0), abar, tab, 2, noise=_t(0.1), eta=0.25)
    assert out["prev_timestep"] == 10 and out["timestep_s"] == 7
    # eta=0 -> 恒等
    out0 = solve(_t(0.4), _t(1.0), abar, tab, 2, noise=_t(0.1), eta=0.0)
    assert out0["timestep_s"] == 10
    # eta=1 -> 0
    out1 = solve(_t(0.4), _t(1.0), abar, tab, 2, noise=_t(0.1), eta=1.0)
    assert out1["timestep_s"] == 0


def check_two_segment(solve):
    """非末步: prev = sqrt(abar_prev/abar_s)*x_s + sqrt(1-ratio)*noise 两段独立成立."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    out = solve(_t(0.5), _t(1.1), abar, tab, 0, noise=_t(0.3), eta=0.3)
    a_prev = float(_ABAR[36])
    a_s = float(_ABAR[math.floor(0.7 * 36)])
    xs_row = out["pred_noised_sample"].to(torch.float64).numpy().tolist()
    n_row = [[0.3, -0.15]]
    expected = _prev_from_xs(xs_row, n_row, a_prev, a_s)
    assert _close(out["prev_sample"], expected)


def check_final_step_no_noise(solve):
    """末步 (step 4): 换不同 noise, prev 逐位相同且 == x_s."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    out_a = solve(_t(0.5), _t(1.0), abar, tab, 4, noise=_t(9.9), eta=0.3)
    out_b = solve(_t(0.5), _t(1.0), abar, tab, 4, noise=_t(-3.3), eta=0.3)
    assert torch.equal(out_a["prev_sample"], out_b["prev_sample"])
    assert out_a["noise_applied"] is False
    assert torch.equal(out_a["prev_sample"], out_a["pred_noised_sample"])


def check_pure_function(solve):
    """纯函数: 同参数两次调用逐位相同."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    kw = dict(noise=_t(0.3), eta=0.3)
    a = solve(_t(0.5), _t(1.0), abar, tab, 0, **kw)
    b = solve(_t(0.5), _t(1.0), abar, tab, 0, **kw)
    for k in ("prev_sample", "pred_noised_sample", "pred_original_sample", "pred_epsilon"):
        assert torch.equal(a[k], b[k]), k
    assert (a["timestep_s"], a["prev_timestep"], a["noise_applied"]) == \
           (b["timestep_s"], b["prev_timestep"], b["noise_applied"])


# ---------------------------------------------------------------- 入参校验
def check_timesteps_validation(solve):
    """timesteps: 类型/维度/空/整型/严格递减/索引范围."""
    abar = _abar_t()
    good = _tab(*_TAB5)
    kw = dict(noise=_t(0.1), eta=0.3)
    assert "timesteps 必须是 torch 张量" in _expect_err(solve, _t(0.5), _t(1.0), abar, _TAB5, 0, **kw)
    assert "一维" in _expect_err(solve, _t(0.5), _t(1.0), abar, good.unsqueeze(0), 0, **kw)
    assert "不能为空" in _expect_err(solve, _t(0.5), _t(1.0), abar, _tab(), 0, **kw)
    assert "整型" in _expect_err(solve, _t(0.5), _t(1.0), abar,
                                torch.tensor([4.0, 3.0], dtype=torch.float64), 0, **kw)
    assert "严格递减" in _expect_err(solve, _t(0.5), _t(1.0), abar, _tab(49, 49, 23), 0, **kw)
    assert "严格递减" in _expect_err(solve, _t(0.5), _t(1.0), abar, _tab(23, 36, 49), 0, **kw)
    assert "索引范围" in _expect_err(solve, _t(0.5), _t(1.0), abar, _tab(50, 0), 0, **kw)
    assert "索引范围" in _expect_err(solve, _t(0.5), _t(1.0), abar, _tab(-1, 0), 0, **kw)


def check_alphas_validation(solve):
    """alphas_cumprod: 类型/维度/长度/dtype."""
    good = _tab(*_TAB5)
    kw = dict(noise=_t(0.1), eta=0.3)
    assert "alphas_cumprod 必须是 torch 张量" in _expect_err(
        solve, _t(0.5), _t(1.0), _ABAR, good, 0, **kw)
    assert "一维" in _expect_err(solve, _t(0.5), _t(1.0), _abar_t().unsqueeze(0), good, 0, **kw)
    assert "长度" in _expect_err(solve, _t(0.5), _t(1.0), torch.tensor([0.9], dtype=torch.float64),
                                good, 0, **kw)
    assert "浮点" in _expect_err(solve, _t(0.5), _t(1.0), torch.arange(10), good, 0, **kw)


def check_step_index_validation(solve):
    """step_index: int 类型与范围."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    kw = dict(noise=_t(0.1), eta=0.3)
    assert "必须是整数" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, True, **kw)
    assert "必须是整数" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 0.0, **kw)
    assert "必须是整数" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, "0", **kw)
    assert "必须落在" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, -1, **kw)
    assert "必须落在" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 5, **kw)
    out = solve(_t(0.5), _t(1.0), abar, tab, 4, **kw)  # 合法上界 (末步)
    assert out["noise_applied"] is False


def check_eta_validation(solve):
    """eta: 数值且 ∈ [0,1] (官方 assert 语义, 判题域 ValueError)."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    kw = dict(noise=_t(0.1))
    assert "gamma must be less than or equal to 1.0" in _expect_err(
        solve, _t(0.5), _t(1.0), abar, tab, 0, eta=1.5, **kw)
    assert "gamma must be less than or equal to 1.0" in _expect_err(
        solve, _t(0.5), _t(1.0), abar, tab, 0, eta=-0.1, **kw)
    assert "必须是数值" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 0, eta="0.3", **kw)
    assert "必须是数值" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 0, eta=True, **kw)
    out = solve(_t(0.5), _t(1.0), abar, tab, 0, eta=1.0, **kw)  # 合法上界
    assert out["noise_applied"] is True


def check_noise_validation(solve):
    """noise: eta>0 且非末步必须给; 类型/形状; eta=0 时忽略."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    assert "必须显式给 noise" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 0,
                                            eta=0.3, noise=None)
    assert "必须是 torch 张量" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 0,
                                            eta=0.3, noise=np.zeros((1, 2)))
    assert "形状必须一致" in _expect_err(solve, _t(0.5), _t(1.0), abar, tab, 0,
                                        eta=0.3, noise=torch.zeros((2, 2), dtype=torch.float64))
    out = solve(_t(0.5), _t(1.0), abar, tab, 4, eta=0.3, noise=None)  # 末步 None 合法
    assert out["noise_applied"] is False
    out0 = solve(_t(0.5), _t(1.0), abar, tab, 0, eta=0.0, noise=None)  # eta=0 None 合法
    assert out0["noise_applied"] is False


def check_misc_validation(solve):
    """mo/sample 类型与形状; prediction_type."""
    abar = _abar_t()
    tab = _tab(*_TAB5)
    kw = dict(noise=_t(0.1), eta=0.3)
    assert "model_output 必须是 torch 张量" in _expect_err(
        solve, np.zeros(3), _t(1.0), abar, tab, 0, **kw)
    assert "sample 必须是 torch 张量" in _expect_err(
        solve, _t(0.5), np.zeros((1, 2)), abar, tab, 0, **kw)
    assert "形状必须一致" in _expect_err(
        solve, torch.zeros((1, 3), dtype=torch.float64), _t(1.0), abar, tab, 0, **kw)
    assert "must be one of" in _expect_err(
        solve, _t(0.5), _t(1.0), abar, tab, 0, prediction_type="trigflow", **kw)
    assert "must be one of" in _expect_err(
        solve, _t(0.5), _t(1.0), abar, tab, 0, prediction_type="EPSILON", **kw)


# ---------------------------------------------------------------- 元信息
check_full_loop_epsilon.example = True
check_one_step.example = True
