"""SCM trigflow 一致性步 -- 测试用例."""

import math

import numpy as np
import torch

_THETA_MAX = 1.57080  # arctan(80/0.5) 附近, Sana-Sprint 默认
_TAB2 = [1.57080, 1.3, 0.0]  # set_timesteps(2, intermediate=1.3) 默认表


def _t(v):
    return torch.tensor([[v, v * 0.5 - 0.3]], dtype=torch.float64)


def _mat(v, b=1):
    return torch.tensor([[v, v * 0.5 - 0.3]] * b, dtype=torch.float64)


def _tab(*vals):
    return torch.tensor(list(vals), dtype=torch.float64)


def _linspace(a, b, n):
    return torch.linspace(a, b, n, dtype=torch.float64)


def _close(actual, expected, tol=1e-9):
    a = np.asarray(torch.detach(actual).to(torch.float64).numpy(), dtype=np.float64).reshape(-1)
    e = np.broadcast_to(np.asarray(expected, dtype=np.float64).reshape(-1), a.shape)
    return bool(np.all(np.abs(a - e) <= tol))


def _oracle(mo, x, tab, i, noise, sd):
    """独立 oracle: math.cos/sin 逐元素实现 (mo/x/noise 为 1x2 list-of-list 数值)."""
    s = float(tab[i])
    t = float(tab[i + 1])
    x0 = [[math.cos(s) * x[b][j] - math.sin(s) * mo[b][j] for j in range(len(x[0]))]
          for b in range(len(x))]
    if noise is None:
        nt = [[0.0] * len(x[0]) for _ in x]
    else:
        nt = [[math.sin(t) * sd * noise[b][j] for j in range(len(x[0]))] for b in range(x.__len__())]
    prev = [[math.cos(t) * x0[b][j] + nt[b][j] for j in range(len(x[0]))] for b in range(len(x))]
    return x0, prev, nt, s, t


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 全循环对拍
def check_two_step_sana_sprint(solve):
    """官方 set_timesteps(2, intermediate_timesteps=1.3) 默认表, 两步全循环."""
    mo1, mo2 = 0.7, -0.4
    x0v, n1, n2 = 1.2, 0.3, -0.25
    tab = _tab(*_TAB2)
    # step 0: s=1.5708, t=1.3
    out1 = solve(_t(mo1), _t(2.0), tab, 0, noise=_t(n1))
    ox0, oprev, ont, s, t = _oracle([[mo1, mo1 * 0.5 - 0.3]], [[2.0, 0.7]], _TAB2, 0,
                                    [[n1, n1 * 0.5 - 0.3]], 0.5)
    assert _close(out1["pred_original_sample"], ox0)
    assert _close(out1["prev_sample"], oprev)
    assert _close(out1["noise_term"], ont)
    assert abs(out1["s_angle"] - s) < 1e-12 and abs(out1["t_angle"] - t) < 1e-12
    # step 1: s=1.3, t=0 -> 噪声项消失; oracle 输入行取 prev 实际两列值
    out2 = solve(_t(mo2), out1["prev_sample"], tab, 1, noise=_t(n2))
    pv = out1["prev_sample"].to(torch.float64).numpy().tolist()
    ox0b, oprevb, _, s2, t2 = _oracle([[mo2, mo2 * 0.5 - 0.3]], pv, _TAB2, 1,
                                      [[n2, n2 * 0.5 - 0.3]], 0.5)
    assert _close(out2["pred_original_sample"], ox0b)
    assert _close(out2["prev_sample"], oprevb)
    assert t2 == 0.0 and s2 == 1.3
    # 末步 prev == pred_x0 (cos(0)=1, sin(0)=0 逐位精确)
    assert _close(out2["prev_sample"], out2["pred_original_sample"], tol=0.0)


def check_linspace_three_step(solve):
    """linspace(max, 0, 4) 三步全循环对拍 (官方默认构造)."""
    tab = _linspace(_THETA_MAX, 0.0, 4)
    x = 2.0
    noises = [0.4, -0.2, 0.15]
    for i in range(3):
        mo = 0.3 * x - 0.5
        nz = noises[i]
        out = solve(_t(mo), _t(x), tab, i, noise=_t(nz))
        row = [[mo, mo * 0.5 - 0.3]]
        xr = [[x, x * 0.5 - 0.3]]
        nr = [[nz, nz * 0.5 - 0.3]]
        ox0, oprev, ont, s, t = _oracle(row, xr, tab.tolist(), i, nr, 0.5)
        assert _close(out["pred_original_sample"], ox0), f"step {i} x0"
        assert _close(out["prev_sample"], oprev), f"step {i} prev"
        assert _close(out["noise_term"], ont), f"step {i} noise_term"
        assert abs(out["s_angle"] - s) < 1e-12 and abs(out["t_angle"] - t) < 1e-12
        x = float(out["prev_sample"].reshape(-1)[0])


def check_single_step_identity(solve):
    """单步表 [theta_max, 0], noise=None 合法, prev == pred_x0 (Sana-Sprint 单步)."""
    tab = _tab(_THETA_MAX, 0.0)
    out = solve(_t(0.6), _t(1.7), tab, 0, noise=None)
    ox0, _, _, s, t = _oracle([[0.6, 0.0]], [[1.7, 0.55]], [1.57080, 0.0], 0, None, 0.5)
    assert _close(out["pred_original_sample"], ox0)
    assert _close(out["prev_sample"], ox0, tol=0.0)
    assert out["s_angle"] == 1.57080 and out["t_angle"] == 0.0
    assert _close(out["noise_term"], 0.0, tol=0.0)


def check_noise_given_but_t_zero(solve):
    """noise 非 None 但 t==0: 合法 (官方采样了但乘零), prev 与 noise 无关."""
    tab = _tab(_THETA_MAX, 0.0)
    out_a = solve(_t(0.6), _t(1.7), tab, 0, noise=_t(9.9))
    out_b = solve(_t(0.6), _t(1.7), tab, 0, noise=_t(-3.3))
    assert torch.equal(out_a["prev_sample"], out_b["prev_sample"])
    assert _close(out_a["noise_term"], 0.0, tol=0.0)


def check_trigflow_roundtrip(solve):
    """trigflow 合成一致性: x = cos(s)x0 + sin(s)sd*eps, mo = sd*cos(s)eps - sin(s)x0 -> 恢复 x0."""
    sd = 0.5
    s = 0.9
    x0 = [[1.3, -0.8]]
    eps = [[0.4, 0.9]]
    x = [[math.cos(s) * x0[0][j] + math.sin(s) * sd * eps[0][j] for j in range(2)]]
    mo = [[sd * math.cos(s) * eps[0][j] - math.sin(s) * x0[0][j] for j in range(2)]]
    tab = _tab(s, 0.0)
    out = solve(torch.tensor(mo, dtype=torch.float64), torch.tensor(x, dtype=torch.float64),
                tab, 0, noise=None)
    assert _close(out["pred_original_sample"], x0, tol=1e-12)


def check_sigma_data_scaling(solve):
    """sigma_data 只缩放噪声项: prev(2.0) - prev(1.0) = sin(t)*(2-1)*noise (逐列)."""
    tab = _tab(*_TAB2)
    mo, x, nz = 0.5, 1.1, 0.35
    nz_row = np.array([nz, nz * 0.5 - 0.3])
    out1 = solve(_t(mo), _t(x), tab, 0, noise=_t(nz), sigma_data=1.0)
    out2 = solve(_t(mo), _t(x), tab, 0, noise=_t(nz), sigma_data=2.0)
    assert torch.equal(out1["pred_original_sample"], out2["pred_original_sample"])
    diff = out2["prev_sample"] - out1["prev_sample"]
    assert _close(diff, math.sin(1.3) * 1.0 * nz_row)
    # 官方默认 sigma_data=0.5
    out_d = solve(_t(mo), _t(x), tab, 0, noise=_t(nz))
    x0_row = out_d["pred_original_sample"].to(torch.float64).numpy().reshape(-1)
    expected = math.cos(1.3) * x0_row + math.sin(1.3) * 0.5 * nz_row
    assert _close(out_d["prev_sample"], expected)


def check_adjacent_equal_angles(solve):
    """相邻相等角度合法 (官方 index_for_timestep 专门处理重复值)."""
    tab = _tab(0.9, 0.9, 0.0)
    out = solve(_t(0.4), _t(1.0), tab, 0, noise=_t(0.2))
    ox0, oprev, _, s, t = _oracle([[0.4, -0.1]], [[1.0, 0.2]], [0.9, 0.9, 0.0], 0,
                                  [[0.2, -0.2]], 0.5)
    assert s == 0.9 and t == 0.9
    assert _close(out["pred_original_sample"], ox0)
    assert _close(out["prev_sample"], oprev)


def check_pure_function(solve):
    """纯函数: 无隐藏状态, 同参数两次调用逐位相同 (官方 _step_index 状态在判题域外置)."""
    tab = _tab(*_TAB2)
    kw = dict(noise=_t(0.3), sigma_data=0.5)
    a = solve(_t(0.5), _t(1.0), tab, 0, **kw)
    b = solve(_t(0.5), _t(1.0), tab, 0, **kw)
    for k in ("prev_sample", "pred_original_sample", "noise_term"):
        assert torch.equal(a[k], b[k]), k
    assert a["s_angle"] == b["s_angle"] and a["t_angle"] == b["t_angle"]


# ---------------------------------------------------------------- 入参校验
def check_table_validation(solve):
    """timesteps: 类型/维度/长度/dtype/单调不增."""
    good_tab = _tab(*_TAB2)
    assert "必须是 torch 张量" in _expect_err(solve, np.zeros(3), _t(1.0), good_tab, 0, noise=_t(0.1))
    assert "timesteps 必须是 torch 张量" in _expect_err(solve, _t(0.5), _t(1.0), _TAB2, 0, noise=_t(0.1))
    assert "必须是 torch 张量" in _expect_err(solve, _t(0.5), np.zeros((1, 2)), good_tab, 0, noise=_t(0.1))
    assert "一维" in _expect_err(solve, _t(0.5), _t(1.0), good_tab.unsqueeze(0), 0, noise=_t(0.1))
    assert "表长" in _expect_err(solve, _t(0.5), _t(1.0), _tab(1.0), 0, noise=_t(0.1))
    assert "浮点" in _expect_err(solve, _t(0.5), _t(1.0), torch.tensor([2, 1, 0], dtype=torch.int64), 0,
                                noise=_t(0.1))
    assert "单调不增" in _expect_err(solve, _t(0.5), _t(1.0), _tab(0.0, 0.7, 1.3), 0, noise=_t(0.1))


def check_step_index_validation(solve):
    """step_index: int 类型与范围."""
    tab = _tab(*_TAB2)
    kw = dict(noise=_t(0.1))
    assert "必须是整数" in _expect_err(solve, _t(0.5), _t(1.0), tab, True, **kw)
    assert "必须是整数" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0.0, **kw)
    assert "必须是整数" in _expect_err(solve, _t(0.5), _t(1.0), tab, "0", **kw)
    assert "必须落在" in _expect_err(solve, _t(0.5), _t(1.0), tab, -1, **kw)
    assert "必须落在" in _expect_err(solve, _t(0.5), _t(1.0), tab, 2, **kw)  # len 3 -> [0, 1]
    # 合法上界: len-2
    out = solve(_t(0.5), _t(1.0), tab, 1, **kw)
    assert out["t_angle"] == 0.0


def check_noise_validation(solve):
    """noise: t!=0 必须显式给; 形状一致; 类型."""
    tab = _tab(*_TAB2)
    assert "必须显式给 noise" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0, noise=None)
    assert "必须是 torch 张量" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0,
                                            noise=np.zeros((1, 2)))
    assert "形状必须一致" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0,
                                        noise=torch.zeros((2, 2), dtype=torch.float64))
    assert "形状必须一致" in _expect_err(solve, torch.zeros((1, 3), dtype=torch.float64),
                                        _t(1.0), tab, 0, noise=_t(0.1))
    # noise=None 在 t==0 合法
    out = solve(_t(0.5), _t(1.0), tab, 1, noise=None)
    assert _close(out["noise_term"], 0.0, tol=0.0)


def check_misc_validation(solve):
    """prediction_type / sigma_data."""
    tab = _tab(*_TAB2)
    kw = dict(noise=_t(0.1))
    assert "Unsupported parameterization" in _expect_err(
        solve, _t(0.5), _t(1.0), tab, 0, prediction_type="epsilon", **kw)
    assert "Unsupported parameterization" in _expect_err(
        solve, _t(0.5), _t(1.0), tab, 0, prediction_type="v_prediction", **kw)
    assert "必须是数值" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0, sigma_data="0.5", **kw)
    assert "必须是数值" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0, sigma_data=True, **kw)
    assert "必须 > 0" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0, sigma_data=0.0, **kw)
    assert "必须 > 0" in _expect_err(solve, _t(0.5), _t(1.0), tab, 0, sigma_data=-0.5, **kw)


# ---------------------------------------------------------------- 元信息
check_two_step_sana_sprint.example = True
check_single_step_identity.example = True
