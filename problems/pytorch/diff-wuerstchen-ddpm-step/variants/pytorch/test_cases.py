import math

import numpy as np
import torch


def _oracle_ac(t, scaler, s):
    """独立 numpy 重写官方 _alpha_cumprod（scaler 幂变换 + cosine 平方 +
    归一化 + clamp）。"""
    if scaler > 1:
        t = 1 - (1 - t) ** scaler
    elif scaler < 1:
        t = t**scaler
    init = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    ac = math.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2 / init
    return min(max(ac, 0.0001), 0.9999)


def _oracle(mo, timestep, sample, table, noise, scaler=1.0, s=0.008):
    """独立 numpy 重写 DDPMWuerstchen 单步（公式从 diffusers
    scheduling_ddpm_wuerstchen.py 的 step L202-221 / _alpha_cumprod L121-129 /
    previous_timestep L255-269 数学定义直接推出）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    nz = np.asarray(noise, dtype=np.float64)
    tab = np.asarray(table, dtype=np.float64)
    t0 = float(np.asarray(timestep, dtype=np.float64).reshape(-1)[0])

    index = int(np.argmin(np.abs(tab - t0)))
    prev_t = float(tab[index + 1])

    ac = _oracle_ac(t0, scaler, s)
    acp = _oracle_ac(prev_t, scaler, s)
    alpha = ac / acp

    mu = np.sqrt(1.0 / alpha) * (x - (1 - alpha) * mo_ / np.sqrt(1 - ac))
    std = np.sqrt((1 - alpha) * (1 - acp) / (1 - ac)) * nz
    gate = 1.0 if prev_t != 0.0 else 0.0
    prev = mu + std * gate

    return {
        "prev_sample": prev,
        "alpha_cumprod": ac,
        "alpha_cumprod_prev": acp,
        "alpha_ratio": alpha,
        "mu": mu,
        "noise_applied": prev_t != 0.0,
        "prev_t": prev_t,
    }


def _t(v, batch=1):
    return torch.full((batch, 1), float(v), dtype=torch.float64)


def _ts(v, batch=1):
    return torch.full((batch,), float(v), dtype=torch.float64)


def _table(*vals):
    return torch.tensor(list(vals), dtype=torch.float64)


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
    assert _close(out["mu"], o["mu"]), f"{tag} mu"
    assert _close(out["alpha_cumprod"], o["alpha_cumprod"]), f"{tag} ac"
    assert _close(out["alpha_cumprod_prev"], o["alpha_cumprod_prev"]), f"{tag} acp"
    assert _close(out["alpha_ratio"], o["alpha_ratio"]), f"{tag} alpha"
    assert out["noise_applied"] == o["noise_applied"], f"{tag} noise_applied"
    assert out["prev_t"] == o["prev_t"], f"{tag} prev_t"


_TAB = (1.0, 0.75, 0.5, 0.25, 0.0)  # 官方 linspace(1, 0, N+1)，N=4


def check_first_step(solve):
    """示例：t=1（最噪端）：ᾱ(1)=0 → clamp 下界 0.0001 精确断言 + 全输出对拍。"""
    out = solve(_t(0.5), _ts(1.0), _t(3.0), _table(*_TAB), noise=_t(0.2))
    o = _oracle(0.5, [1.0], 3.0, _TAB, 0.2)
    _assert_out_matches(out, o, "first")
    assert _close(out["alpha_cumprod"], 0.0001), "ᾱ(1) clamp 下界"
    assert out["noise_applied"] is True and out["prev_t"] == 0.75
    # 闭式独立验证 mu 与 prev
    acp = o["alpha_cumprod_prev"]
    alpha = 0.0001 / acp
    mu = math.sqrt(1 / alpha) * (3.0 - (1 - alpha) * 0.5 / math.sqrt(1 - 0.0001))
    std = math.sqrt((1 - alpha) * (1 - acp) / (1 - 0.0001)) * 0.2
    assert _close(out["mu"], mu)
    assert _close(out["prev_sample"], mu + std)


check_first_step.example = True


def check_last_step_gate(solve):
    """末步门控：prev_t 精确为 0 → 不加噪声，换 noise 输出不变（独立断言）。"""
    out = solve(_t(0.3), _ts(0.25), _t(1.2), _table(*_TAB), noise=_t(0.4))
    o = _oracle(0.3, [0.25], 1.2, _TAB, 0.4)
    _assert_out_matches(out, o, "last")
    assert out["noise_applied"] is False and out["prev_t"] == 0.0
    assert _close(out["prev_sample"], out["mu"]), "末步 prev == mu"
    out2 = solve(_t(0.3), _ts(0.25), _t(1.2), _table(*_TAB), noise=_t(9.9))
    assert _close(out2["prev_sample"], out["prev_sample"]), "末步换 noise 不变"
    assert _close(out2["mu"], out["mu"]) and not _close(out2["alpha_ratio"], 0.0)


def check_mid_step(solve):
    """中间步：t=0.5、prev=0.25，闭式 ᾱ 与 α < 1 断言 + 对拍。"""
    out = solve(_t(-0.2), _ts(0.5), _t(0.8), _table(*_TAB), noise=_t(0.1))
    o = _oracle(-0.2, [0.5], 0.8, _TAB, 0.1)
    _assert_out_matches(out, o, "mid")
    s = 0.008
    init = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    ac = math.cos((0.5 + s) / (1 + s) * math.pi * 0.5) ** 2 / init
    acp = math.cos((0.25 + s) / (1 + s) * math.pi * 0.5) ** 2 / init
    assert _close(out["alpha_cumprod"], ac), "ᾱ(0.5) 闭式"
    assert _close(out["alpha_cumprod_prev"], acp), "ᾱ(0.25) 闭式"
    assert float(out["alpha_ratio"]) < 1.0, "α = ᾱ_t/ᾱ_prev < 1（t 越大 ᾱ 越小）"
    mu = math.sqrt(1 / (ac / acp)) * (0.8 - (1 - ac / acp) * (-0.2) / math.sqrt(1 - ac))
    assert _close(out["mu"], mu), "mu 闭式"


def check_identity_step(solve):
    """恒等步：表 [1, 0, 0]、t=0 → argmin 平局取首个 0（index=1）、
    ᾱ(0)=1 → clamp 上界 0.9999、α=1 → mu=sample、prev=sample。"""
    out = solve(_t(0.3), _ts(0.0), _t(1.2), _table(1.0, 0.0, 0.0), noise=_t(0.5))
    o = _oracle(0.3, [0.0], 1.2, (1.0, 0.0, 0.0), 0.5)
    _assert_out_matches(out, o, "identity")
    assert out["prev_t"] == 0.0 and out["noise_applied"] is False
    assert _close(out["alpha_cumprod"], 0.9999), "ᾱ(0) clamp 上界"
    assert _close(out["alpha_cumprod_prev"], 0.9999)
    assert _close(out["alpha_ratio"], 1.0)
    assert _close(out["mu"], 1.2), "α=1 → mu = sample"
    assert _close(out["prev_sample"], 1.2), "恒等步 prev = sample"


def check_scaler_gt1(solve):
    """scaler=2（prior 级）：t' = 1−(1−t)²，闭式验证 + 对拍。"""
    out = solve(_t(0.4), _ts(0.5), _t(1.0), _table(*_TAB), noise=_t(0.3), scaler=2.0)
    o = _oracle(0.4, [0.5], 1.0, _TAB, 0.3, scaler=2.0)
    _assert_out_matches(out, o, "scaler>1")
    s = 0.008
    init = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    tp = 1 - (1 - 0.5) ** 2
    assert _close(out["alpha_cumprod"], math.cos((tp + s) / (1 + s) * math.pi * 0.5) ** 2 / init)


def check_scaler_lt1(solve):
    """scaler=0.5：t' = √t，闭式验证 + 对拍。"""
    out = solve(_t(0.4), _ts(0.25), _t(1.0), _table(*_TAB), noise=_t(0.3), scaler=0.5)
    o = _oracle(0.4, [0.25], 1.0, _TAB, 0.3, scaler=0.5)
    _assert_out_matches(out, o, "scaler<1")
    s = 0.008
    init = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    tp = math.sqrt(0.25)
    assert _close(out["alpha_cumprod"], math.cos((tp + s) / (1 + s) * math.pi * 0.5) ** 2 / init)


def check_batch(solve):
    """B=2 同 t：广播一致 + 对拍；混合 batch 拒绝。"""
    out = solve(torch.full((2, 1), 0.4, dtype=torch.float64), _ts(0.5, 2),
                torch.tensor([[1.0], [2.0]], dtype=torch.float64), _table(*_TAB),
                noise=torch.tensor([[0.1], [-0.2]], dtype=torch.float64))
    o1 = _oracle(0.4, [0.5], 1.0, _TAB, 0.1)
    o2 = _oracle(0.4, [0.5], 2.0, _TAB, -0.2)
    assert _close(out["prev_sample"][0], o1["prev_sample"]), "batch[0]"
    assert _close(out["prev_sample"][1], o2["prev_sample"]), "batch[1]"
    assert _close(out["alpha_cumprod"][0], out["alpha_cumprod"][1])
    try:
        solve(torch.full((2, 1), 0.4, dtype=torch.float64),
              torch.tensor([0.5, 0.25], dtype=torch.float64),
              torch.full((2, 1), 1.0, dtype=torch.float64), _table(*_TAB),
              noise=torch.full((2, 1), 0.1, dtype=torch.float64))
    except ValueError as e:
        assert "batch 内 timestep 必须全部相等" in str(e), str(e)
    else:
        raise AssertionError("混合 batch 应抛 ValueError")


def check_full_loop(solve):
    """示例：N=4 步全循环（纯函数无状态），逐步对拍 + 末步门控。"""
    tab = list(_TAB)
    cur = 2.0
    for i, t_val in enumerate(tab[:-1]):
        mo = 0.25 * cur + 0.5
        noise = 0.11 * (i + 1) - 0.3
        out = solve(_t(mo), _ts(t_val), _t(cur), _table(*tab), noise=_t(noise))
        o = _oracle(mo, [t_val], cur, tab, noise)
        _assert_out_matches(out, o, f"loop {i}")
        # 独立断言：prev_t 恰是表中下一个值；门控标志与表尾一致
        assert out["prev_t"] == tab[i + 1], f"loop {i} prev_t"
        assert out["noise_applied"] == (tab[i + 1] != 0.0), f"loop {i} gate"
        cur = float(out["prev_sample"])
    assert cur == float(out["mu"]), "末步 prev == mu（门控）"


check_full_loop.example = True


def check_table_rejects(solve):
    """表契约：长度 / 单调 / 有限 / 表尾无后继。"""
    mo, x, ts, nz = _t(0.3), _t(1.0), _ts(0.5), _t(0.1)

    def expect_err(*a, **kw):
        try:
            solve(*a, **kw)
        except ValueError as e:
            return str(e)
        raise AssertionError(f"{kw} 应抛 ValueError")

    assert "长度至少为 2" in expect_err(mo, ts, x, _table(0.5), noise=nz)
    assert "单调不增" in expect_err(mo, ts, x, _table(0.25, 0.5, 1.0), noise=nz)
    assert "全部有限" in expect_err(mo, ts, x, _table(1.0, float("nan"), 0.0), noise=nz)
    assert "必须是浮点张量" in expect_err(mo, ts, x, torch.tensor([1, 0], dtype=torch.int64), noise=nz)
    assert "一维张量" in expect_err(mo, ts, x, torch.tensor([[1.0, 0.0]], dtype=torch.float64), noise=nz)
    # t=0 是表尾（index=4，无 index+1）
    assert "表尾" in expect_err(mo, _ts(0.0), x, _table(*_TAB), noise=nz)


def check_timestep_rejects(solve):
    """timestep 契约：值域 / 有限 / 混合 batch / 维度 / dtype。"""
    mo, x, tab, nz = _t(0.3), _t(1.0), _table(*_TAB), _t(0.1)

    def expect_err(t, **kw):
        try:
            solve(mo, t, x, tab, noise=nz, **kw)
        except ValueError as e:
            return str(e)
        raise AssertionError("应抛 ValueError")

    assert "落在 [0, 1]" in expect_err(_ts(-0.1))
    assert "落在 [0, 1]" in expect_err(_ts(1.1))
    assert "全部有限" in expect_err(_ts(float("nan")))
    assert "batch 内 timestep 必须全部相等" in expect_err(torch.tensor([0.5, 0.25], dtype=torch.float64))
    assert "一维张量" in expect_err(torch.tensor([[0.5]], dtype=torch.float64))
    assert "必须是浮点张量" in expect_err(torch.tensor([0], dtype=torch.int64))


def check_param_rejects(solve):
    """noise / scaler / s / 形状校验。"""
    mo, x, tab, ts = _t(0.3), _t(1.0), _table(*_TAB), _ts(0.5)

    def expect_err(*a, **kw):
        try:
            solve(*a, **kw)
        except ValueError as e:
            return str(e)
        raise AssertionError(f"{kw} 应抛 ValueError")

    assert "必须显式给 noise" in expect_err(mo, ts, x, tab, noise=None)
    assert "noise 形状" in expect_err(mo, ts, x, tab, noise=torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert "形状必须一致" in expect_err(mo, ts, torch.tensor([[1.0, 2.0]], dtype=torch.float64), tab, noise=_t(0.1))
    assert "必须是 torch 张量" in expect_err(0.3, ts, x, tab, noise=_t(0.1))
    assert "正有限" in expect_err(mo, ts, x, tab, noise=_t(0.1), scaler=0)
    assert "正有限" in expect_err(mo, ts, x, tab, noise=_t(0.1), scaler=-1.0)
    assert "必须是数值" in expect_err(mo, ts, x, tab, noise=_t(0.1), scaler=True)
    assert "正有限" in expect_err(mo, ts, x, tab, noise=_t(0.1), s=0)
    assert "必须是数值" in expect_err(mo, ts, x, tab, noise=_t(0.1), s=True)
