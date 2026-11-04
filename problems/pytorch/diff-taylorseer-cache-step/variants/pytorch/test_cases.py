"""TaylorSeer 缓存步 -- 测试用例."""

import math

import numpy as np
import torch

# 独立 oracle ----------------------------------------------------------------


def _gates(step, before, interval, after):
    is_warmup = step < before
    is_interval = (step - before - 1) % interval == 0
    is_cooldown = after is not None and step >= after
    return is_warmup or is_interval or is_cooldown


def _oracle_update(factors, lus, outputs, step, max_order):
    """独立递推: factors 为 {order: np.array}."""
    new = {0: np.asarray(outputs, dtype=np.float64).copy()}
    if lus is not None:
        delta = step - lus
        for j in range(max_order):
            if j not in factors or factors[j].shape != new[j].shape:
                break
            new[j + 1] = (new[j] - factors[j].astype(np.float64)) / delta
    return {o: t for o, t in new.items()}, step


def _oracle_predict(factors, lus, step):
    offset = step - lus
    orders = sorted(factors)
    out = np.zeros_like(factors[orders[0]], dtype=np.float64)
    for o in orders:
        out = out + factors[o].astype(np.float64) * (offset ** o) / math.factorial(o)
    return out


def _close(actual, expected, tol=1e-9):
    a = np.asarray(torch.detach(actual).to(torch.float64).numpy(), dtype=np.float64).reshape(-1)
    e = np.broadcast_to(np.asarray(expected, dtype=np.float64).reshape(-1), a.shape)
    return bool(np.all(np.abs(a - e) <= tol))


def _t(v):
    return torch.tensor([[float(v), v * 0.5 - 0.3]], dtype=torch.float64)


def _gen_output(step, shape=(1, 2)):
    """确定性生成第 step 步的原始输出 (平滑变化, 便于差分近似有意义)."""
    v = 0.7 * step - 0.05 * step * step
    return torch.tensor([[v, v * 0.5 - 0.3]], dtype=torch.float64)


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 全循环对拍
def check_full_loop_order2(solve):
    """before=3, interval=5, max_order=2, step 0..11 全循环对拍."""
    st = None
    for step in range(12):
        out_t = _gen_output(step)
        r = solve(out_t, st, step, cache_interval=5,
                  disable_cache_before_step=3, max_order=2)
        compute = _gates(step, 3, 5, None)
        assert r["computed"] is compute, f"step {step} gate"
        if compute:
            # oracle 递推
            prev_factors = {} if st is None else {int(k): v.to(torch.float64).numpy()
                                                  for k, v in st["taylor_factors"].items()}
            lus = None if st is None else st["last_update_step"]
            new_f, new_lus = _oracle_update(prev_factors, lus,
                                            out_t.to(torch.float64).numpy(), step, 2)
            assert _close(r["output"], out_t.to(torch.float64).numpy(), tol=0.0), f"step {step}"
            st = r["state"]
            assert st["last_update_step"] == new_lus
            assert set(st["taylor_factors"]) == set(new_f), f"step {step} orders {sorted(st['taylor_factors'])}"
            for o, arr in new_f.items():
                assert _close(st["taylor_factors"][o], arr), f"step {step} order {o}"
        else:
            expected = _oracle_predict(
                {int(k): v.to(torch.float64).numpy() for k, v in st["taylor_factors"].items()},
                st["last_update_step"], step)
            assert _close(r["output"], expected), f"step {step} predict"
            st = r["state"]
    # compute 步: 0,1,2 (warmup) + 4, 9; 3/5/6/7/8/10/11 predict
    # (由上面的 computed 断言覆盖)


def check_warmup_updates_factors(solve):
    """warmup 步 (0-2) 也更新因子: step 3 的 predict 必须用到 f[0..2]."""
    st = None
    for step in range(3):
        r = solve(_gen_output(step), st, step, cache_interval=5,
                  disable_cache_before_step=3, max_order=2)
        st = r["state"]
    assert set(st["taylor_factors"]) == {0, 1, 2}
    r3 = solve(_gen_output(3), st, 3, cache_interval=5,
               disable_cache_before_step=3, max_order=2)
    assert r3["computed"] is False
    # 手算: L=2, offset=1; f2 由 (f1_new - f1_old)/delta 递推
    f = {int(k): v.to(torch.float64).numpy() for k, v in st["taylor_factors"].items()}
    expected = _oracle_predict(f, 2, 3)
    assert _close(r3["output"], expected)


def check_interval_alignment(solve):
    """interval 对齐: before=1, interval=3 -> compute 步 0, 2, 5, 8 (step 3/4 不是)."""
    st = None
    computed_steps = []
    for step in range(10):
        r = solve(_gen_output(step), st, step, cache_interval=3,
                  disable_cache_before_step=1, max_order=1)
        if r["computed"]:
            computed_steps.append(step)
        st = r["state"]
    assert computed_steps == [0, 2, 5, 8], computed_steps


def check_cooldown(solve):
    """cooldown: after=6 时 step>=6 强制 compute (即使 interval 不对齐)."""
    st = None
    computed_steps = []
    for step in range(9):
        r = solve(_gen_output(step), st, step, cache_interval=5,
                  disable_cache_before_step=3, disable_cache_after_step=6, max_order=1)
        if r["computed"]:
            computed_steps.append(step)
        st = r["state"]
    assert computed_steps == [0, 1, 2, 4, 6, 7, 8], computed_steps


# ---------------------------------------------------------------- 递推与外推
def check_order1_predict_linear(solve):
    """max_order=1: predict = f0 + offset * f1 (一阶外推)."""
    # 构造 state: L=4, f0/f1 已知
    f0 = _t(1.0)
    f1 = _t(0.4)
    st = {"taylor_factors": {0: f0, 1: f1}, "last_update_step": 4}
    r = solve(_gen_output(6), st, 6, cache_interval=5,
              disable_cache_before_step=3, max_order=1)
    assert r["computed"] is False
    assert _close(r["output"], np.asarray([[1.0, 0.2]]) + 2 * np.asarray([[0.4, -0.1]]))


def check_order2_predict_quadratic(solve):
    """max_order=2: 二阶项 offset^2/2 * f2."""
    f0 = _t(1.0)
    f1 = _t(0.4)
    f2 = _t(-0.2)
    st = {"taylor_factors": {0: f0, 1: f1, 2: f2}, "last_update_step": 4}
    r = solve(_gen_output(7), st, 7, cache_interval=5,
              disable_cache_before_step=3, max_order=2)
    base = np.asarray([[1.0, 0.2]]) + 3 * np.asarray([[0.4, -0.1]]) \
        + 9.0 / 2 * np.asarray([[-0.2, -0.4]])
    assert _close(r["output"], base)
    # state 未变 (predict 步)
    assert r["state"]["last_update_step"] == 4
    assert set(r["state"]["taylor_factors"]) == {0, 1, 2}


def check_shape_change_restart(solve):
    """两次 compute 间输出形状变化: 因子重启 (只留 order 0)."""
    st = None
    r0 = solve(_t(0.5), st, 0, cache_interval=5, disable_cache_before_step=1, max_order=2)
    st = r0["state"]  # step 0: warmup compute, f={0}
    r1 = solve(_t(0.7), st, 1, cache_interval=5, disable_cache_before_step=1, max_order=2)
    st = r1["state"]  # before=1: (1-2)%5=4 -> predict
    assert r1["computed"] is False
    r2 = solve(_t(0.9), st, 2, cache_interval=5, disable_cache_before_step=1, max_order=2)
    st = r2["state"]
    assert r2["computed"] is True and set(st["taylor_factors"]) == {0, 1}
    # 形状变化: (2,2) 在下一个 compute 步 (step 7)
    r3 = solve(torch.zeros((2, 2), dtype=torch.float64), st, 7,
               cache_interval=5, disable_cache_before_step=1, max_order=2)
    st3 = r3["state"]
    assert r3["computed"] is True
    assert set(st3["taylor_factors"]) == {0}, "形状变化应重启展开"


def check_state_purity(solve):
    """纯函数式: 同入参两次调用结果一致, 且入参 state 不被修改."""
    st_in = {"taylor_factors": {0: _t(1.0), 1: _t(0.4)}, "last_update_step": 4}
    snapshot = {int(k): v.clone() for k, v in st_in["taylor_factors"].items()}
    a = solve(_gen_output(6), st_in, 6, cache_interval=5, disable_cache_before_step=3)
    b = solve(_gen_output(6), st_in, 6, cache_interval=5, disable_cache_before_step=3)
    assert torch.equal(a["output"], b["output"])
    assert a["computed"] == b["computed"]
    # 入参未被修改
    assert set(st_in["taylor_factors"]) == set(snapshot)
    for k, v in snapshot.items():
        assert torch.equal(st_in["taylor_factors"][k], v)
    assert st_in["last_update_step"] == 4


# ---------------------------------------------------------------- 边界与校验
def check_empty_predict_reject(solve):
    """before=0 且首步不对齐 -> predict 无初始化 -> raise."""
    err = _expect_err(solve, _t(0.5), None, 0, cache_interval=5,
                      disable_cache_before_step=0)
    assert "Cannot predict without prior initialization" in err
    # before=0 时 (step-1)%5==0 -> step 1, 6, ... 是 compute 步
    r = solve(_t(0.5), None, 1, cache_interval=5, disable_cache_before_step=0)
    assert r["computed"] is True and r["state"]["last_update_step"] == 1


def check_delta_zero_reject(solve):
    """compute 步且 last_update_step == step -> delta 0 raise."""
    st = {"taylor_factors": {0: _t(1.0)}, "last_update_step": 4}
    err = _expect_err(solve, _t(0.5), st, 4, cache_interval=5,
                      disable_cache_before_step=3)
    assert "Delta step cannot be zero" in err


def check_state_validation(solve):
    """state 结构校验."""
    kw = dict(cache_interval=5, disable_cache_before_step=0)
    # 非法 state 类型
    assert "state 必须是 dict" in _expect_err(solve, _t(0.5), [], 0, **kw)
    # 未知键
    assert "未知键" in _expect_err(solve, _t(0.5), {"foo": 1}, 0, **kw)
    # factors 非字典
    assert "必须是 dict" in _expect_err(
        solve, _t(0.5), {"taylor_factors": [_t(0.1)], "last_update_step": None}, 0, **kw)
    # 阶数键非法
    assert "非负整数" in _expect_err(
        solve, _t(0.5), {"taylor_factors": {-1: _t(0.1)}, "last_update_step": None}, 0, **kw)
    assert "非负整数" in _expect_err(
        solve, _t(0.5), {"taylor_factors": {True: _t(0.1)}, "last_update_step": None}, 0, **kw)
    # 值非张量
    assert "torch 张量" in _expect_err(
        solve, _t(0.5), {"taylor_factors": {0: 0.1}, "last_update_step": None}, 0, **kw)
    # last_update_step 类型
    assert "last_update_step" in _expect_err(
        solve, _t(0.5), {"taylor_factors": {}, "last_update_step": True}, 0, **kw)
    assert "last_update_step" in _expect_err(
        solve, _t(0.5), {"taylor_factors": {}, "last_update_step": -1}, 0, **kw)
    # last_update_step 非 None 但 factors 空 -> predict 报初始化缺失
    # (构造 predict 步: before=0, step=1: (1-1)%5=0 -> compute; step=2 predict)
    err = _expect_err(solve, _t(0.5), {"taylor_factors": {}, "last_update_step": 1}, 2,
                      cache_interval=5, disable_cache_before_step=0)
    assert "Cannot predict" in err
    # 反向不一致: last_update_step 为 None 但 factors 非空 -> 官方同样 raise
    # (predict 步: before=3, step=3: (3-4)%5=4 -> predict)
    err = _expect_err(solve, _t(0.5), {"taylor_factors": {0: _t(1.0)}, "last_update_step": None}, 3,
                      cache_interval=5, disable_cache_before_step=3)
    assert "Cannot predict" in err


def check_args_validation(solve):
    """outputs/step/interval/before/after/max_order 校验."""
    good_st = {"taylor_factors": {0: _t(1.0)}, "last_update_step": 4}
    assert "outputs 必须是 torch 张量" in _expect_err(
        solve, 0.5, None, 0, cache_interval=5, disable_cache_before_step=3)
    assert "step 必须是非负整数" in _expect_err(
        solve, _t(0.5), None, -1, cache_interval=5, disable_cache_before_step=3)
    assert "step 必须是非负整数" in _expect_err(
        solve, _t(0.5), None, True, cache_interval=5, disable_cache_before_step=3)
    assert "step 必须是非负整数" in _expect_err(
        solve, _t(0.5), None, 0.0, cache_interval=5, disable_cache_before_step=3)
    assert "cache_interval" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=0, disable_cache_before_step=3)
    assert "cache_interval" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=True, disable_cache_before_step=3)
    assert "disable_cache_before_step" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=5, disable_cache_before_step=-1)
    assert "disable_cache_before_step" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=5, disable_cache_before_step=True)
    assert "disable_cache_after_step" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=5, disable_cache_before_step=3,
        disable_cache_after_step=-2)
    assert "max_order" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=5, disable_cache_before_step=3, max_order=0)
    assert "max_order" in _expect_err(
        solve, _t(0.5), None, 0, cache_interval=5, disable_cache_before_step=3, max_order=True)
    # 合法 after=None 默认
    r = solve(_t(0.5), None, 0, cache_interval=5, disable_cache_before_step=3)
    assert r["computed"] is True


def check_before_zero_alignment(solve):
    """before=0: step 0 是 predict (无初始化 raise), 对齐步 compute."""
    err = _expect_err(solve, _t(0.5), None, 0, cache_interval=4,
                      disable_cache_before_step=0)
    assert "Cannot predict" in err
    # step 3: (3-1)%4=2 -> predict; step 4: (4-1)%4=3; step 5: 0 -> compute
    r = solve(_t(0.5), None, 5, cache_interval=4, disable_cache_before_step=0)
    assert r["computed"] is True


# ---------------------------------------------------------------- 元信息
check_full_loop_order2.example = True
check_order1_predict_linear.example = True
