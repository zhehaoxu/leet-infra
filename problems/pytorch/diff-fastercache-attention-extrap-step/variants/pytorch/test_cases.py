"""FasterCache 注意力外推广播步 -- 测试用例."""

import torch


def _mat(v, rows):
    """确定性 (rows, 4) 张量: 元素 = v + 0.1*row + 0.01*col."""
    base = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    row_idx = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    col_idx = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    return (v + 0.1 * row_idx + 0.01 * col_idx).contiguous()


def _cond(full):
    return full[2:]  # batch 4 的 conditional 半批 (rows 2:4)


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


def _fresh(**over):
    st = {"iteration": 0, "batch_size": None, "cache": None}
    st.update(over)
    return st


# ---------------------------------------------------------------- 完整 batch 舞步
def check_full_dance(solve):
    """非蒸馏完整舞步: 整批(2B)与条件半批(B)交替, 切半批缓存 + 外推."""
    full_a, full_b = _mat(1.0, 4), _mat(2.0, 4)
    ca, cb = _cond(full_a), _cond(full_b)
    # iter0: 整批, iteration%2==0 -> compute; cache 取 conditional 半批
    r0 = solve(full_a, None, 500)
    assert r0["computed"] is True
    assert r0["output"] is r0["output"] and torch.equal(r0["output"], full_a)
    st = r0["state"]
    assert st["batch_size"] == 4 and st["iteration"] == 1
    assert len(st["cache"]) == 2
    assert torch.equal(st["cache"][0], ca) and torch.equal(st["cache"][1], ca)
    # iter1: 条件半批 (batch 2 != 4) -> skip; cache 两条同值 -> 外推 = ca
    r1 = solve(_mat(5.0, 2), st, 500)
    assert r1["computed"] is False
    assert torch.allclose(r1["output"], ca, atol=1e-6)
    st = r1["state"]
    assert st["batch_size"] == 4
    # iter2: 整批 -> compute; cache = [ca, cb]
    r2 = solve(full_b, st, 500)
    assert r2["computed"] is True
    st = r2["state"]
    assert torch.equal(st["cache"][0], ca) and torch.equal(st["cache"][1], cb)
    # iter3: 条件半批 -> skip; 外推 = cb + (cb - ca) * 0.5
    r3 = solve(_mat(6.0, 2), st, 500)
    assert r3["computed"] is False
    expected = cb + (cb - ca) * 0.5
    assert torch.allclose(r3["output"], expected, atol=1e-6), (r3["output"], expected)
    assert r3["state"]["iteration"] == 4
    # iter4: 整批 -> compute; cache 滚动为 [approx3, cond_d]
    # (槽滚动必须用旧 cache[1]=approx3 而非旧 cache[0]=cb——两者此处已不同)
    full_d = _mat(4.0, 4)
    cd = _cond(full_d)
    approx3 = expected
    r4 = solve(full_d, r3["state"], 500)
    assert r4["computed"] is True
    st = r4["state"]
    assert torch.allclose(st["cache"][0], approx3, atol=1e-6)
    assert torch.equal(st["cache"][1], cd)
    # iter5: 条件半批 -> skip; 外推参照 approx3 (非 cb)
    r5 = solve(_mat(7.0, 2), st, 500)
    assert r5["computed"] is False
    expected5 = cd + (cd - approx3) * 0.5
    assert torch.allclose(r5["output"], expected5, atol=1e-6), (r5["output"], expected5)


def check_distilled_sequence(solve):
    """蒸馏模型: batch 恒定, 缓存不做半批切分, skip 不需 batch 失配."""
    m1, m2, m3 = _mat(1.0, 2), _mat(2.0, 2), _mat(3.0, 2)
    kw = dict(is_guidance_distilled=True)
    r0 = solve(m1, None, 500, **kw)
    assert r0["computed"] is True
    st = r0["state"]
    assert st["batch_size"] == 2
    assert torch.equal(st["cache"][0], m1)  # 不切半批, 缓存整批 (此处即全量)
    r1 = solve(m2, st, 500, **kw)
    assert r1["computed"] is False  # 蒸馏时无需 batch 失配即可 skip
    assert torch.allclose(r1["output"], m1, atol=1e-6)
    r2 = solve(m3, r1["state"], 500, **kw)
    assert r2["computed"] is True
    st = r2["state"]
    assert torch.equal(st["cache"][0], m1) and torch.equal(st["cache"][1], m3)
    r3 = solve(m2, st, 500, **kw)
    expected = m3 + (m3 - m1) * 0.5
    assert torch.allclose(r3["output"], expected, atol=1e-6)


def check_weight_extremes(solve):
    """weight=0 -> 外推退化为 t_1; weight=1 -> 2*t_1 - t_2."""
    m1, m3 = _mat(1.0, 2), _mat(3.0, 2)
    kw0 = dict(is_guidance_distilled=True, weight=0.0)
    r0 = solve(m1, None, 500, **kw0)
    r1 = solve(m1, r0["state"], 500, **kw0)
    r2 = solve(m3, r1["state"], 500, **kw0)
    r3 = solve(m3, r2["state"], 500, **kw0)
    assert torch.allclose(r3["output"], m3, atol=1e-6)  # w=0: t_1
    kw1 = dict(is_guidance_distilled=True, weight=1.0)
    s0 = solve(m1, None, 500, **kw1)
    s1 = solve(m1, s0["state"], 500, **kw1)
    s2 = solve(m3, s1["state"], 500, **kw1)
    s3 = solve(m3, s2["state"], 500, **kw1)
    assert torch.allclose(s3["output"], 2 * m3 - m1, atol=1e-6)  # w=1


def check_open_interval(solve):
    """时间步开区间: t==lo 或 t==hi 时连条件半批调用也恒 compute
    (区间只「许可」broadcast, 端点必须封死); 区间内条件半批才真正 broadcast."""
    m = _mat(1.0, 4)
    for t in (-1, 681):
        r0 = solve(m, None, t)  # 整批: compute, batch_size=4
        st = r0["state"]
        for i in range(3):
            r = solve(_mat(5.0 + i, 2), st, t)  # 条件半批 + 奇 iteration
            assert r["computed"] is True, (t, i)  # 端点外禁 broadcast
            st = r["state"]
    # 区间内 (-1, 681) 的中间值: 条件半批真正 broadcast
    r0 = solve(m, None, 680)
    r1 = solve(_mat(5.0, 2), r0["state"], 680)
    assert r1["computed"] is False


def check_slice_path(solve):
    """cache 条目为整批 (size 4) 而当前 batch=2: 外推前切 conditional 半批 [2:]."""
    t2, t1 = _mat(1.0, 4), _mat(2.0, 4)
    st = _fresh(iteration=1, batch_size=4, cache=[t2, t1])  # 1%2=1 -> skip 候选
    r = solve(_mat(9.0, 2), st, 500)  # 4 != 2 -> 确认 skip
    assert r["computed"] is False
    expected = t1[2:] + (t1[2:] - t2[2:]) * 0.5
    assert torch.allclose(r["output"], expected, atol=1e-6), (r["output"], expected)
    # 切片是 [batch:] (conditional 在后), 不是 [:batch]
    assert not torch.allclose(r["output"], t1[:2] + (t1[:2] - t2[:2]) * 0.5, atol=1e-6)


def check_cache_slot_link(solve):
    """cache 两槽滚动: 新 cache[0] == 旧 cache[1]; 首步两槽同值."""
    full_a, full_b = _mat(1.0, 4), _mat(2.0, 4)
    r0 = solve(full_a, None, 500)
    st = r0["state"]
    assert torch.equal(st["cache"][0], st["cache"][1])  # 首步 [c, c]
    r1 = solve(_mat(5.0, 2), st, 500)  # skip: cache 值不变
    st = r1["state"]
    r2 = solve(full_b, st, 500)  # compute: cache = [旧[1], 新条件半批]
    st = r2["state"]
    assert torch.equal(st["cache"][0], _cond(full_a))
    assert torch.equal(st["cache"][1], _cond(full_b))


def check_pure_function(solve):
    """纯函数: 入参 state 不被修改, 同入参两次调用一致."""
    st_in = _fresh(batch_size=4, cache=[_mat(1.0, 4), _mat(2.0, 4)])
    snap = (st_in["iteration"], st_in["batch_size"],
            st_in["cache"][0].clone(), st_in["cache"][1].clone())
    a = solve(_mat(9.0, 2), st_in, 500)
    b = solve(_mat(9.0, 2), st_in, 500)
    assert torch.equal(a["output"], b["output"]) and a["computed"] == b["computed"]
    assert st_in["iteration"] == snap[0] and st_in["batch_size"] == snap[1]
    assert torch.equal(st_in["cache"][0], snap[2])
    assert torch.equal(st_in["cache"][1], snap[3])


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """张量/current_timestep/block_skip_range/区间/weight/distilled 校验."""
    m = _mat(1.0, 2)
    assert "attention_output 必须是 torch 张量" in _expect_err(
        solve, 1.0, None, 500)
    assert "至少是一维张量" in _expect_err(
        solve, torch.tensor(1.0), None, 500)
    assert "current_timestep 必须是整数" in _expect_err(
        solve, m, None, 500.0)
    assert "current_timestep 必须是整数" in _expect_err(
        solve, m, None, True)
    assert "block_skip_range 必须 >= 1" in _expect_err(
        solve, m, None, 500, block_skip_range=0)
    assert "block_skip_range 必须 >= 1" in _expect_err(
        solve, m, None, 500, block_skip_range=True)
    assert "timestep_skip_range 必须是 (lo, hi) 二元整数组" in _expect_err(
        solve, m, None, 500, timestep_skip_range=(100,))
    assert "timestep_skip_range 必须是 (lo, hi) 二元整数组" in _expect_err(
        solve, m, None, 500, timestep_skip_range=(100, 800.0))
    assert "timestep_skip_range 必须是 (lo, hi) 二元整数组" in _expect_err(
        solve, m, None, 500, timestep_skip_range="-1,681")
    assert "weight 必须是有限数值" in _expect_err(
        solve, m, None, 500, weight=float("inf"))
    assert "weight 必须是有限数值" in _expect_err(
        solve, m, None, 500, weight=True)
    assert "is_guidance_distilled 必须是 bool" in _expect_err(
        solve, m, None, 500, is_guidance_distilled=1)


def check_state_validation(solve):
    """state 三键结构校验."""
    m = _mat(1.0, 2)
    assert "state 必须是 dict" in _expect_err(solve, m, [], 500)
    assert "未知键" in _expect_err(solve, m, {"foo": 1}, 500)
    assert "缺少键" in _expect_err(solve, m, {"iteration": 0}, 500)
    assert "state['iteration'] 必须是非负整数" in _expect_err(
        solve, m, _fresh(iteration=-1), 500)
    assert "state['iteration'] 必须是非负整数" in _expect_err(
        solve, m, _fresh(iteration=True), 500)
    assert "state['batch_size'] 必须是正整数或 None" in _expect_err(
        solve, m, _fresh(batch_size=0), 500)
    assert "state['batch_size'] 必须是正整数或 None" in _expect_err(
        solve, m, _fresh(batch_size=True), 500)
    assert "state['cache'] 必须是 [t_2, t_1] 两个张量的 list 或 None" in _expect_err(
        solve, m, _fresh(cache=[_mat(1.0, 2)]), 500)
    assert "state['cache'] 必须是 [t_2, t_1] 两个张量的 list 或 None" in _expect_err(
        solve, m, _fresh(cache="x"), 500)
    assert "state['cache'] 必须是 [t_2, t_1] 两个张量的 list 或 None" in _expect_err(
        solve, m, _fresh(cache=[_mat(1.0, 2), 0.5]), 500)


# ---------------------------------------------------------------- 元信息
check_full_dance.example = True
check_slice_path.example = True
