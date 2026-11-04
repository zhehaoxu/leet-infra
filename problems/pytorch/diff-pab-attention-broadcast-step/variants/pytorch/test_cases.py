"""Pyramid Attention Broadcast 广播步 -- 测试用例."""

import torch


def _t(v):
    return torch.tensor([[float(v), v * 0.5 - 0.3]], dtype=torch.float32)


def _out(step):
    return _t(0.7 * step - 0.05 * step * step)


def _fresh(**over):
    st = {"iteration": 0, "cache": None}
    st.update(over)
    return st


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


def _kw(**over):
    kw = dict(spatial_attention_block_skip_range=2,
              spatial_attention_timestep_skip_range=(100, 800))
    kw.update(over)
    return kw


# ---------------------------------------------------------------- 分发
def check_dispatch_spatial(solve):
    """spatial 识别: 层名命中 spatial_ids 且非 cross -> block_type spatial."""
    r = solve("transformer_blocks.0.attn1", False, _t(1.0), None, 500, **_kw())
    assert r["block_type"] == "spatial" and r["computed"] is True  # 首步 cache None
    assert r["state"]["iteration"] == 1
    # cross 层不认领 spatial, 即使层名命中
    r2 = solve("transformer_blocks.0.attn2", True, _t(1.0), None, 500, **_kw())
    assert r2["block_type"] is None and r2["computed"] is True  # cross_range 未设
    assert r2["state"] == {"iteration": 0, "cache": None}  # state 原样透传


def check_dispatch_priority(solve):
    """temporal 层名含 'blocks' -> 官方默认 spatial_ids 先认领 (优先级即语义);
    spatial_range=None 时才轮到 temporal."""
    kw = _kw(temporal_attention_block_skip_range=3)
    r = solve("temporal_transformer_blocks.0.attn1", False, _t(1.0), None, 500, **kw)
    assert r["block_type"] == "spatial"  # spatial_range=2 已设, 先命中
    kw2 = _kw(spatial_attention_block_skip_range=None,
              temporal_attention_block_skip_range=3)
    r2 = solve("temporal_transformer_blocks.0.attn1", False, _t(1.0), None, 500, **kw2)
    assert r2["block_type"] == "temporal"
    # temporal 层是 cross 时不认领 temporal, 认领 cross (cross_ids 命中 'blocks')
    kw3 = _kw(temporal_attention_block_skip_range=3, cross_attention_block_skip_range=4)
    r3 = solve("temporal_transformer_blocks.0.attn2", True, _t(1.0), None, 500, **kw3)
    assert r3["block_type"] == "cross"


def check_dispatch_cross(solve):
    """cross 层: 需要 is_cross=True + cross_range 设定 + 层名命中."""
    kw = _kw(cross_attention_block_skip_range=4)
    r = solve("transformer_blocks.0.attn2", True, _t(1.0), None, 500, **kw)
    assert r["block_type"] == "cross"
    # is_cross=False 的同名层走 spatial
    r2 = solve("transformer_blocks.0.attn2", False, _t(1.0), None, 500, **kw)
    assert r2["block_type"] == "spatial"
    # cross_ids 未命中 (name 无 blocks/layers) -> 不匹配
    r3 = solve("proj_out", True, _t(1.0), None, 500, **kw)
    assert r3["block_type"] is None


def check_dispatch_unmatched(solve):
    """未匹配层: 恒 compute, state 原样透传 (iteration/cache 不动)."""
    st = _fresh(iteration=7, cache=_t(9.9))
    r = solve("norm_out", False, _t(1.0), st, 500, **_kw())
    assert r["computed"] is True and r["block_type"] is None
    assert r["output"] is r["output"]  # hidden_states 直接返回
    assert r["state"]["iteration"] == 7
    assert torch.equal(r["state"]["cache"], _t(9.9))


def check_default_spatial_range(solve):
    """三个 block_skip_range 全 None: 官方回退 spatial=2."""
    kw = dict(spatial_attention_block_skip_range=None)
    name = "transformer_blocks.0.attn1"
    r0 = solve(name, False, _t(1.0), None, 500, **kw)
    assert r0["block_type"] == "spatial" and r0["computed"] is True
    r1 = solve(name, False, _t(2.0), r0["state"], 500, **kw)
    assert r1["computed"] is False  # spatial=2 生效: 第 2 次调用 broadcast


# ---------------------------------------------------------------- 门控
def check_broadcast_sequence(solve):
    """block_skip_range=2, 区间内: compute/broadcast 交替 [T,F,T,F,T]."""
    name = "transformer_blocks.0.attn1"
    st = None
    seq = []
    for i in range(5):
        r = solve(name, False, _out(i), st, 500, **_kw())
        seq.append(r["computed"])
        st = r["state"]
    assert seq == [True, False, True, False, True], seq
    # broadcast 步 output = 上次 compute 的输出 (cache), 不是本步 hidden_states
    st = None
    outs = []
    for i in range(4):
        r = solve(name, False, _out(i), st, 500, **_kw())
        outs.append(r["output"])
        st = r["state"]
    assert torch.equal(outs[1], _out(0))  # 步 1 broadcast -> 步 0 的输出
    assert torch.equal(outs[2], _out(2))  # 步 2 compute (2%2==0)
    assert torch.equal(outs[3], _out(2))  # 步 3 broadcast -> 步 2 的输出


def check_broadcast_sequence_range3(solve):
    """block_skip_range=3: [T,F,F,T] (iteration % 3 == 0 时 compute)."""
    name = "transformer_blocks.0.attn1"
    st = None
    seq = []
    for i in range(5):
        r = solve(name, False, _out(i), st, 500,
                  spatial_attention_block_skip_range=3,
                  spatial_attention_timestep_skip_range=(100, 800))
        seq.append(r["computed"])
        st = r["state"]
    assert seq == [True, False, False, True, False], seq


def check_open_interval(solve):
    """时间步开区间 (lo, hi): t==lo 或 t==hi 时不在区间内, 恒 compute."""
    name = "transformer_blocks.0.attn1"
    for t in (100, 800):
        st = None
        for i in range(3):
            r = solve(name, False, _out(i), st, t, **_kw())
            assert r["computed"] is True, (t, i)  # 区间外禁 broadcast
            st = r["state"]
    # 区间内 (100, 800) 的中间值才允许 broadcast
    st = None
    r0 = solve(name, False, _out(0), st, 799, **_kw())
    r1 = solve(name, False, _out(1), r0["state"], 799, **_kw())
    assert r1["computed"] is False


def check_cache_none_forces_compute(solve):
    """cache 为 None 恒 compute (即使 iteration>0 且区间内)."""
    name = "transformer_blocks.0.attn1"
    st = _fresh(iteration=5, cache=None)
    r = solve(name, False, _t(1.0), st, 500, **_kw())
    assert r["computed"] is True and r["state"]["cache"] is not None


def check_iteration_zero_forces_compute(solve):
    """iteration=0 (带非 None cache 的手工 state): 0 % n == 0 强制 compute."""
    name = "transformer_blocks.0.attn1"
    st = _fresh(iteration=0, cache=_t(9.9))
    r = solve(name, False, _t(1.0), st, 500, **_kw())
    assert r["computed"] is True
    assert torch.equal(r["output"], _t(1.0))  # compute 步输出真实结果
    assert r["state"]["iteration"] == 1


def check_outside_range_no_broadcast(solve):
    """cache 已建立但时间步出区间 -> not within 强制 compute (缓存保留旧值? 官方更新)."""
    name = "transformer_blocks.0.attn1"
    r0 = solve(name, False, _out(0), None, 500, **_kw())
    st = r0["state"]
    r1 = solve(name, False, _out(1), st, 900, **_kw())  # 区间外
    assert r1["computed"] is True
    assert torch.equal(r1["state"]["cache"], _out(1))  # cache 更新为本次输出


def check_temporal_and_cross_ranges(solve):
    """temporal/cross 各用各的 block/timestep 区间."""
    kw = dict(spatial_attention_block_skip_range=None,
              temporal_attention_block_skip_range=2,
              temporal_attention_timestep_skip_range=(0, 50),
              cross_attention_block_skip_range=3,
              cross_attention_timestep_skip_range=(50, 100))
    st = None
    seq = []
    for i in range(4):
        r = solve("temporal_transformer_blocks.0.attn1", False, _out(i), st, 25, **kw)
        seq.append(r["computed"])
        st = r["state"]
    assert seq == [True, False, True, False]
    # cross 层: 区间 (50, 100) 内, range=3
    st = None
    seq2 = []
    for i in range(4):
        r = solve("transformer_blocks.0.attn2", True, _out(i), st, 75, **kw)
        seq2.append(r["computed"])
        st = r["state"]
    assert seq2 == [True, False, False, True]


def check_pure_function(solve):
    """纯函数: 入参 state 不被修改, 同入参两次调用一致."""
    name = "transformer_blocks.0.attn1"
    r0 = solve(name, False, _out(0), None, 500, **_kw())
    st_in = r0["state"]
    snap = (st_in["iteration"], st_in["cache"].clone())
    a = solve(name, False, _out(1), st_in, 500, **_kw())
    b = solve(name, False, _out(1), st_in, 500, **_kw())
    assert torch.equal(a["output"], b["output"]) and a["computed"] == b["computed"]
    assert st_in["iteration"] == snap[0]
    assert torch.equal(st_in["cache"], snap[1])


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """name/is_cross/张量/current_timestep/skip_range/时间步区间/identifiers 校验."""
    kw = _kw()
    assert "name 必须是字符串" in _expect_err(
        solve, 1.0, False, _t(1.0), None, 500, **kw)
    assert "is_cross_attention 必须是 bool" in _expect_err(
        solve, "blocks", 1, _t(1.0), None, 500, **kw)
    assert "hidden_states 必须是 torch 张量" in _expect_err(
        solve, "blocks", False, 1.0, None, 500, **kw)
    assert "current_timestep 必须是整数" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500.0, **kw)
    assert "current_timestep 必须是整数" in _expect_err(
        solve, "blocks", False, _t(1.0), None, True, **kw)
    assert "spatial_attention_block_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        spatial_attention_block_skip_range=0)
    assert "spatial_attention_block_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        spatial_attention_block_skip_range=True)
    assert "temporal_attention_block_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        temporal_attention_block_skip_range=-1)
    assert "cross_attention_block_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        cross_attention_block_skip_range=0)
    assert "spatial_attention_timestep_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        spatial_attention_timestep_skip_range=(100,))
    assert "spatial_attention_timestep_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        spatial_attention_timestep_skip_range=(100, 800.0))
    assert "temporal_attention_timestep_skip_range" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        temporal_attention_timestep_skip_range="100-800")
    assert "spatial_attention_block_identifiers" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        spatial_attention_block_identifiers="blocks")
    assert "cross_attention_block_identifiers" in _expect_err(
        solve, "blocks", False, _t(1.0), None, 500,
        cross_attention_block_identifiers=("blocks", 1))


def check_state_validation(solve):
    """state 两键结构校验."""
    kw = _kw()
    assert "state 必须是 dict" in _expect_err(
        solve, "blocks", False, _t(1.0), [], 500, **kw)
    assert "未知键" in _expect_err(
        solve, "blocks", False, _t(1.0), {"foo": 1}, 500, **kw)
    assert "缺少键" in _expect_err(
        solve, "blocks", False, _t(1.0), {"iteration": 0}, 500, **kw)
    assert "state['iteration'] 必须是非负整数" in _expect_err(
        solve, "blocks", False, _t(1.0), {"iteration": -1, "cache": None}, 500, **kw)
    assert "state['iteration'] 必须是非负整数" in _expect_err(
        solve, "blocks", False, _t(1.0), {"iteration": True, "cache": None}, 500, **kw)
    assert "state['cache'] 必须是 torch 张量或 None" in _expect_err(
        solve, "blocks", False, _t(1.0), {"iteration": 0, "cache": 0.5}, 500, **kw)


# ---------------------------------------------------------------- 元信息
check_broadcast_sequence.example = True
check_dispatch_priority.example = True
