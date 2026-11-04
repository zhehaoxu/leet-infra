"""First Block Cache 残差门控步 -- 测试用例."""

import numpy as np
import torch


def _t(v, b=1):
    return torch.tensor([[float(v), v * 0.5 - 0.3]] * b, dtype=torch.float64)


def _r(a, b):
    """显式两列残差张量 (不依赖 _t 的线性模式)."""
    return torch.tensor([[float(a), float(b)]], dtype=torch.float64)


def _absmean_np(a):
    return float(np.abs(np.asarray(a, dtype=np.float64)).mean())


def _close(actual, expected, tol=1e-9):
    a = np.asarray(torch.detach(actual).to(torch.float64).numpy(), dtype=np.float64).reshape(-1)
    e = np.broadcast_to(np.asarray(expected, dtype=np.float64).reshape(-1), a.shape)
    return bool(np.all(np.abs(a - e) <= tol))


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 基本路径
def check_first_step_compute(solve):
    """首步 (state=None): 强制 compute, final == tail_output, state 三元组填充."""
    hi, ho, to = _t(1.0), _t(1.2), _t(1.5)
    r = solve(hi, ho, to, None, threshold=0.05)
    assert r["computed"] is True and r["diff"] is None
    assert torch.equal(r["final_output"], to)
    st = r["state"]
    assert _close(st["head_block_output"], ho.to(torch.float64).numpy(), tol=0.0)
    assert _close(st["head_block_residual"], (ho - hi).to(torch.float64).numpy())
    assert _close(st["tail_block_residuals"], (to - ho).to(torch.float64).numpy())


def check_compute_path(solve):
    """残差变化大 (diff > threshold): compute, 最终输出 = tail_output."""
    hi = _t(1.0)
    ho = hi + _r(0.2, 0.15)   # 当前残差 (0.2, 0.15)
    to = _t(1.5)
    prev_res = _r(0.05, 0.04)  # prev 残差与当前差异大
    st = {"head_block_output": hi + _r(0.1, 0.08), "head_block_residual": prev_res,
          "tail_block_residuals": _r(0.3, 0.25)}
    r = solve(hi, ho, to, st, threshold=0.05)
    assert r["computed"] is True
    assert torch.equal(r["final_output"], to)
    # diff 手算
    num = _absmean_np((ho - hi).to(torch.float64).numpy() - prev_res.to(torch.float64).numpy())
    den = _absmean_np(prev_res.to(torch.float64).numpy())
    assert abs(r["diff"] - num / den) < 1e-12
    # state 更新为本次值
    assert _close(r["state"]["tail_block_residuals"],
                  (to - ho).to(torch.float64).numpy())


def check_cache_path(solve):
    """残差变化小 (diff <= threshold): cache, final = cached_tail_res + head_output."""
    hi = _t(1.0)
    ho = hi + _r(0.205, 0.105)  # 当前残差与 prev (0.2, 0.10) 差 (0.005, 0.005) -> diff = 0.005/0.15 = 0.0333
    to = _t(9.9)                # cache 路径不使用 tail_output
    prev_res = _r(0.2, 0.10)
    cached_tail = _r(0.3, 0.25)
    st = {"head_block_output": hi + _r(0.1, 0.08), "head_block_residual": prev_res,
          "tail_block_residuals": cached_tail}
    r = solve(hi, ho, to, st, threshold=0.05)
    assert r["computed"] is False, r["diff"]
    assert _close(r["final_output"], (cached_tail + ho).to(torch.float64).numpy(), tol=0.0)
    # state 全部保持旧值 (stale)
    assert _close(r["state"]["head_block_residual"], prev_res.to(torch.float64).numpy(), tol=0.0)
    assert _close(r["state"]["tail_block_residuals"], cached_tail.to(torch.float64).numpy(), tol=0.0)
    assert _close(r["state"]["head_block_output"], (hi + _r(0.1, 0.08)).to(torch.float64).numpy(), tol=0.0)


def check_threshold_boundary(solve):
    """diff == threshold -> cache (严格 >); 稍大 -> compute. 用 2 的幂精确构造."""
    hi = _t(1.0)
    prev_res = _r(0.5, 0.5)          # den = 0.5 (精确)
    st = {"head_block_output": hi + _r(0.25, 0.25), "head_block_residual": prev_res,
          "tail_block_residuals": _r(0.3, 0.25)}
    # 当前残差 (0.75, 0.75) -> 与 prev 差 (0.25, 0.25) -> num = 0.25, diff = 0.5 精确
    ho = hi + _r(0.75, 0.75)
    thr = 0.5
    r = solve(hi, ho, _t(1.5), st, threshold=thr)
    assert r["computed"] is False and r["diff"] == thr, (r["diff"], thr)
    # 残差差 (0.25 + 2^-40, ...) -> diff 略大于阈值
    ho2 = hi + _r(0.75 + 2.0 ** -40, 0.75 + 2.0 ** -40)
    r2 = solve(hi, ho2, _t(1.5), st, threshold=thr)
    assert r2["computed"] is True and r2["diff"] > thr


def check_decreasing_residual(solve):
    """残差比 prev 小 (负差值): abs 后 diff 为正 -> compute; 无 abs 的实现会误 cache."""
    hi = _t(1.0)
    prev_res = _r(0.2, 0.10)
    st = {"head_block_output": hi + _r(0.1, 0.08), "head_block_residual": prev_res,
          "tail_block_residuals": _r(0.3, 0.25)}
    den = _absmean_np(prev_res.to(torch.float64).numpy())
    # 残差 (0.15, 0.05): 差 (-0.05, -0.05) -> absmean = 0.05 -> diff = 0.05/0.15 = 1/3 > 0.05
    r = solve(hi, hi + _r(0.15, 0.05), _t(1.5), st, threshold=0.05)
    assert r["computed"] is True
    assert abs(r["diff"] - 0.05 / den) < 1e-12


def check_stale_comparison(solve):
    """连续 cache 步都与上次 compute 的残差比较 (stale)."""
    hi = _t(1.0)
    prev_res = _r(0.2, 0.10)
    st = {"head_block_output": hi + _r(0.1, 0.08), "head_block_residual": prev_res,
          "tail_block_residuals": _r(0.3, 0.25)}
    den = _absmean_np(prev_res.to(torch.float64).numpy())
    # 步 A: 残差差 (0.005, 0.002) -> num=0.0035 -> diff=0.0035/0.15 < 0.05 cache
    rA = solve(hi, hi + _r(0.205, 0.102), _t(1.5), st, threshold=0.05)
    assert rA["computed"] is False
    # 步 B: 残差 (0.23, 0.13) -> 与 prev 比较 diff = absmean(0.03,0.03)/0.15 = 0.2 > 0.05
    # (不是与步 A 的残差 0.205/0.102 比较 -> 那会得 0.025/0.15 < 0.05 cache)
    rB = solve(hi, hi + _r(0.23, 0.13), _t(1.5), rA["state"], threshold=0.05)
    assert rB["computed"] is True
    num = _absmean_np(_r(0.03, 0.03).to(torch.float64).numpy())
    assert abs(rB["diff"] - num / den) < 1e-12


def check_state_update_semantics(solve):
    """cache 步 state 不更新; compute 步三项全部更新 (对照 cache 输入)."""
    st_in = {"head_block_output": hi0 + _r(0.1, 0.08), "head_block_residual": _r(0.2, 0.10),
             "tail_block_residuals": _r(0.3, 0.25)} if (hi0 := _t(1.0)) is not None else None
    # cache 步: 残差差 (0.01, 0.01) -> diff = 0.01/0.15 = 0.0667 > 0.05? 不, 0.01/0.15=0.0667
    # 需要 diff <= 0.05: 用残差差 (0.005, 0.005) -> num 0.005 / 0.15 = 0.0333
    r = solve(_t(1.0), _t(1.0) + _r(0.205, 0.105), _t(9.9), st_in, threshold=0.05)
    st_c = r["state"]
    for k in st_in:
        assert torch.equal(st_c[k], st_in[k]), k
    # compute 步: 残差 (0.6, 0.55) 与 prev 差 (0.4, 0.45) -> num 大 -> compute
    hi = _t(1.0)
    ho2 = hi + _r(0.6, 0.55)
    r2 = solve(hi, ho2, _t(2.0), st_in, threshold=0.05)
    st_n = r2["state"]
    assert _close(st_n["head_block_output"], ho2.to(torch.float64).numpy(), tol=0.0)
    # 期望残差 = ho2 - hi (浮点上 1.0+0.6-1.0 != 0.6, 不能直接写 _r(0.6, 0.55))
    assert _close(st_n["head_block_residual"], (ho2 - hi).to(torch.float64).numpy(), tol=0.0)
    assert _close(st_n["tail_block_residuals"], (_t(2.0) - ho2).to(torch.float64).numpy(), tol=0.0)


def check_pure_function(solve):
    """纯函数: 同入参两次调用一致, 入参 state 不被修改."""
    st_in = {"head_block_output": _t(1.0) + _r(0.1, 0.08), "head_block_residual": _r(0.2, 0.10),
             "tail_block_residuals": _r(0.3, 0.25)}
    snap = {k: v.clone() for k, v in st_in.items()}
    a = solve(_t(1.0), _t(1.0) + _r(0.205, 0.105), _t(1.5), st_in, threshold=0.05)
    b = solve(_t(1.0), _t(1.0) + _r(0.205, 0.105), _t(1.5), st_in, threshold=0.05)
    assert torch.equal(a["final_output"], b["final_output"])
    assert a["computed"] == b["computed"] and a["diff"] == b["diff"]
    for k, v in snap.items():
        assert torch.equal(st_in[k], v)


# ---------------------------------------------------------------- 边界
def check_zero_prev_residual(solve):
    """prev 残差全零: 分母 0 规范化 (分子 0 -> cache / 非零 -> compute)."""
    hi = _t(1.0)
    zero_res = torch.zeros((1, 2), dtype=torch.float64)
    st = {"head_block_output": _t(1.0), "head_block_residual": zero_res,
          "tail_block_residuals": _t(0.3)}
    # 当前残差非零 -> inf -> compute
    r = solve(hi, _t(1.2), _t(1.5), st, threshold=0.05)
    assert r["computed"] is True and r["diff"] == float("inf")
    # 当前残差也全零 -> 0.0 -> cache
    r2 = solve(hi, hi.clone(), _t(1.5), st, threshold=0.05)
    assert r2["computed"] is False and r2["diff"] == 0.0
    assert _close(r2["final_output"], (_t(0.3) + hi).to(torch.float64).numpy(), tol=0.0)


def check_tail_missing_cache_reject(solve):
    """cache 决策但 tail_block_residuals 为 None (官方 reset 后的不一致组合) -> raise."""
    st = {"head_block_output": _t(1.0) + _r(0.1, 0.08), "head_block_residual": _r(0.2, 0.10),
          "tail_block_residuals": None}
    # 残差差 (0.005, 0.005) -> diff=0.0333 <= 0.05 -> cache -> tail 缺失 raise
    err = _expect_err(solve, _t(1.0), _t(1.0) + _r(0.205, 0.105), _t(1.5), st, threshold=0.05)
    assert "cache 路径需要" in err
    # 同 state 但残差变化大 -> compute 合法 (官方 reset 后首步路径)
    r = solve(_t(1.0), _t(1.0) + _r(0.6, 0.55), _t(1.5), st, threshold=0.05)
    assert r["computed"] is True
    assert _close(r["state"]["tail_block_residuals"],
                  (_t(1.5) - (_t(1.0) + _r(0.6, 0.55))).to(torch.float64).numpy())


# ---------------------------------------------------------------- 校验
def check_state_validation(solve):
    """state 结构校验."""
    kw = dict(threshold=0.05)
    assert "state 必须是 dict" in _expect_err(solve, _t(1.0), _t(1.2), _t(1.5), [], **kw)
    assert "未知键" in _expect_err(solve, _t(1.0), _t(1.2), _t(1.5),
                                  {"foo": 1}, **kw)
    assert "必须是 torch 张量或 None" in _expect_err(
        solve, _t(1.0), _t(1.2), _t(1.5), {"head_block_output": 1.0}, **kw)
    assert "必须是 torch 张量或 None" in _expect_err(
        solve, _t(1.0), _t(1.2), _t(1.5),
        {"head_block_output": _t(1.1), "head_block_residual": "x"}, **kw)
    assert "必须是 torch 张量或 None" in _expect_err(
        solve, _t(1.0), _t(1.2), _t(1.5),
        {"head_block_output": _t(1.1), "head_block_residual": _t(0.2),
         "tail_block_residuals": True}, **kw)


def check_args_validation(solve):
    """张量类型/形状与 threshold 校验."""
    good_st = {"head_block_output": _t(1.0) + _r(0.1, 0.08), "head_block_residual": _r(0.2, 0.10),
               "tail_block_residuals": _r(0.3, 0.25)}
    assert "head_input 必须是 torch 张量" in _expect_err(
        solve, 1.0, _t(1.2), _t(1.5), None, **{"threshold": 0.05})
    assert "head_output 必须是 torch 张量" in _expect_err(
        solve, _t(1.0), np.zeros((1, 2)), _t(1.5), None, threshold=0.05)
    assert "tail_output 必须是 torch 张量" in _expect_err(
        solve, _t(1.0), _t(1.2), None, None, threshold=0.05)
    assert "形状必须一致" in _expect_err(
        solve, _t(1.0), torch.zeros((2, 2), dtype=torch.float64), _t(1.5), None, threshold=0.05)
    assert "形状必须一致" in _expect_err(
        solve, _t(1.0), _t(1.2), torch.zeros((2, 2), dtype=torch.float64), None, threshold=0.05)
    assert "threshold 必须是非负数值" in _expect_err(
        solve, _t(1.0), _t(1.2), _t(1.5), None, threshold=-0.1)
    assert "threshold 必须是非负数值" in _expect_err(
        solve, _t(1.0), _t(1.2), _t(1.5), None, threshold=True)
    assert "threshold 必须是非负数值" in _expect_err(
        solve, _t(1.0), _t(1.2), _t(1.5), None, threshold="0.05")
    # threshold=0 合法 (diff>0 即 compute)
    r = solve(_t(1.0), _t(1.0) + _r(0.6, 0.55), _t(1.5), good_st, threshold=0)
    assert r["computed"] is True


# ---------------------------------------------------------------- 元信息
check_first_step_compute.example = True
check_cache_path.example = True
