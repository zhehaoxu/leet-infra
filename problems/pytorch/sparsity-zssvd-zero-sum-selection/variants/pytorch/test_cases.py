"""Zero-Sum SVD 全局奇异分量选择步 -- 测试用例."""

import math

import numpy as np
import torch


def _mat(v, rows, cols):
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * (0.271 + v * 0.017)) % 1.0) * 2.0 - 1.0
    return x.reshape(rows, cols).contiguous()


def _cholesky_factor(v, dim):
    raw = _mat(v, dim, dim)
    H = raw @ raw.t() + torch.eye(dim) * 0.5
    return torch.linalg.cholesky(H).contiguous()


SHAPES = {"a": (12, 8), "b": (8, 16), "c": (10, 10)}


def _fixture(v=0.0, ratio=0.2, keep_ratio=0.3, beta=None, order="first_order",
             with_L=True):
    modules = {}
    for i, (name, (m, n)) in enumerate(sorted(SHAPES.items())):
        modules[name] = {
            "weight": _mat(v + i * 3.0, m, n),
            "grad": _mat(v + i * 5.0 + 1.0, m, n) * 0.1,
            "L": _cholesky_factor(v + i * 7.0 + 2.0, n) if with_L else None,
        }
    return dict(modules=modules, global_prune_ratio=ratio,
                keep_rank_ratio=keep_ratio, beta_mixing=beta,
                delta_order=order)


def _np_delta(sigma, grad, beta, order):
    if beta is None:
        if order == "second_order":
            return -sigma * grad + 0.5 * sigma ** 2 * grad ** 2
        return -sigma * grad
    if grad == 0.0 and sigma == 0.0:
        return 0.0
    sig_abs = max(abs(sigma), 1e-12)
    grad_abs = max(abs(grad), 1e-12)
    sign = 1.0 if grad >= 0.0 else -1.0
    return -sign * (sig_abs ** beta) * (grad_abs ** (1.0 - beta))


def _oracle(modules, global_prune_ratio=0.2, keep_rank_ratio=0.3,
            beta_mixing=None, delta_order="first_order"):
    states = {}
    initial = 0
    for name in sorted(modules):
        mod = modules[name]
        w = mod["weight"].numpy().astype(np.float64)
        g = mod["grad"].numpy().astype(np.float64)
        m, n = w.shape
        if mod["L"] is None:
            A, H = w, g
        else:
            L = mod["L"].numpy().astype(np.float64)
            A = w @ L
            H = np.linalg.solve(L, g.T).T  # G @ L^{-T}
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        order = np.argsort(-S, kind="stable")
        S, U, VT = S[order], U[:, order], VT[order, :]
        V = VT.T
        grad_sigma = np.diagonal(U.T @ H @ V)
        r_star = max(1, (m * n - 1) // (m + n))
        min_rank = max(1, math.ceil(keep_rank_ratio * r_star))
        states[name] = {"shape": (m, n), "sigma": S, "grad": grad_sigma,
                        "r_star": r_star,
                        "max_svs": max(0, min(m, n) - min_rank),
                        "consumed": 0, "trunc": 0.0}
        initial += m * n

    budget = initial - int((1.0 - global_prune_ratio) * initial)
    running = 0.0
    dense_removed = 0
    selections = []

    def eff_delta(shape, r_now, r_star):
        m, n = shape
        if r_now > r_star:
            return 0
        if r_now == r_star:
            return m * n - r_star * (m + n)
        return m + n

    while dense_removed < budget:
        prefer_pos = running <= 0.0
        cands = []
        # 官方 pop_prefer_then_any: 先偏好符号堆, 空了落到另一符号堆
        for prefer in (prefer_pos, not prefer_pos):
            for name in sorted(states):
                st = states[name]
                remaining = st["max_svs"] - st["consumed"]
                if remaining <= 0:
                    continue
                idx = st["sigma"].size - 1 - st["consumed"]
                d = _np_delta(float(st["sigma"][idx]), float(st["grad"][idx]),
                              beta_mixing, delta_order)
                if prefer and d < 0.0:
                    continue
                if not prefer and d >= 0.0:
                    continue
                cands.append((abs(d), name, d))
            if cands:
                break
        if not cands:
            break
        cands.sort(key=lambda t: (t[0], t[1]))
        _, name, d = cands[0]
        st = states[name]
        st["consumed"] += 1
        st["trunc"] += float(st["sigma"][st["sigma"].size - st["consumed"]] ** 2)
        running += d
        r_now = st["sigma"].size - st["consumed"]
        dense_removed += eff_delta(st["shape"], r_now, st["r_star"])
        selections.append((name, d))

    ranks = {n: states[n]["sigma"].size - states[n]["consumed"]
             for n in states}
    masks = {n: [True] * ranks[n] + [False] * states[n]["consumed"]
             for n in states}
    return {"ranks": ranks, "keep_masks": masks,
            "truncated_sigma_sq": {n: states[n]["trunc"] for n in states},
            "running_sum": running, "dense_removed": dense_removed,
            "stage_budget": budget, "selections": selections}


def _assert_close(a, e, atol=1e-6):
    if isinstance(e, (int, float)):
        assert abs(a - e) <= atol, (a, e)
        return
    assert a == e or abs(a - e) <= atol, (a, e)


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 主体
def check_zero_sum_oracle(solve):
    """默认参数: 秩/mask/截断谱/running_sum/记账全对拍; 且秩异质."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    assert r["ranks"] == exp["ranks"], (r["ranks"], exp["ranks"])
    for name in exp["keep_masks"]:
        assert r["keep_masks"][name] == exp["keep_masks"][name], name
        _assert_close(r["truncated_sigma_sq"][name],
                      exp["truncated_sigma_sq"][name])
    _assert_close(r["running_sum"], exp["running_sum"], 1e-9)
    assert r["dense_removed"] == exp["dense_removed"]
    assert r["stage_budget"] == exp["stage_budget"]
    assert [(n, round(d, 9)) for n, d in r["selections"]] == \
        [(n, round(d, 9)) for n, d in exp["selections"]]
    # 异质秩: 至少两个模块秩不同
    assert len(set(r["ranks"].values())) >= 2, r["ranks"]


def check_delta_order_variants(solve):
    """second_order 与 beta_mixing 各自对拍; 且与 first_order 结果不同."""
    kw2 = _fixture(order="second_order")
    r2 = solve(**kw2)
    _assert_matches(r2, _oracle(**kw2))
    kwb = _fixture(beta=0.7)  # 0.5 时 beta/(1-beta) 指数互换恰好等价
    rb = solve(**kwb)
    _assert_matches(rb, _oracle(**kwb))
    rb5 = solve(**_fixture(beta=0.5))
    _assert_matches(rb5, _oracle(**_fixture(beta=0.5)))
    r1 = solve(**_fixture())
    assert r1["ranks"] != rb["ranks"] or r1["selections"] != rb["selections"]


def _assert_matches(r, exp):
    assert r["ranks"] == exp["ranks"]
    for name in exp["keep_masks"]:
        assert r["keep_masks"][name] == exp["keep_masks"][name], name
    _assert_close(r["running_sum"], exp["running_sum"], 1e-9)
    assert r["dense_removed"] == exp["dense_removed"]


def check_zero_sum_sign_balance(solve):
    """零和性: running_sum 被压在 0 附近 (|running_sum| < max|单步 δ| 量级)."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_matches(r, exp)
    # 交替选择: selections 的 delta 符号应频繁翻转 (zero-sum 特征)
    deltas = [d for _, d in r["selections"]]
    signs = [d >= 0 for d in deltas]
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    assert flips >= len(signs) // 3, (flips, len(signs))


def check_break_even_accounting(solve):
    """参数记账: r* 分段 (r>r* 记 0, ==r* 记 mn−r*(m+n), <r* 记 m+n); 预算公式."""
    kw = _fixture(ratio=0.0)
    r0 = solve(**kw)
    assert r0["dense_removed"] == 0 and r0["stage_budget"] == 0
    assert all(c == 0 for c in
               [sum(1 for b in r0["keep_masks"][n] if not b)
                for n in r0["keep_masks"]])
    kw = _fixture(ratio=0.5)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_matches(r, exp)
    total = sum(m * n for m, n in SHAPES.values())
    assert r["stage_budget"] == total - int(0.5 * total)
    # 大 budget: 候选耗尽路径 -> 每模块秩 == min_rank (cap 全用满)
    kw9 = _fixture(ratio=0.9)
    r9 = solve(**kw9)
    _assert_matches(r9, _oracle(**kw9))
    for name, (m, n) in SHAPES.items():
        r_star = max(1, (m * n - 1) // (m + n))
        min_rank = max(1, math.ceil(0.3 * r_star))
        assert r9["ranks"][name] == min_rank, (name, r9["ranks"][name], min_rank)
    # keep_ratio=0.5: ceil(0.5*r*) 与 floor 及 r*±1 分化 (break_even 变异的抓点)
    kw5 = _fixture(ratio=0.9, keep_ratio=0.5)
    r5 = solve(**kw5)
    _assert_matches(r5, _oracle(**kw5))


def check_keep_rank_floor(solve):
    """秩下限: 每模块秩 >= ceil(keep_rank_ratio * r*) 的 cap 生效 (mask 全真段)."""
    kw = _fixture(keep_ratio=0.9)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_matches(r, exp)
    for name, (m, n) in SHAPES.items():
        r_star = max(1, (m * n - 1) // (m + n))
        floor = max(1, math.ceil(0.9 * r_star))
        assert r["ranks"][name] >= min(m, n) - (min(m, n) - floor), \
            (name, r["ranks"][name], floor)
    # keep_ratio=1.0 -> 下限为 r* (盈亏平衡秩), 各模块秩 >= r*
    r1 = solve(**_fixture(keep_ratio=1.0))
    exp1 = _oracle(**_fixture(keep_ratio=1.0))
    _assert_matches(r1, exp1)
    for name, (m, n) in SHAPES.items():
        r_star = max(1, (m * n - 1) // (m + n))
        assert r1["ranks"][name] >= r_star


def check_no_whitening_path(solve):
    """L=None: 恒等白化路径对拍, 且与带 L 结果不同."""
    kw = _fixture(with_L=False)
    r = solve(**kw)
    _assert_matches(r, _oracle(**kw))
    r_L = solve(**_fixture(with_L=True))
    assert r["selections"] != r_L["selections"]


def check_purity(solve):
    """纯函数: 所有入参张量不被修改."""
    kw = _fixture()
    snap = {}
    for name, mod in kw["modules"].items():
        snap[name] = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                      for k, v in mod.items()}
    solve(**kw)
    for name, mod in kw["modules"].items():
        for k, v in mod.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, snap[name][k]), (name, k)


def check_args_validation(solve):
    """modules 结构/形状/ratio/keep_ratio/beta/delta_order 校验."""
    kw = _fixture()
    assert "modules 必须是非空 dict" in _expect_err(solve, {})
    bad = dict(kw, modules={"a": {"weight": kw["modules"]["a"]["weight"]}})
    assert "必须含 weight/grad/L" in _expect_err(solve, **bad)
    bad = dict(kw, modules={1: kw["modules"]["a"]})
    assert "键必须是非空字符串" in _expect_err(solve, **bad)
    bad_mods = {n: dict(m) for n, m in kw["modules"].items()}
    bad_mods["a"] = dict(bad_mods["a"], grad=_mat(0.0, 3, 3))
    bad = dict(kw, modules=bad_mods)
    assert "grad 必须是 (12, 8) 形状" in _expect_err(solve, **bad)
    bad_mods = {n: dict(m) for n, m in kw["modules"].items()}
    bad_mods["a"] = dict(bad_mods["a"], L=torch.eye(5))
    bad = dict(kw, modules=bad_mods)
    assert "L 必须是 (8, 8) 形状" in _expect_err(solve, **bad)
    for rv in (-0.1, 1.0, True, float("inf"), "0.2"):
        bad = dict(kw, global_prune_ratio=rv)
        assert "global_prune_ratio 必须是 [0, 1) 内的有限数值" in _expect_err(
            solve, **bad)
    bad = dict(kw, global_prune_ratio=0)
    solve(**bad)  # int 0 合法
    for kv in (-0.1, 1.5, True, float("nan")):
        bad = dict(kw, keep_rank_ratio=kv)
        assert "keep_rank_ratio 必须是 [0, 1] 内的有限数值" in _expect_err(
            solve, **bad)
    bad = dict(kw, beta_mixing=1.5)
    assert "beta_mixing 必须是 [0, 1] 内的有限数值或 None" in _expect_err(
        solve, **bad)
    bad = dict(kw, delta_order="third")
    assert "delta_order 必须是" in _expect_err(solve, **bad)


# ---------------------------------------------------------------- 元信息
check_zero_sum_oracle.example = True
check_delta_order_variants.example = True
check_zero_sum_sign_balance.example = True
check_break_even_accounting.example = True
check_keep_rank_floor.example = True
check_no_whitening_path.example = True
check_purity.example = True
check_args_validation.example = True
