"""DPMSolverMultistep 反演多步更新账——判题用例（order 832）。

独立 oracle 用纯 Python math 从规则重写，不 import solution。
手算锚点选 flow 约定下 σ = 1/9, 1/5, 1/3（λ = log8, log4, log2），
使 h、h_0、h_1 均为 −log2、r0 = r1 = 1，系数可逐字手算。
"""

import math

import torch


def check(solve):
    check_sigma_to_alpha(solve)
    check_convert(solve)
    check_first(solve)
    check_second(solve)
    check_third(solve)
    check_step(solve)
    print("all checks passed")


# ---------------------------------------------------------------- 独立 oracle

def ref_s2a(s, flow):
    if flow:
        return 1.0 - s, s
    a = 1.0 / math.sqrt(s * s + 1.0)
    return a, s * a


def ref_lam(s, flow):
    a, st = ref_s2a(s, flow)
    return math.log(a) - math.log(st), a, st


def ref_convert(mo, sample, sigma, alg, pred, flow):
    a, st = ref_s2a(sigma, flow)
    if alg == "dpmsolver++":
        if pred == "epsilon":
            return (sample - st * mo) / a, "x0_pred"
        if pred == "sample":
            return mo, "x0_pred"
        if pred == "v_prediction":
            return a * sample - st * mo, "x0_pred"
        if pred == "flow_prediction":
            return sample - sigma * mo, "x0_pred"
        raise ValueError
    if pred == "epsilon":
        return mo, "epsilon"
    if pred == "sample":
        return (sample - a * mo) / st, "epsilon"
    if pred == "v_prediction":
        return a * mo + st * sample, "epsilon"
    raise ValueError


def ref_first(sig, i, sample, m0, alg, flow):
    s0, st = sig[i], sig[i + 1]
    ls, a_s, ss = ref_lam(s0, flow)
    lt, a_t, stt = ref_lam(st, flow)
    h = lt - ls
    if alg == "dpmsolver++":
        return (stt / ss) * sample - (a_t * (math.exp(-h) - 1.0)) * m0
    return (a_t / a_s) * sample - (stt * (math.exp(h) - 1.0)) * m0


def ref_second(sig, i, sample, m1, m0, alg, stype, flow):
    s1, s0, st = sig[i - 1], sig[i], sig[i + 1]
    l1, a1, ss1 = ref_lam(s1, flow)
    l0, a0, ss0 = ref_lam(s0, flow)
    lt, at, stt = ref_lam(st, flow)
    h, h0 = lt - l0, l0 - l1
    r0 = h0 / h
    D0 = m0
    D1 = (1.0 / r0) * (m0 - m1)
    if alg == "dpmsolver++":
        c = stt / ss0
        b = at * (math.exp(-h) - 1.0)
        cd1 = -0.5 * b if stype == "midpoint" else at * ((math.exp(-h) - 1.0) / h + 1.0)
        return c * sample - b * D0 + cd1 * D1
    c = at / a0
    b = stt * (math.exp(h) - 1.0)
    cd1 = -0.5 * b if stype == "midpoint" else -(stt * ((math.exp(h) - 1.0) / h - 1.0))
    return c * sample - b * D0 + cd1 * D1


def ref_third(sig, i, sample, m2, m1, m0, alg, flow):
    s2, s1, s0, st = sig[i - 2], sig[i - 1], sig[i], sig[i + 1]
    l2, a2, ss2 = ref_lam(s2, flow)
    l1, a1, ss1 = ref_lam(s1, flow)
    l0, a0, ss0 = ref_lam(s0, flow)
    lt, at, stt = ref_lam(st, flow)
    h, h0, h1 = lt - l0, l0 - l1, l1 - l2
    r0, r1 = h0 / h, h1 / h
    d10 = (1.0 / r0) * (m0 - m1)
    d11 = (1.0 / r1) * (m1 - m2)
    D1 = d10 + (r0 / (r0 + r1)) * (d10 - d11)
    D2 = (1.0 / (r0 + r1)) * (d10 - d11)
    if alg == "dpmsolver++":
        return ((stt / ss0) * sample
                - (at * (math.exp(-h) - 1.0)) * m0
                + (at * ((math.exp(-h) - 1.0) / h + 1.0)) * D1
                - (at * ((math.exp(-h) - 1.0 + h) / (h * h) - 0.5)) * D2)
    return ((at / a0) * sample
            - (stt * (math.exp(h) - 1.0)) * m0
            - (stt * ((math.exp(h) - 1.0) / h - 1.0)) * D1
            - (stt * ((math.exp(h) - 1.0 - h) / (h * h) - 0.5)) * D2)


def close(a, b, tol=1e-9):
    if isinstance(a, torch.Tensor):
        a = a.detach().to(torch.float64).tolist()
    if isinstance(b, torch.Tensor):
        b = b.detach().to(torch.float64).tolist()
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        a = list(a) if isinstance(a, (list, tuple)) else [a]
        b = list(b) if isinstance(b, (list, tuple)) else [b]
        assert len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            close(x, y, tol)
        return
    assert math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol), (a, b)


def T(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float64)
    return torch.tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------- 各 check_*

def check_sigma_to_alpha(solve):
    """_sigma_to_alpha_sigma_t：VP 开方约定 / flow 线性约定 + 校验。"""
    r = solve(mode="sigma_to_alpha", sigma=1.0, use_flow_sigmas=False)
    close(r["alpha"], 1.0 / math.sqrt(2.0))
    close(r["sigma_t"], 1.0 / math.sqrt(2.0))
    r = solve(mode="sigma_to_alpha", sigma=2.0, use_flow_sigmas=False)
    close(r["alpha"], 1.0 / math.sqrt(5.0))
    close(r["sigma_t"], 2.0 / math.sqrt(5.0))
    r = solve(mode="sigma_to_alpha", sigma=0.25, use_flow_sigmas=True)
    close(r["alpha"], 0.75)
    close(r["sigma_t"], 0.25)
    # 手算：flow σ=1/3 → α=2/3；VP σ=3/4 → α=4/5, σ_t=3/5
    r = solve(mode="sigma_to_alpha", sigma=1.0 / 3.0, use_flow_sigmas=True)
    close(r["alpha"], 2.0 / 3.0)
    r = solve(mode="sigma_to_alpha", sigma=0.75, use_flow_sigmas=False)
    close(r["alpha"], 0.8)
    close(r["sigma_t"], 0.6)
    for bad in (0.0, -1.0, float("inf")):
        try:
            solve(mode="sigma_to_alpha", sigma=bad, use_flow_sigmas=False)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        solve(mode="sigma_to_alpha", sigma=1.0, use_flow_sigmas=True)
        raise AssertionError("expected ValueError for flow sigma >= 1")
    except ValueError:
        pass


def check_convert(solve):
    """convert_model_output：7 条 prediction × algorithm 组合 + 错误分支。"""
    sigma = 0.5                       # flow: α=0.5, σ_t=0.5
    cases = [
        ("dpmsolver++", "epsilon", 2.0, 0.5, 3.5, "x0_pred"),
        ("dpmsolver++", "sample", 2.0, 0.5, 0.5, "x0_pred"),
        ("dpmsolver++", "v_prediction", 2.0, 0.5, 0.75, "x0_pred"),
        ("dpmsolver++", "flow_prediction", 2.0, 0.5, 1.75, "x0_pred"),
        ("dpmsolver", "epsilon", 2.0, 0.5, 0.5, "epsilon"),
        ("dpmsolver", "sample", 2.0, 0.5, 3.5, "epsilon"),
        ("dpmsolver", "v_prediction", 2.0, 0.5, 1.25, "epsilon"),
    ]
    for alg, pred, sample, mo, expected, kind in cases:
        r = solve(mode="convert", model_output=T(mo), sample=T(sample),
                  sigma=sigma, algorithm_type=alg, prediction_type=pred,
                  use_flow_sigmas=True)
        assert r["kind"] == kind
        close(r["converted"], expected)
        # 与独立 oracle 对拍
        ref, _ = ref_convert(mo, sample, sigma, alg, pred, True)
        close(r["converted"], ref)
    # VP 约定下同样对拍；特意取 σ=1.0 与 σ=2.0 两组，令 alpha_t != sigma_t，
    # 使 dpmsolver/sample 的 /sigma_t 分母变异不至于退化为真等价
    mo_l, sample_l = [1.0, -0.5, 2.0], [0.5, 1.0, -1.0]
    for sigma_vp in (1.0, 2.0):
        for alg, pred in [("dpmsolver++", "epsilon"),
                          ("dpmsolver++", "v_prediction"),
                          ("dpmsolver", "sample"),
                          ("dpmsolver", "v_prediction")]:
            r = solve(mode="convert", model_output=T(mo_l), sample=T(sample_l),
                      sigma=sigma_vp, algorithm_type=alg, prediction_type=pred,
                      use_flow_sigmas=False)
            ref = [ref_convert(m, s, sigma_vp, alg, pred, False)[0]
                   for m, s in zip(mo_l, sample_l)]
            close(r["converted"], ref)
    # flow 约定下 alpha_t = 1-sigma、sigma_t = sigma，只有 sigma=0.5 时两者相等；
    # 再取 sigma=0.25 覆盖 /sigma_t 与 /alpha_t 的分叉
    r = solve(mode="convert", model_output=T(0.5), sample=T(2.0), sigma=0.25,
              algorithm_type="dpmsolver", prediction_type="sample",
              use_flow_sigmas=True)
    close(r["converted"], (2.0 - 0.75 * 0.5) / 0.25)
    r = solve(mode="convert", model_output=T([1.0, 2.0]), sample=T([2.0, 2.0]),
              sigma=0.25, algorithm_type="dpmsolver++",
              prediction_type="flow_prediction", use_flow_sigmas=False)
    # VP: σ_t=0.25/√1.0625≈0.2425，但官方用原始 σ=0.25
    close(r["converted"], [2.0 - 0.25 * 1.0, 2.0 - 0.25 * 2.0])
    for alg, pred in [("dpmsolver++", "bogus"), ("dpmsolver", "flow_prediction"),
                      ("bogus", "epsilon")]:
        try:
            solve(mode="convert", model_output=T(1.0), sample=T(1.0), sigma=0.5,
                  algorithm_type=alg, prediction_type=pred, use_flow_sigmas=True)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def check_first(solve):
    """一阶更新：flow 手算锚点、VP 手算锚点、双 algorithm 对拍、h<0。"""
    # flow σ=[0.2, 0.4]: λ=log4 → log1.5, h=log(3/8), exp(-h)=8/3, exp(h)=3/8
    sig = [0.2, 0.4]
    r = solve(mode="first", sigmas=T(sig), step_index=0,
              sample=T(1.0), model_output=T(0.5),
              algorithm_type="dpmsolver++", use_flow_sigmas=True)
    close(r["h"], math.log(1.5) - math.log(4.0))
    assert r["h"] < 0
    close(r["c_sample"], 2.0)
    close(r["c_model"], -1.0)
    close(r["prev_sample"], 1.5)
    # dpmsolver：c_sample=0.6/0.8=0.75，c_model=−(0.4*(3/8−1))=0.25
    r = solve(mode="first", sigmas=T(sig), step_index=0,
              sample=T(1.0), model_output=T(0.5),
              algorithm_type="dpmsolver", use_flow_sigmas=True)
    close(r["c_sample"], 0.75)
    close(r["c_model"], 0.25)
    close(r["prev_sample"], 0.875)
    # VP σ=[1,2]: λ_s=0, λ_t=−log2, exp(−h)=2
    r = solve(mode="first", sigmas=T([1.0, 2.0]), step_index=0,
              sample=T(1.0), model_output=T(0.5),
              algorithm_type="dpmsolver++", use_flow_sigmas=False)
    close(r["h"], -math.log(2.0))
    close(r["c_sample"], 2.0 * math.sqrt(2.0 / 5.0))
    close(r["c_model"], -1.0 / math.sqrt(5.0))
    close(r["prev_sample"],
          2.0 * math.sqrt(2.0 / 5.0) - 0.5 / math.sqrt(5.0))
    # 张量逐元素 + 双 algorithm / 双约定对拍
    for flow, sg in [(True, [0.1, 0.25, 0.5]), (False, [0.5, 1.0, 3.0]),
                     (True, [0.05, 0.2, 0.5]), (False, [0.3, 1.0, 4.0])]:
        for alg in ("dpmsolver++", "dpmsolver"):
            sample = [0.3, -1.2, 2.5]
            mo = [1.1, 0.4, -0.7]
            rr = solve(mode="first", sigmas=T(sg), step_index=1,
                       sample=T(sample), model_output=T(mo),
                       algorithm_type=alg, use_flow_sigmas=flow)
            ref = [ref_first(sg, 1, x, m, alg, flow) for x, m in zip(sample, mo)]
            close(rr["prev_sample"], ref)
    # 校验：σ 非递增 / step_index 越界 / flow σ≥1
    for sg, i, flow in [([0.4, 0.2], 0, True), ([0.2, 0.4], 1, True),
                        ([0.2, 0.4], -1, True), ([0.2, 1.0], 0, True)]:
        try:
            solve(mode="first", sigmas=T(sg), step_index=i, sample=T(1.0),
                  model_output=T(0.5), algorithm_type="dpmsolver++",
                  use_flow_sigmas=flow)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def check_second(solve):
    """二阶多步：r0/D1 手算、midpoint/heun × 双 algorithm 对拍、错误分支。"""
    # flow σ=[1/9,1/5,1/3]: λ=log8,log4,log2 → h=h0=−log2, r0=1
    sig = [1.0 / 9.0, 1.0 / 5.0, 1.0 / 3.0]
    m1, m0, sample = 0.2, 0.6, 1.0
    r = solve(mode="second", sigmas=T(sig), step_index=1,
              sample=T(sample), model_outputs=[T(m1), T(m0)],
              algorithm_type="dpmsolver++", solver_type="midpoint",
              use_flow_sigmas=True)
    close(r["h"], -math.log(2.0))
    close(r["h_0"], -math.log(2.0))
    close(r["r0"], 1.0)
    close(r["D1"], m0 - m1)
    close(r["c_sample"], 5.0 / 3.0)
    close(r["c_d0"], -2.0 / 3.0)
    close(r["c_d1"], -1.0 / 3.0)
    # prev = 5/3 − (2/3)*0.6 − (1/3)*0.4 = 1.1333...
    close(r["prev_sample"], 5.0 / 3.0 - 2.0 / 3.0 * 0.6 - 1.0 / 3.0 * 0.4)
    for flow, sg in [(True, [1 / 17, 1 / 9, 1 / 5, 1 / 3]),
                     (False, [0.5, 1.0, 2.0, 4.0]),
                     (True, [0.05, 0.1, 0.2, 0.5]),
                     (False, [0.1, 0.4, 1.0, 3.0])]:
        for alg in ("dpmsolver++", "dpmsolver"):
            for stype in ("midpoint", "heun"):
                sample_l = [0.3, -1.2, 2.5]
                m1_l = [0.1, 0.7, -0.4]
                m0_l = [1.1, 0.4, -0.7]
                rr = solve(mode="second", sigmas=T(sg), step_index=2,
                           sample=T(sample_l),
                           model_outputs=[T(m1_l), T(m0_l)],
                           algorithm_type=alg, solver_type=stype,
                           use_flow_sigmas=flow)
                ref = [ref_second(sg, 2, x, a, b, alg, stype, flow)
                       for x, a, b in zip(sample_l, m1_l, m0_l)]
                close(rr["prev_sample"], ref)
                close(rr["r0"], (ref_lam(sg[2], flow)[0] - ref_lam(sg[1], flow)[0])
                      / (ref_lam(sg[3], flow)[0] - ref_lam(sg[2], flow)[0]))
    for stype, alg in [("bogus", "dpmsolver++"), ("midpoint", "bogus")]:
        try:
            solve(mode="second", sigmas=T(sig), step_index=1, sample=T(1.0),
                  model_outputs=[T(0.2), T(0.6)], algorithm_type=alg,
                  solver_type=stype, use_flow_sigmas=True)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        solve(mode="second", sigmas=T(sig), step_index=1, sample=T(1.0),
              model_outputs=[T(0.2)], algorithm_type="dpmsolver++",
              solver_type="midpoint", use_flow_sigmas=True)
        raise AssertionError("expected ValueError for history length")
    except ValueError:
        pass
    try:
        solve(mode="second", sigmas=T([0.2, 0.4]), step_index=1,
              sample=T(1.0), model_outputs=[T(0.2), T(0.6)],
              algorithm_type="dpmsolver++", solver_type="midpoint",
              use_flow_sigmas=True)
        raise AssertionError("expected ValueError for missing preceding sigma")
    except ValueError:
        pass


def check_third(solve):
    """三阶多步：r0=r1=1、D1_0/D1_1/D1/D2、双 algorithm 对拍、错误分支。"""
    # flow λ=log16,log8,log4,log2 → h=h0=h1=−log2, r0=r1=1
    sig = [1 / 17, 1 / 9, 1 / 5, 1 / 3]
    m2, m1, m0, sample = 0.1, 0.2, 0.6, 1.0
    r = solve(mode="third", sigmas=T(sig), step_index=2,
              sample=T(sample), model_outputs=[T(m2), T(m1), T(m0)],
              algorithm_type="dpmsolver++", use_flow_sigmas=True)
    close(r["h"], -math.log(2.0))
    close(r["h_0"], -math.log(2.0))
    close(r["h_1"], -math.log(2.0))
    close(r["r0"], 1.0)
    close(r["r1"], 1.0)
    close(r["D1_0"], m0 - m1)
    close(r["D1_1"], m1 - m2)
    close(r["D1"], (m0 - m1) + 0.5 * ((m0 - m1) - (m1 - m2)))
    close(r["D2"], 0.5 * ((m0 - m1) - (m1 - m2)))
    close(r["c_sample"], 5.0 / 3.0)
    close(r["c_d0"], -2.0 / 3.0)
    for flow, sg in [(True, [1 / 33, 1 / 17, 1 / 9, 1 / 5, 1 / 3]),
                     (False, [0.25, 0.5, 1.0, 2.0, 4.0]),
                     (True, [0.02, 0.05, 0.1, 0.2, 0.5]),
                     (False, [0.1, 0.4, 1.0, 3.0, 9.0])]:
        for alg in ("dpmsolver++", "dpmsolver"):
            sample_l = [0.3, -1.2, 2.5]
            m2_l = [-0.2, 0.9, 0.3]
            m1_l = [0.1, 0.7, -0.4]
            m0_l = [1.1, 0.4, -0.7]
            rr = solve(mode="third", sigmas=T(sg), step_index=3,
                       sample=T(sample_l),
                       model_outputs=[T(m2_l), T(m1_l), T(m0_l)],
                       algorithm_type=alg, use_flow_sigmas=flow)
            ref = [ref_third(sg, 3, x, c, a, b, alg, flow)
                   for x, c, a, b in zip(sample_l, m2_l, m1_l, m0_l)]
            close(rr["prev_sample"], ref)
            close(rr["D2"], (rr["D1_0"] - rr["D1_1"]) / (rr["r0"] + rr["r1"]))
    for alg in ("bogus",):
        try:
            solve(mode="third", sigmas=T(sig), step_index=2, sample=T(1.0),
                  model_outputs=[T(0.1), T(0.2), T(0.6)],
                  algorithm_type=alg, use_flow_sigmas=True)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        solve(mode="third", sigmas=T(sig), step_index=2, sample=T(1.0),
              model_outputs=[T(0.1), T(0.2)],
              algorithm_type="dpmsolver++", use_flow_sigmas=True)
        raise AssertionError("expected ValueError for history length")
    except ValueError:
        pass


def check_step(solve):
    """单调用状态机：降阶标志、历史移位、选阶、计数器、state 往返。"""
    # 6 步 flow 表，λ=log64..log2（等比数列），每步 h=−log2
    n = 6
    sig = [1.0 / (1 + 2 ** (n + 1 - i)) for i in range(n + 1)]
    cfg = dict(sigmas=T(sig), n_steps=n, algorithm_type="dpmsolver++",
               prediction_type="sample", solver_type="midpoint",
               solver_order=3, lower_order_final=True, euler_at_final=False,
               use_flow_sigmas=True)
    state = None
    orders = []
    prev = None
    for i in range(n):
        mo = 0.1 * (i + 1)
        r = solve(mode="step", step_index=i, sample=T(prev if prev is not None else 1.0),
                  model_output=T(mo), state=state, **cfg)
        orders.append(r["used_order"])
        # 独立 oracle：按返回 used_order 重算
        hist = r["state"]["model_outputs"]
        conv = r["converted"]
        sample = prev if prev is not None else 1.0
        if r["used_order"] == 1:
            ref = ref_first(sig, i, sample, conv, "dpmsolver++", True)
        elif r["used_order"] == 2:
            ref = ref_second(sig, i, sample, hist[-2], hist[-1],
                             "dpmsolver++", "midpoint", True)
        else:
            ref = ref_third(sig, i, sample, hist[-3], hist[-2], hist[-1],
                            "dpmsolver++", True)
        close(r["prev_sample"], ref)
        # prediction_type="sample" 时 dpmsolver++ 的 converted 就是 mo
        close(r["converted"], mo)
        assert r["kind"] == "x0_pred"
        assert r["lower_order_nums_after"] == min(i + 1, 3)
        assert len(hist) == 3
        prev = r["prev_sample"]
        state = r["state"]
    # N=6<15 → 末步降一阶、倒数第二步降二阶；warmup 1,2,3,3,2,1
    assert orders == [1, 2, 3, 3, 2, 1], orders
    assert state["lower_order_nums"] == 3
    # n_steps>=15 时 lower_order_final/second 都不再触发（N=15 恰好边界）
    sig15 = [1.0 / (1 + 2 ** (16 - i)) for i in range(16)]
    r0 = solve(mode="step", step_index=14, sample=T(1.0), model_output=T(0.3),
               state={"model_outputs": [T(0.1), T(0.2), T(0.3)],
                      "lower_order_nums": 2},
               sigmas=T(sig15), n_steps=15, algorithm_type="dpmsolver++",
               prediction_type="sample", solver_type="midpoint",
               solver_order=3, lower_order_final=True, euler_at_final=False,
               use_flow_sigmas=True)
    assert r0["lower_order_final"] is False
    assert r0["lower_order_second"] is False
    assert r0["used_order"] == 3
    r1 = solve(mode="step", step_index=13, sample=T(1.0), model_output=T(0.3),
               state={"model_outputs": [T(0.1), T(0.2), T(0.3)],
                      "lower_order_nums": 2},
               sigmas=T(sig15), n_steps=15, algorithm_type="dpmsolver++",
               prediction_type="sample", solver_type="midpoint",
               solver_order=3, lower_order_final=True, euler_at_final=False,
               use_flow_sigmas=True)
    assert r1["lower_order_second"] is False
    # euler_at_final 单独触发末步降阶
    r2 = solve(mode="step", step_index=14, sample=T(1.0), model_output=T(0.3),
               state=None, sigmas=T(sig15), n_steps=15,
               algorithm_type="dpmsolver", prediction_type="epsilon",
               solver_type="heun", solver_order=2, lower_order_final=False,
               euler_at_final=True, use_flow_sigmas=False)
    assert r2["lower_order_final"] is True
    assert r2["used_order"] == 1
    assert r2["kind"] == "epsilon"
    assert r2["lower_order_nums_after"] == 1
    # 历史左移一格、末位写本步结果
    r3 = solve(mode="step", step_index=2, sample=T(1.0), model_output=T(0.9),
               state={"model_outputs": [T(0.1), T(0.2), T(0.3)],
                      "lower_order_nums": 2},
               sigmas=T(sig), n_steps=n, algorithm_type="dpmsolver++",
               prediction_type="sample", solver_type="midpoint",
               solver_order=3, lower_order_final=True, euler_at_final=False,
               use_flow_sigmas=True)
    hist = r3["state"]["model_outputs"]
    close(hist[0], 0.2)
    close(hist[1], 0.3)
    close(hist[2], 0.9)
    # 错误分支
    for bad in [dict(solver_order=4), dict(n_steps=7), dict(sigmas=T([0.4, 0.2]))]:
        kw = dict(cfg)
        kw.update(bad)
        kw.pop("sigmas", None)
        sg = bad.get("sigmas", T(sig))
        try:
            solve(mode="step", sigmas=sg, step_index=0, sample=T(1.0),
                  model_output=T(0.5), state=None, **kw)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
