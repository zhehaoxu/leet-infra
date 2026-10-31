import math

import torch


def _py(r, k, v, w, u, state=None):
    """独立纯 Python oracle：逐通道 (a, b, p) log-space 递推。"""
    T = len(k)
    C = len(k[0])
    if state is None:
        a = [0.0] * C
        b = [0.0] * C
        p = [-1e38] * C
    else:
        a = [float(x) for x in state["a"]]
        b = [float(x) for x in state["b"]]
        p = [float(x) for x in state["p"]]
    wkv = []
    y = []
    p_trace = []
    for t in range(T):
        row = []
        yrow = []
        ptr = []
        for c in range(C):
            kt, vt, wt, rt, ut = k[t][c], v[t][c], w[t][c], r[t][c], u[c]
            p_out = max(p[c], ut)
            e1 = math.exp(p[c] - p_out)
            e2 = math.exp(ut - p_out)
            wkv_t = (e1 * a[c] + e2 * vt) / (e1 * b[c] + e2)
            ww = p[c] + wt
            p_new = max(ww, kt)
            e1n = math.exp(ww - p_new)
            e2n = math.exp(kt - p_new)
            a[c] = e1n * a[c] + e2n * vt
            b[c] = e1n * b[c] + e2n
            p[c] = p_new
            row.append(wkv_t)
            yrow.append(rt * wkv_t)
            ptr.append(p[c])
        wkv.append(row)
        y.append(yrow)
        p_trace.append(ptr)
    return {
        "wkv": wkv,
        "y": y,
        "final_state": {"a": list(a), "b": list(b), "p": list(p)},
        "p_trace": p_trace,
    }


def _close(actual, expect, atol=1e-5, rtol=1e-5):
    if torch.is_tensor(actual):
        actual = actual.tolist()
    if isinstance(expect, torch.Tensor):
        expect = expect.tolist()

    def flat(x):
        if isinstance(x, list):
            out = []
            for it in x:
                out.extend(flat(it))
            return out
        return [x]

    fa, fe = flat(actual), flat(expect)
    assert len(fa) == len(fe), f"length mismatch {len(fa)} vs {len(fe)}"
    for av, ev in zip(fa, fe):
        assert abs(av - ev) <= atol + rtol * abs(ev), f"{av} vs {ev}"


def _compare(solve, r, k, v, w, u, state=None):
    led = solve(r, k, v, w, u, state)
    ref = _py(
        r.tolist(), k.tolist(), v.tolist(), w.tolist(), u.tolist(),
        None if state is None else {kk: vv.tolist() for kk, vv in state.items()},
    )
    _close(led["wkv"], ref["wkv"])
    _close(led["y"], ref["y"])
    _close(led["final_state"]["a"], ref["final_state"]["a"])
    _close(led["final_state"]["b"], ref["final_state"]["b"])
    _close(led["final_state"]["p"], ref["final_state"]["p"])
    _close(led["p_trace"], ref["p_trace"])
    return led


def check_zero_state_anchor(solve):
    # 零初始态：t=0 的 wkv 恒等于 v（a=b=0），p_trace[0] = k0
    r = torch.tensor([[0.5], [0.7], [0.9]])
    k = torch.tensor([[-1.0], [0.2], [0.6]])
    v = torch.tensor([[3.0], [-2.0], [1.5]])
    w = torch.tensor([[-0.5], [-0.5], [-0.5]])
    u = torch.tensor([0.7])
    led = _compare(solve, r, k, v, w, u)
    assert abs(led["wkv"][0][0].item() - 3.0) < 1e-6
    assert abs(led["p_trace"][0][0] - (-1.0)) < 1e-5


def check_decay_on_old_p_anchor(solve):
    # 手算锚点：初始 p=-2, a=1, b=1；t0: k=-3, v=2, w=-1, u=0
    # 输出分支: p_out=max(-2,0)=0, e1=exp(-2), e2=1 → wkv0=(exp(-2)+2)/(exp(-2)+1)
    # 更新分支: ww=-2-1=-3, p=max(-3,-3)=-3, a=exp(0)*1+exp(0)*2=3, b=2, p=-3
    e = math.exp(-2.0)
    r = torch.tensor([[1.0], [1.0]])
    k = torch.tensor([[-3.0], [-3.0]])
    v = torch.tensor([[2.0], [2.0]])
    w = torch.tensor([[-1.0], [-1.0]])
    u = torch.tensor([0.0])
    state = {"a": torch.tensor([1.0]), "b": torch.tensor([1.0]), "p": torch.tensor([-2.0])}
    led = _compare(solve, r, k, v, w, u, state)
    assert abs(led["wkv"][0][0].item() - (e + 2.0) / (e + 1.0)) < 1e-5
    # t1: pp=-3, ww=-4, p=max(-4,-3)=-3 → a=exp(-1)*3+2, b=exp(-1)*2+1
    e1 = math.exp(-1.0)
    assert abs(led["final_state"]["a"][0].item() - (e1 * 3.0 + 2.0)) < 1e-5
    assert abs(led["final_state"]["b"][0].item() - (e1 * 2.0 + 1.0)) < 1e-5
    assert abs(led["p_trace"][1][0] - (-3.0)) < 1e-5


def check_bonus_anchor(solve):
    # u 比 pp 高：p_out=u，wkv 由 bonus 分支主导
    r = torch.tensor([[1.0]])
    k = torch.tensor([[-0.3]])
    v = torch.tensor([[5.0]])
    w = torch.tensor([[-0.5]])
    u = torch.tensor([0.7])
    state = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0]), "p": torch.tensor([0.0])}
    led = _compare(solve, r, k, v, w, u, state)
    e = math.exp(-0.7)
    expect = (e * 2.0 + 5.0) / (e + 1.0)
    assert abs(led["wkv"][0][0].item() - expect) < 1e-5


def check_oracle_multichannel(solve):
    torch.manual_seed(7)
    T, C = 6, 4
    r = torch.randn(T, C)
    k = torch.randn(T, C) * 0.8
    v = torch.randn(T, C)
    w = -torch.rand(T, C) * 2.0  # 负衰减
    u = torch.randn(C) * 0.5
    led = _compare(solve, r, k, v, w, u)
    # receptance 原样乘回
    assert abs(led["y"][3][2].item() - r[3][2].item() * led["wkv"][3][2].item()) < 1e-6


def check_state_composition(solve):
    # 分两段喂入与整段一次喂入结果一致（递推的因果性）
    torch.manual_seed(11)
    T, C = 6, 2
    r = torch.randn(T, C)
    k = torch.randn(T, C) * 0.5
    v = torch.randn(T, C)
    w = -torch.rand(T, C) * 1.5
    u = torch.randn(C) * 0.3
    full = solve(r, k, v, w, u)
    part1 = solve(r[:3], k[:3], v[:3], w[:3], u)
    part2 = solve(r[3:], k[3:], v[3:], w[3:], u, part1["final_state"])
    _close(part2["final_state"]["a"], full["final_state"]["a"], atol=1e-6, rtol=1e-6)
    _close(part2["final_state"]["b"], full["final_state"]["b"], atol=1e-6, rtol=1e-6)
    _close(part2["final_state"]["p"], full["final_state"]["p"], atol=1e-6, rtol=1e-6)
    _close(part2["wkv"], full["wkv"][3:], atol=1e-6, rtol=1e-6)


def check_channel_independence(solve):
    # 只扰动通道 1，其余通道输出逐位不变
    torch.manual_seed(3)
    T, C = 4, 3
    r = torch.randn(T, C)
    k = torch.randn(T, C) * 0.6
    v = torch.randn(T, C)
    w = -torch.rand(T, C)
    u = torch.zeros(C)
    base = solve(r, k, v, w, u)
    r2 = r.clone(); r2[:, 1] += 5.0
    k2 = k.clone(); k2[:, 1] -= 2.0
    v2 = v.clone(); v2[:, 1] += 3.0
    w2 = w.clone(); w2[:, 1] -= 1.0
    other = solve(r2, k2, v2, w2, u)
    assert torch.equal(base["wkv"][:, 0], other["wkv"][:, 0])
    assert torch.equal(base["wkv"][:, 2], other["wkv"][:, 2])
    assert not torch.allclose(base["wkv"][:, 1], other["wkv"][:, 1])


def check_single_token_zero_state(solve):
    r = torch.tensor([[0.5, 2.0]])
    k = torch.tensor([[9.0, -9.0]])
    v = torch.tensor([[4.0, -6.0]])
    w = torch.tensor([[-0.1, -0.2]])
    u = torch.tensor([0.3, 0.4])
    led = _compare(solve, r, k, v, w, u)
    assert abs(led["y"][0][0].item() - 2.0) < 1e-6
    assert abs(led["y"][0][1].item() - (-12.0)) < 1e-6
    # 更新分支：p = max(-1e38 + w, k) = k
    assert abs(led["p_trace"][0][0] - 9.0) < 1e-4
    assert abs(led["p_trace"][0][1] - (-9.0)) < 1e-4


def check_invalid_inputs(solve):
    r = torch.randn(2, 3)
    k = torch.randn(2, 3)
    v = torch.randn(2, 3)
    w = -torch.rand(2, 3)
    u = torch.randn(3)
    bad_cases = [
        lambda: solve(r, k, v[:, :2], w, u),
        lambda: solve(r, k, v, w, u[:2]),
        lambda: solve(r, torch.randn(3, 3), v, w, u),
        lambda: solve(torch.randn(2), k, v, w, u),
        lambda: solve(r, k, v, w, torch.randn(3, 1)),
        lambda: solve(torch.zeros(0, 3), torch.zeros(0, 3), torch.zeros(0, 3),
                      torch.zeros(0, 3), u),
        lambda: solve(r, k, v, w, u, state={"a": torch.zeros(3), "b": torch.zeros(3)}),
        lambda: solve(r, k, v, w, u,
                      state={"a": torch.zeros(2), "b": torch.zeros(3), "p": torch.zeros(3)}),
    ]
    for fn in bad_cases:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
