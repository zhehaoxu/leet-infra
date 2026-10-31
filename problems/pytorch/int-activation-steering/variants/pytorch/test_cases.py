import math

import torch


def _py(hiddens, s, alphas, layer_indices, normalize=False):
    """独立纯 Python oracle。"""
    L, T, D = len(hiddens), len(hiddens[0]), len(hiddens[0][0])
    sn = math.sqrt(sum(float(x) ** 2 for x in s))
    s_eff = [float(x) / sn for x in s] if normalize else [float(x) for x in s]
    s_eff_norm = math.sqrt(sum(x * x for x in s_eff))
    steered = [[[float(x) for x in row] for row in layer] for layer in hiddens]
    layer_sum = [0.0] * L
    for k in range(len(alphas)):
        idx = layer_indices[k]
        a = float(alphas[k])
        for t in range(T):
            for d in range(D):
                steered[idx][t][d] += a * s_eff[d]
        layer_sum[idx] += a
    ss = sum(x * x for x in s)
    cos_before, cos_after, proj_coef = [], [], []
    for l in range(L):
        cb, ca, pc = [], [], []
        for t in range(T):
            h = [float(x) for x in hiddens[l][t]]
            hp = steered[l][t]
            hn = math.sqrt(sum(x * x for x in h))
            hpn = math.sqrt(sum(x * x for x in hp))
            dot = sum(h[d] * float(s[d]) for d in range(D))
            cb.append(dot / (hn * sn))
            ca.append(sum(hp[d] * float(s[d]) for d in range(D)) / (hpn * sn))
            pc.append(dot / ss)
        cos_before.append(cb)
        cos_after.append(ca)
        proj_coef.append(pc)
    return {
        "steered": steered,
        "s_norm": sn,
        "layer_delta": [abs(x) * s_eff_norm for x in layer_sum],
        "cos_before": cos_before,
        "cos_after": cos_after,
        "proj_coef": proj_coef,
    }


def _close(actual, expect, atol=1e-9, rtol=1e-9):
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


def _compare(solve, hiddens, s, alphas, layer_indices, normalize=False):
    led = solve(hiddens, s, alphas, layer_indices, normalize)
    ref = _py(
        hiddens.tolist(), s.tolist(), alphas, layer_indices, normalize
    )
    _close(led["steered"], ref["steered"])
    _close(led["s_norm"], ref["s_norm"])
    _close(led["layer_delta"], ref["layer_delta"])
    _close(led["cos_before"], ref["cos_before"])
    _close(led["cos_after"], ref["cos_after"])
    _close(led["proj_coef"], ref["proj_coef"])
    return led


def check_anchor_orthogonal(solve):
    # h=[3,0]、s=[0,4]、α=0.5：正交注入，cos_before=0、cos_after=2/√13
    hiddens = torch.tensor([[[3.0, 0.0]]])
    s = torch.tensor([0.0, 4.0])
    led = _compare(solve, hiddens, s, [0.5], [0])
    assert abs(led["s_norm"] - 4.0) < 1e-12
    assert abs(led["steered"][0][0][0].item() - 3.0) < 1e-12
    assert abs(led["steered"][0][0][1].item() - 2.0) < 1e-12
    assert abs(led["cos_before"][0][0]) < 1e-12
    assert abs(led["cos_after"][0][0] - 2.0 / math.sqrt(13.0)) < 1e-12
    assert abs(led["layer_delta"][0] - 2.0) < 1e-12
    assert led["layer_delta"] == [2.0] or len(led["layer_delta"]) == 1
    # proj_coef = (h·s)/|s|² = 0：α=0 时 h' 已 ⊥ s
    assert abs(led["proj_coef"][0][0]) < 1e-12


def check_normalize(solve):
    # normalize=True：s/|s| 注入
    hiddens = torch.tensor([[[1.0, 1.0]]])
    s = torch.tensor([0.0, 4.0])
    led = _compare(solve, hiddens, s, [0.5], [0], normalize=True)
    assert abs(led["steered"][0][0][1].item() - 1.5) < 1e-12
    assert abs(led["layer_delta"][0] - 0.5) < 1e-12
    # 不归一化时同样的 α 注入 4 倍长度
    led2 = _compare(solve, hiddens, s, [0.5], [0], normalize=False)
    assert abs(led2["steered"][0][0][1].item() - 3.0) < 1e-12  # 1 + 0.5*4
    assert abs(led2["layer_delta"][0] - 2.0) < 1e-12


def check_duplicate_layer_accumulate(solve):
    # 同层两次注入必须累加：α=1,2 → 层内总注入 3|s_eff|
    torch.manual_seed(2)
    hiddens = torch.randn(2, 3, 4)
    s = torch.randn(4)
    led = _compare(solve, hiddens, s, [1.0, 2.0], [0, 0])
    assert abs(led["layer_delta"][0] - 3.0 * float(s.double().norm())) < 1e-9
    assert led["layer_delta"][1] == 0.0
    # 等价于单次 α=3
    led3 = _compare(solve, hiddens, s, [3.0], [0])
    _close(led["steered"], led3["steered"])


def check_multi_layer_order(solve):
    # 跨层注入：每层只动自己的行；cos_before 用原始 hiddens
    torch.manual_seed(7)
    hiddens = torch.randn(3, 2, 5)
    s = torch.randn(5)
    led = _compare(solve, hiddens, s, [0.7, -1.2], [2, 0])
    # 未注入层 1 的 steered == 原值
    _close(led["steered"][1], hiddens[1].double())
    # cos_before 全层都基于原始值（不受注入影响）
    _close(led["cos_before"], [[
        sum(float(hiddens[l, t, d]) * float(s[d]) for d in range(5))
        / (float(hiddens[l, t].double().norm()) * float(s.double().norm()))
        for t in range(2)
    ] for l in range(3)])


def check_proj_coefficient(solve):
    # α = −proj_coef 时 h' ⊥ s
    torch.manual_seed(11)
    hiddens = torch.randn(1, 4, 6)
    s = torch.randn(6)
    led = _compare(solve, hiddens, s, [1.0], [0])
    for t in range(4):
        pc = led["proj_coef"][0][t]
        led2 = solve(hiddens, s, [-pc], [0])
        assert abs(led2["cos_after"][0][t]) < 1e-9


def check_input_not_mutated(solve):
    torch.manual_seed(3)
    hiddens = torch.randn(2, 2, 3)
    snapshot = hiddens.clone()
    s = torch.randn(3)
    led = solve(hiddens, s, [2.0, 1.0], [0, 1])
    assert torch.equal(hiddens, snapshot)
    _close(led["cos_before"], _py(snapshot.tolist(), s.tolist(), [2.0, 1.0], [0, 1])["cos_before"])
    # float64 输入：实现不得原位修改入参（.double() 对 float64 是恒等映射）
    hd = hiddens.double()
    snap2 = hd.clone()
    led2 = solve(hd, s, [2.0, 1.0], [0, 1])
    assert torch.equal(hd, snap2)
    _close(led2["cos_before"], _py(snapshot.tolist(), s.tolist(), [2.0, 1.0], [0, 1])["cos_before"])


def check_invalid_inputs(solve):
    h = torch.randn(2, 3, 4)
    s = torch.randn(4)
    bad = [
        lambda: solve(torch.randn(2, 3), s, [1.0], [0]),
        lambda: solve(h, torch.randn(5), [1.0], [0]),
        lambda: solve(h, torch.zeros(4), [1.0], [0]),
        lambda: solve(h, s, [1.0, 2.0], [0]),
        lambda: solve(h, s, [], []),
        lambda: solve(h, s, [1.0], [2]),
        lambda: solve(h, s, [1.0], [-1]),
        lambda: solve(h, s, [1.0], [0.5]),
        lambda: solve(h, s, [float("nan")], [0]),
        lambda: solve(h, s, [float("inf")], [0]),
        lambda: solve(h, s, [True], [0]),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
