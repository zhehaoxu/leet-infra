import math

import torch

_WIDTH = 0.3
_PEAK = 5.0
_SHARP = 10.0
_INV = 1.0 / _WIDTH


def _oracle(logits, target, decay, ws, wt, u):
    """独立重算：逐行 python 计算峰形变换 + 原顺序逆变换 + EMA 更新。"""
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    uf = u.reshape(-1)
    n = flat.shape[-1]
    out_f = torch.zeros_like(flat)
    out_c = []
    out_ws = []
    out_wt = []
    for r in range(flat.shape[0]):
        p = [float(x) for x in flat[r]]
        at = target if wt == 0.0 else min(1.0, max(0.0, 2.0 * target - ws / wt))
        at = min(1.0, max(0.0, at))
        nl = [_PEAK - _SHARP * (abs(x - at) * _INV) ** 2 / (1.0 + abs(x - at) * _INV) for x in p]
        mx = max(nl)
        ex = [math.exp(v - mx) for v in nl]
        s = sum(ex)
        pf = [e / s for e in ex]
        out_f[r] = torch.tensor(pf)
        cdf = 0.0
        pick = n - 1
        for i in range(n):
            cdf += pf[i]
            if cdf >= float(uf[r]):
                pick = i
                break
        out_c.append(pick)
        out_ws.append(p[pick] + decay * ws)
        out_wt.append(1.0 + decay * wt)
    return (out_f.reshape(probs.shape), torch.tensor(out_c),
            torch.tensor(out_ws), torch.tensor(out_wt))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    f, c, ws, wt = solve(logits, 0.5, 0.9, 0.0, 0.0, torch.tensor(0.5))
    exp = torch.tensor([0.971617, 0.025311, 0.002747, 0.000157, 0.000092, 0.000075])
    assert torch.allclose(f, exp, atol=1e-5), f.tolist()
    assert c.item() == 0, c.item()
    assert abs(float(ws) - 0.600294) < 1e-5, float(ws)
    assert abs(float(wt) - 1.0) < 1e-6, float(wt)


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for (target, ws, wt) in [(0.5, 0.0, 0.0), (0.3, 0.5, 0.6), (0.8, 0.2, 1.5)]:
        for _ in range(20):
            logits = torch.randn(torch.randint(4, 15, (1,)).item())
            f, c, nws, nwt = solve(logits, target, 0.9, ws, wt, torch.tensor(0.5))
            of, oc, ows, owt = _oracle(logits, target, 0.9, ws, wt, torch.tensor(0.5))
            assert torch.allclose(f, of, atol=1e-6), (target, ws, wt, f.tolist(), of.tolist())
            assert c.item() == oc.item(), (target, c.item(), oc.item())
            assert abs(float(nws) - float(ows)) < 1e-5, (target, ws, wt, float(nws), float(ows))
            assert abs(float(nwt) - float(owt)) < 1e-6


def check_all_positive_and_in_set(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c, ws, wt = solve(logits, 0.5, 0.9, 0.0, 0.0, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert bool((f > 0).all())  # 不删候选
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())


def check_peak_at_target(solve):
    # total_weight==0 → adapted_target==target → filtered 的峰在原概率最接近 target 处
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    probs = torch.softmax(logits, dim=-1)
    f, _, _, _ = solve(logits, 0.1, 0.9, 0.0, 0.0, torch.tensor(0.5))
    arg = int((probs - 0.1).abs().argmin())
    assert int(f.argmax()) == arg, (int(f.argmax()), arg)


def check_ema_total_weight(solve):
    # wt_new = 1 + decay*wt，与 probs 无关
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    _, _, _, wt = solve(logits, 0.5, 0.9, 0.7, 2.0, torch.tensor(0.5))
    assert abs(float(wt) - (1.0 + 0.9 * 2.0)) < 1e-6, float(wt)


check_example.example = True
