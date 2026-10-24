import math

import torch

_TINY = 1e-30


def _oracle(logits, temp, delta, exponent):
    """独立重算：逐行 python 计算熵 + 动态温度 + softmax(logits/dyn)。"""
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[-1]
    flat = probs.reshape(-1, n)
    lflat = logits.reshape(-1, n)
    out = torch.zeros_like(flat)
    dyns = []
    for r in range(flat.shape[0]):
        if delta <= 0:
            dyn = temp
        else:
            p = [float(x) for x in flat[r]]
            maxe = math.log(n)
            ent = sum(-x * math.log(max(x, _TINY)) for x in p)
            c = ent / maxe
            tmin = max(0.0, temp - delta)
            tmax = temp + delta
            dyn = tmin + (tmax - tmin) * (c ** exponent)
        z = [float(x) / dyn for x in lflat[r]]
        mx = max(z)
        ex = [math.exp(v - mx) for v in z]
        s = sum(ex)
        out[r] = torch.tensor([e / s for e in ex])
        dyns.append(dyn)
    return out.reshape(probs.shape), torch.tensor(dyns).reshape(probs.shape[:-1])


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    f, c, dyn = solve(logits, 1.0, 0.3, 1.0, torch.tensor(0.5))
    exp = torch.tensor([0.579178, 0.226207, 0.141368, 0.034506, 0.013477, 0.005264])
    assert torch.allclose(f, exp, atol=1e-5), f.tolist()
    assert c.item() == 0, c.item()
    assert abs(float(dyn) - 1.063648) < 1e-4, float(dyn)


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for (temp, delta, exponent) in [(1.0, 0.3, 1.0), (1.0, 0.5, 2.0), (0.7, 0.0, 1.0), (1.5, 0.2, 0.5)]:
        for _ in range(20):
            logits = torch.randn(torch.randint(4, 15, (1,)).item())
            f, _, dyn = solve(logits, temp, delta, exponent, torch.tensor(0.5))
            of, odyn = _oracle(logits, temp, delta, exponent)
            assert torch.allclose(f, of, atol=1e-6), (temp, delta, exponent)
            assert abs(float(dyn) - float(odyn)) < 1e-5, (temp, delta, exponent, float(dyn), float(odyn))


def check_sum_and_in_set(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c, dyn = solve(logits, 1.0, 0.4, 1.0, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert bool((f > 0).all())
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())
    assert dyn.shape == (5,), dyn.shape


def check_delta_zero_is_static_temp(solve):
    # delta<=0 → 退化固定温度：dyn==temp, filtered==softmax(logits/temp)
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    f, _, dyn = solve(logits, 2.0, 0.0, 1.0, torch.tensor(0.5))
    assert float(dyn) == 2.0, float(dyn)
    assert torch.allclose(f, torch.softmax(logits / 2.0, dim=-1), atol=1e-6), f.tolist()


def check_uniform_hits_max_temp(solve):
    # 均匀分布 C=1 → dyn = T_max = temp + delta
    logits = torch.zeros(6)
    _, _, dyn = solve(logits, 1.0, 0.3, 1.0, torch.tensor(0.5))
    assert abs(float(dyn) - 1.3) < 1e-6, float(dyn)


def check_sharp_hits_min_temp(solve):
    # 极尖分布 C→0 → dyn → T_min = max(0, temp-delta)
    logits = torch.tensor([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, _, dyn = solve(logits, 1.0, 0.3, 1.0, torch.tensor(0.5))
    assert float(dyn) < 0.75, float(dyn)


check_example.example = True
