import math

import torch

PRI = torch.tensor([1.0, 2.0, 3.0, 2.0])
U = torch.tensor([0.05, 0.40, 0.72, 0.95])


def _oracle(priorities, uniform_draws, alpha, beta):
    """独立重算：纯 Python 列表 + 逐次累加的线性扫描定位下标（与解法的 cumsum+searchsorted 结构不同）。"""
    per = [max(float(x), 0.0) for x in priorities.reshape(-1).tolist()]
    us = [float(x) for x in uniform_draws.reshape(-1).tolist()]
    n = len(per)
    a = float(alpha)
    b = float(beta)
    if n == 0:                        # 空池：下标 -1、权重 0、probs 空
        return (torch.full((len(us),), -1, dtype=torch.long),
                torch.zeros(len(us), dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32))
    powered = [math.pow(p, a) for p in per]      # 0**0 == 1
    total = math.fsum(powered)                   # 高精度求和，与向量化路径不同的求和顺序
    probs = [1.0 / n] * n if total == 0.0 else [q / total for q in powered]

    idx = []
    for u in us:
        hit = n - 1
        acc = 0.0
        for j in range(n):
            acc += probs[j]
            if u < acc:
                hit = j
                break
        idx.append(hit)

    raw = [math.pow(float(n) * probs[i], -b) if probs[i] > 0.0 else float("inf") for i in idx]
    mx = max(raw) if raw else 1.0
    weights = [v / mx for v in raw]
    return (torch.tensor(idx, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(probs, dtype=torch.float32))


# ---- 样例用例（点「运行」只跑 example = True 的） ------------------------------------
def check_example(solve):
    idx, w, probs = solve(PRI, U, 1.0, 0.5)
    assert torch.equal(idx, torch.tensor([0, 2, 2, 3])), idx
    assert torch.allclose(w, torch.tensor([1.0, 0.57735026, 0.57735026, 0.70710677]),
                          atol=1e-5, rtol=1e-5), w
    assert torch.allclose(probs, torch.tensor([0.125, 0.25, 0.375, 0.25]),
                          atol=1e-5, rtol=1e-5), probs


check_example.example = True


def check_example_beta_one(solve):
    idx, w, _ = solve(PRI, U, 1.0, 1.0)
    assert torch.equal(idx, torch.tensor([0, 2, 2, 3])), idx
    assert torch.allclose(w, torch.tensor([1.0, 1.0 / 3.0, 1.0 / 3.0, 0.5]),
                          atol=1e-5, rtol=1e-5), w


check_example_beta_one.example = True


# ---- 退化探针 --------------------------------------------------------------------
def check_alpha_zero_is_uniform(solve):
    """alpha=0 → P≡1/N、w≡1，下标就是等宽分箱 floor(u*N)。"""
    u = torch.tensor([0.0, 0.1, 0.49, 0.5, 0.7, 0.9999999])
    idx, w, probs = solve(PRI, u, 0.0, 0.9)
    assert torch.allclose(probs, torch.full((4,), 0.25), atol=1e-6), probs
    assert torch.equal(idx, torch.tensor([0, 0, 1, 2, 2, 3])), idx
    assert torch.allclose(w, torch.ones(6), atol=1e-6, rtol=1e-6), w
    assert torch.equal(idx, _oracle(PRI, u, 0.0, 0.9)[0])


def check_beta_zero_no_correction(solve):
    """beta=0 → 原始权重全 1，归一后仍全 1，但采样概率仍按 alpha 倾斜。"""
    idx, w, probs = solve(PRI, U, 1.7, 0.0)
    assert torch.allclose(w, torch.ones(4), atol=1e-6, rtol=1e-6), w
    assert probs[2] > probs[0], probs
    assert torch.equal(idx, _oracle(PRI, U, 1.7, 0.0)[0])


def check_single_transition_buffer(solve):
    """N=1：下标恒 0、N*P=1 → 权重恒 1，与 alpha/beta 无关。"""
    for alpha, beta in [(0.0, 0.0), (0.4, 0.7), (2.0, 1.0)]:
        idx, w, probs = solve(torch.tensor([3.5]), U, alpha, beta)
        assert torch.equal(idx, torch.zeros(4, dtype=torch.long)), idx
        assert torch.allclose(w, torch.ones(4), atol=1e-6, rtol=1e-6), w
        assert torch.allclose(probs, torch.ones(1), atol=1e-6), probs


def check_zero_priority_never_sampled(solve):
    """p_i=0 且 alpha>0 → P(i)=0，随机数永远不会落到这些下标；u 恰等于前缀和时进下一格（严格大于）。"""
    per = torch.tensor([0.0, 5.0, 0.0, 3.0])
    u = torch.tensor([0.0, 0.3, 0.6249, 0.625, 0.99, 0.6250001])
    idx, w, probs = solve(per, u, 1.0, 0.6)
    assert torch.allclose(probs, torch.tensor([0.0, 0.625, 0.0, 0.375]), atol=1e-6), probs
    assert torch.equal(idx, torch.tensor([1, 1, 1, 3, 3, 3])), idx
    assert w.max().item() == 1.0, w
    ref_i, ref_w, ref_p = _oracle(per, u, 1.0, 0.6)
    assert torch.equal(idx, ref_i) and torch.allclose(w, ref_w, atol=1e-5, rtol=1e-5)
    assert torch.allclose(probs, ref_p, atol=1e-5, rtol=1e-5)


def check_all_zero_priority_falls_back_uniform(solve):
    """全零优先级 + alpha>0 → Σ=0，按题面退化成均匀分布，权重全 1。"""
    per = torch.tensor([0.0, 0.0, 0.0])
    u = torch.tensor([0.05, 0.4, 0.7, 0.999])
    idx, w, probs = solve(per, u, 2.0, 0.5)
    assert torch.allclose(probs, torch.full((3,), 1.0 / 3.0), atol=1e-6), probs
    assert torch.equal(idx, torch.tensor([0, 1, 2, 2])), idx
    assert torch.allclose(w, torch.ones(4), atol=1e-6, rtol=1e-6), w


def check_equal_priorities_uniform(solve):
    per = torch.full((6,), 2.5)
    u = torch.tensor([0.0, 0.16, 0.33, 0.5, 0.83, 0.999])
    idx, w, probs = solve(per, u, 3.0, 0.8)
    assert torch.allclose(probs, torch.full((6,), 1.0 / 6.0), atol=1e-6), probs
    assert torch.allclose(w, torch.ones(6), atol=1e-6, rtol=1e-6), w
    assert torch.equal(idx, torch.tensor([0, 0, 1, 3, 4, 5])), idx


def check_empty_inputs(solve):
    """空池 N=0：indices 全 -1、weights 全 0、probs 空；K=0：只回 probs。"""
    idx, w, probs = solve(torch.zeros(0), torch.tensor([0.1, 0.9]), 1.0, 0.5)
    assert torch.equal(idx, torch.tensor([-1, -1])), idx
    assert w.numel() == 2 and float(w.abs().sum()) == 0.0
    assert probs.numel() == 0
    assert torch.equal(idx, _oracle(torch.zeros(0), torch.tensor([0.1, 0.9]), 1.0, 0.5)[0])
    idx, w, probs = solve(PRI, torch.zeros(0), 1.0, 0.5)
    assert idx.numel() == 0 and w.numel() == 0
    assert probs.numel() == 4
    assert torch.allclose(probs, torch.tensor([0.125, 0.25, 0.375, 0.25]), atol=1e-6)


# ---- 规模用例：与独立 oracle 对拍 --------------------------------------------------
def check_matches_oracle_varied(solve):
    torch.manual_seed(7)
    cases = [
        (torch.tensor([0.25, 4.0, 1.5, 0.0, 9.0, 2.0, 0.75, 3.0]), 0.4, 0.6),
        (torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), 1.0, 1.0),
        (1.0 + 5.0 * torch.rand(50), 1.0, 0.9),
        (1.0 + 50.0 * torch.rand(64), 0.6, 0.4),
        (torch.rand(37) ** 2, 1.0, 1.0),
    ]
    u = torch.linspace(0.0, 1.0 - 1e-7, 101)
    for per, alpha, beta in cases:
        idx, w, probs = solve(per, u, alpha, beta)
        ref_i, ref_w, ref_p = _oracle(per, u, alpha, beta)
        assert torch.equal(idx, ref_i), (per, alpha, beta, idx, ref_i)
        assert torch.allclose(w, ref_w, atol=1e-5, rtol=1e-5), (alpha, beta, w, ref_w)
        assert torch.allclose(probs, ref_p, atol=1e-5, rtol=1e-5), (alpha, beta, probs)


def check_normalization_and_distribution_invariants(solve):
    torch.manual_seed(11)
    per = 0.1 + 9.0 * torch.rand(200)
    u = torch.rand(500)
    top = int(per.argmax())
    for alpha, beta in [(1.0, 0.5), (0.3, 1.0), (2.0, 0.0)]:
        idx, w, probs = solve(per, u, alpha, beta)
        assert abs(float(probs.sum()) - 1.0) < 1e-4, float(probs.sum())
        assert float(probs.min()) >= 0.0
        assert int(idx.min()) >= 0 and int(idx.max()) <= 199
        assert abs(float(w.max()) - 1.0) < 1e-6, float(w.max())
        assert float(w.min()) > 0.0
    # alpha 越大 → 最高优先级样本被采中的概率越集中
    p_small = solve(per, u, 0.0, 0.5)[2][top].item()
    p_mid = solve(per, u, 1.0, 0.5)[2][top].item()
    p_big = solve(per, u, 2.0, 0.5)[2][top].item()
    assert p_small < p_mid < p_big, (p_small, p_mid, p_big)


def check_large_buffer_throughput(solve):
    """N=20000、K=2000：形状/取值域正确，且与 oracle 在子集上一致。"""
    torch.manual_seed(3)
    per = torch.rand(20000) * 8.0 + 0.01
    u = torch.rand(2000)
    idx, w, probs = solve(per, u, 1.0, 0.5)
    assert idx.shape == (2000,) and idx.dtype == torch.long
    assert probs.shape == (20000,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert float(w.max()) - 1.0 < 1e-6
    ref_i, ref_w, _ = _oracle(per, u, 1.0, 0.5)
    assert torch.equal(idx, ref_i)
    assert torch.allclose(w, ref_w, atol=1e-5, rtol=1e-5)
