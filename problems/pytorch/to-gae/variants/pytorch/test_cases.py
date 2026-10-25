import torch

REWARDS = torch.tensor([1.0, 0.0, 1.0])
VALUES = torch.tensor([0.5, 0.7, 0.3])


def _oracle(rewards, values, gamma, lam):
    """独立重算：先物化整个 delta 列表，再倒序递推（结构异于参考解的边算边推）。"""
    T = rewards.shape[0]
    delta = []
    for t in range(T):
        nxt = values[t + 1].item() if t + 1 < T else 0.0
        delta.append(rewards[t].item() + gamma * nxt - values[t].item())
    adv = [0.0] * T
    acc = 0.0
    for t in reversed(range(T)):
        acc = delta[t] + gamma * lam * acc
        adv[t] = acc
    a = torch.tensor(adv, dtype=torch.float32)
    return a, a + values


def check_example(solve):
    adv, ret = solve(REWARDS, VALUES, 0.99, 0.95)
    exp_adv = torch.tensor([1.4331567287445068, 0.2553500235080719, 0.699999988079071])
    exp_ret = torch.tensor([1.9331567287445068, 0.9553500413894653, 1.0])
    assert torch.allclose(adv, exp_adv, atol=1e-5, rtol=1e-5), f"adv got {adv.tolist()}"
    assert torch.allclose(ret, exp_ret, atol=1e-5, rtol=1e-5), f"ret got {ret.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(21)
    rewards = torch.randn(8)
    values = torch.randn(8)
    for gamma, lam in [(0.99, 0.95), (0.95, 0.8), (1.0, 1.0)]:
        a, r = solve(rewards, values, gamma, lam)
        oa, orr = _oracle(rewards, values, gamma, lam)
        assert torch.allclose(a, oa, atol=1e-5, rtol=1e-5), (gamma, lam)
        assert torch.allclose(r, orr, atol=1e-5, rtol=1e-5), (gamma, lam)


def check_lambda_zero_is_one_step_td(solve):
    # lam=0 时 A_t = delta_t（单步 TD）
    adv, _ = solve(REWARDS, VALUES, 0.99, 0.0)
    delta = []
    for t in range(len(REWARDS)):
        nxt = VALUES[t + 1].item() if t + 1 < len(REWARDS) else 0.0
        delta.append(REWARDS[t].item() + 0.99 * nxt - VALUES[t].item())
    assert torch.allclose(adv, torch.tensor(delta, dtype=torch.float32), atol=1e-5, rtol=1e-5)


def check_returns_minus_adv_is_values(solve):
    adv, ret = solve(REWARDS, VALUES, 0.9, 0.7)
    assert torch.allclose(ret - adv, VALUES, atol=1e-6, rtol=1e-6), "returns-advantages 应恒等于 values"


def check_single_step_boundary(solve):
    adv, ret = solve(torch.tensor([2.0]), torch.tensor([1.0]), 0.99, 0.95)
    # T=1: delta=2+0-1=1, A=1, R=1+1=2
    assert torch.allclose(adv, torch.tensor([1.0]), atol=1e-6) and torch.allclose(ret, torch.tensor([2.0]), atol=1e-6)
