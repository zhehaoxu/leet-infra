import numpy as np
import torch


def _oracle(alphas, cost_ratio):
    """numpy 双重循环独立算 e(γ) 与 s(γ)（结构异于参考解的前缀增量累加）。"""
    a = alphas.numpy().astype(np.float64)
    g_max = a.size
    expected = np.empty(g_max + 1, dtype=np.float64)
    for g in range(g_max + 1):
        total = 1.0
        for i in range(1, g + 1):
            prod = 1.0
            for j in range(i):
                prod *= a[j]
            total += prod
        expected[g] = total
    gammas = np.arange(g_max + 1, dtype=np.float64)
    speed = expected / (1.0 + gammas * float(cost_ratio))
    return expected, speed, int(np.argmax(speed))


def check_example(solve):
    torch.manual_seed(0)
    expected, speed, best, best_speed = solve(torch.tensor([0.8, 0.6, 0.5]), 0.1)
    assert torch.allclose(expected, torch.tensor([1.0, 1.8, 2.28, 2.52]), atol=1e-6), (
        f"got {expected.tolist()}")
    want = torch.tensor([1.0, 1.8 / 1.1, 2.28 / 1.2, 2.52 / 1.3])
    assert torch.allclose(speed, want, atol=1e-6), f"got {speed.tolist()}"
    assert int(best) == 3, f"最优 γ 应为 3，got {int(best)}"
    assert abs(float(best_speed) - float(want[3])) < 1e-6, f"got {float(best_speed)}"


check_example.example = True


def check_uniform_matches_geometric_closed_form(solve):
    """接受率恒为 α 时期望吐词退化为 (1-α^{γ+1})/(1-α)。"""
    torch.manual_seed(1)
    alpha, g_max = 0.7, 4
    alphas = torch.full((g_max,), alpha)
    expected, speed, best, _ = solve(alphas, 0.15)
    want = torch.tensor([(1 - alpha ** (g + 1)) / (1 - alpha) for g in range(g_max + 1)])
    assert torch.allclose(expected, want, atol=1e-6), f"{expected.tolist()} vs {want.tolist()}"
    assert torch.allclose(speed, want / (1 + 0.15 * torch.arange(g_max + 1)), atol=1e-6)
    exp_np, sp_np, best_np = _oracle(alphas, 0.15)
    assert int(best) == best_np, f"got {int(best)} vs {best_np}"


def check_degenerate_rates(solve):
    """α=0 全不接受 → 只剩 baseline；α=1 全接受 → 线性增长。"""
    torch.manual_seed(2)
    zero = torch.zeros(4)
    expected, speed, best, best_speed = solve(zero, 0.2)
    assert torch.allclose(expected, torch.ones(5)), f"got {expected.tolist()}"
    assert int(best) == 0 and abs(float(best_speed) - 1.0) < 1e-6, "α=0 时不该投机"
    assert bool((speed[1:] < speed[:-1]).all()), "α=0 时加速比应随 γ 单调下降"

    one = torch.ones(4)
    expected, speed, best, _ = solve(one, 0.5)
    assert torch.allclose(expected, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), (
        f"got {expected.tolist()}")
    assert int(best) == 4, f"draft 便宜时 γ 应该拉满，got {int(best)}"
    assert torch.allclose(speed, (torch.arange(1, 6) / (1 + 0.5 * torch.arange(5))), atol=1e-6)


def check_ties_and_cost_ratio(solve):
    """draft 与 target 同价（ratio=1）且 α=1 → 所有 γ 加速比都是 1，并列取最小 γ。"""
    torch.manual_seed(3)
    one = torch.ones(3)
    expected, speed, best, best_speed = solve(one, 1.0)
    assert torch.allclose(expected, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(speed, torch.ones(4), atol=1e-6), f"got {speed.tolist()}"
    assert int(best) == 0, f"并列应取最小 γ，got {int(best)}"
    assert abs(float(best_speed) - 1.0) < 1e-6

    expected, speed, best, _ = solve(torch.full((5,), 0.8), 0.0)
    assert torch.allclose(speed, expected, atol=1e-6), "ratio=0 时加速比即期望吐词数"
    assert abs(float(speed[0]) - 1.0) < 1e-6, "γ=0 的加速比必须是 1（target 自身为基准）"


def check_matches_oracle_and_errors(solve):
    """随机接受率序列对拍 + 内点最优 + 非法输入。"""
    torch.manual_seed(21)
    for g_max, alpha_max, ratio in ((3, 0.9, 0.3), (6, 1.0, 0.1), (2, 0.5, 2.0)):
        alphas = torch.rand(g_max) * alpha_max
        expected, speed, best, best_speed = solve(alphas, ratio)
        exp_np, sp_np, best_np = _oracle(alphas, ratio)
        assert torch.allclose(expected, torch.tensor(exp_np, dtype=torch.float32), atol=1e-5)
        assert torch.allclose(speed, torch.tensor(sp_np, dtype=torch.float32), atol=1e-5)
        assert int(best) == best_np, f"got {int(best)} vs {best_np}"
        assert abs(float(best_speed) - float(sp_np[best_np])) < 1e-5

    # 内点最优：α=0.8、ratio=0.2 时 γ=4 最好，加更多草稿位反而变慢
    expected, speed, best, _ = solve(torch.full((8,), 0.8), 0.2)
    assert int(best) == 4, f"应出现内点最优 γ=4，got {int(best)}"
    assert float(speed[4]) > float(speed[6]), "更长的 γ 应被时延吃掉收益"

    for bad in ((torch.tensor([]), 0.1), (torch.tensor([1.2]), 0.1), (torch.tensor([0.5]), -0.1)):
        try:
            solve(*bad)
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
