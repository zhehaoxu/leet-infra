import random

import torch


def _oracle(prompt_len, start, out_tokens, block_size, price_out, price_block):
    """独立 oracle：逐 (请求, 步) 枚举累加，不用任何闭式公式。"""
    n = len(prompt_len)
    residency = [0] * n
    timeline = {}
    for r in range(n):
        for k in range(out_tokens[r]):
            blocks = -(-(prompt_len[r] + k) // block_size)
            residency[r] += blocks
            t = start[r] + k
            timeline[t] = timeline.get(t, 0) + blocks
    charge = [price_out * out_tokens[r] + price_block * residency[r] for r in range(n)]
    return residency, charge, max(timeline.values()) if timeline else 0, sum(charge)


def check_example(solve):
    torch.manual_seed(0)
    residency, charge, peak, total = solve(
        torch.tensor([4, 10]),
        torch.tensor([0, 1]),
        torch.tensor([3, 2]),
        4,
        0.5,
        0.1,
    )
    assert residency.tolist() == [5, 6], f"块·步积分应为 [5, 6]，got {residency.tolist()}"
    want = torch.tensor([2.0, 1.6])
    assert torch.allclose(charge, want, atol=1e-6), f"账单应为 [2.0, 1.6]，got {charge.tolist()}"
    assert int(peak) == 5, f"峰值占块应为 5，got {int(peak)}"
    assert abs(float(total) - 3.6) < 1e-6, f"got {float(total)}"


check_example.example = True


def check_prompt_aligned_to_block(solve):
    """prompt 正好占满整块时，第一个生成 token 立刻要下一块。"""
    torch.manual_seed(1)
    residency, charge, peak, total = solve(
        torch.tensor([8]), torch.tensor([0]), torch.tensor([5]), 4, 1.0, 1.0
    )
    # 8 token -> 2 块；随后每多 4 个 token 加一块：2,3,3,3,3
    assert residency.tolist() == [14], f"应为 2+3+3+3+3=14，got {residency.tolist()}"
    assert abs(float(charge[0]) - (5.0 + 14.0)) < 1e-6, f"got {float(charge[0])}"
    assert int(peak) == 3, f"峰值应为 3 块，got {int(peak)}"


def check_non_overlapping_peak(solve):
    """首尾相接不重叠的请求：峰值只是单个请求的最大占块数。"""
    torch.manual_seed(2)
    residency, charge, peak, total = solve(
        torch.tensor([4, 4, 4]),
        torch.tensor([0, 3, 6]),
        torch.tensor([3, 3, 3]),
        4,
        0.0,
        1.0,
    )
    assert residency.tolist() == [5, 5, 5], f"got {residency.tolist()}"
    assert int(peak) == 2, f"任一时刻只有一个请求在跑，峰值 2 块，got {int(peak)}"
    assert abs(float(total) - 15.0) < 1e-6, f"got {float(total)}"

    # 完全重叠时峰值翻倍
    _, _, peak, _ = solve(
        torch.tensor([4, 4]), torch.tensor([0, 0]), torch.tensor([3, 3]), 4, 0.0, 1.0
    )
    assert int(peak) == 4, f"两个请求同时跑，峰值应为 4，got {int(peak)}"


def check_zero_output_and_random(solve):
    """out_tokens=0 不计任何驻留；随机负载对拍逐 (请求, 步) 枚举的 oracle。"""
    torch.manual_seed(3)
    residency, charge, peak, total = solve(
        torch.tensor([5, 1]), torch.tensor([0, 2]), torch.tensor([0, 0]), 4, 2.0, 0.7
    )
    assert residency.tolist() == [0, 0], f"没生成就不占块·步，got {residency.tolist()}"
    assert float(charge.abs().sum()) == 0.0 and int(peak) == 0 and float(total) == 0.0

    rng = random.Random(5)
    for _ in range(35):
        n = rng.randint(1, 4)
        bs = rng.randint(1, 6)
        ps = [rng.randint(1, 12) for _ in range(n)]
        st = [rng.randint(0, 6) for _ in range(n)]
        ot = [rng.randint(0, 6) for _ in range(n)]
        po = round(rng.uniform(0, 2), 2)
        pb = round(rng.uniform(0, 1), 2)
        got = solve(
            torch.tensor(ps), torch.tensor(st), torch.tensor(ot), bs, po, pb
        )
        want = _oracle(ps, st, ot, bs, po, pb)
        assert got[0].tolist() == want[0], f"residency 不一致：{got[0].tolist()} vs {want[0]}"
        assert int(got[2]) == want[2], f"peak 不一致：{int(got[2])} vs {want[2]}"
        for i in range(n):
            assert abs(float(got[1][i]) - want[1][i]) < 1e-4, "charge 不一致"
        assert abs(float(got[3]) - want[3]) < 1e-4, "total 不一致"


def check_invalid_inputs(solve):
    a = torch.tensor([4])
    for bad in (
        lambda: solve(a, a, a, 0, 1.0, 1.0),
        lambda: solve(a, a, a, 4, -1.0, 1.0),
        lambda: solve(a, a, a, 4, 1.0, -0.1),
        lambda: solve(torch.tensor([[4]]), a, a, 4, 1.0, 1.0),
        lambda: solve(a, torch.tensor([0, 1]), a, 4, 1.0, 1.0),
        lambda: solve(torch.tensor([0]), a, a, 4, 1.0, 1.0),
        lambda: solve(a, torch.tensor([-1]), a, 4, 1.0, 1.0),
        lambda: solve(a, a, torch.tensor([-2]), 4, 1.0, 1.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
