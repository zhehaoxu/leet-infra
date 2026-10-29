import math

import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _oracle(hidden, routers, blocks, caps):
    """逐层 python/numpy 重算：sorted 选容量、逐行算因果 softmax，不复用向量化代码。"""
    x = np.array(hidden, dtype=np.float64)
    routers = np.array(routers, dtype=np.float64)
    blocks = np.array(blocks, dtype=np.float64)
    tokens, dim = x.shape
    layers = routers.shape[0]
    scale = 1.0 / math.sqrt(dim)
    masks = np.zeros((layers, tokens), dtype=bool)
    all_scores = np.zeros((layers, tokens), dtype=np.float64)
    for layer in range(layers):
        scores = x @ routers[layer]
        all_scores[layer] = scores
        capacity = min(int(caps[layer]), tokens)
        if capacity == 0:
            continue
        chosen = sorted(sorted(range(tokens), key=lambda i: (-scores[i], i))[:capacity])
        masks[layer, chosen] = True
        subset = x[chosen]
        projected = subset @ blocks[layer]
        size = len(chosen)
        for row in range(size):
            logits = (projected[: row + 1] @ projected[row]) * scale
            shifted = np.exp(logits - logits.max())
            probs = shifted / shifted.sum()
            mixed = probs @ projected[: row + 1]
            x[chosen[row]] = subset[row] + scores[chosen[row]] * mixed
    return x, masks, all_scores


def _rand(seed, *shape, low=-2.0, high=2.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * (high - low) + low


def check_example(solve):
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    routers = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    blocks = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    output, route_mask, router_scores = solve(hidden, routers, blocks, [2, 1])
    expect_out = torch.tensor(
        [[2.0, 0.0], [0.0, 1.0], [5.33952332, 4.45786524]], dtype=torch.float64
    )
    expect_mask = torch.tensor([[True, False, True], [False, False, True]])
    expect_scores = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.66976154]], dtype=torch.float64
    )
    assert output.shape == (3, 2) and route_mask.shape == (2, 3)
    assert router_scores.shape == (2, 3)
    assert output.dtype == torch.float32 and router_scores.dtype == torch.float32
    assert route_mask.dtype == torch.bool
    assert _close(output, expect_out, 1e-6), output
    assert torch.equal(route_mask, expect_mask), route_mask
    assert _close(router_scores, expect_scores, 1e-6), router_scores


check_example.example = True


def check_capacity_and_tie_break(solve):
    hidden = torch.tensor([[0.5], [0.9], [0.5], [0.9], [0.1]], dtype=torch.float64)
    routers = torch.ones(1, 1, dtype=torch.float64)
    blocks = torch.ones(1, 1, 1, dtype=torch.float64)
    _, route_mask, _ = solve(hidden, routers, blocks, [3])
    # 分数 0.9 并列时下标小的先选，0.5 并列时同理 → 选中 {1, 3, 0}
    assert torch.equal(
        route_mask, torch.tensor([[True, True, False, True, False]])
    ), route_mask
    # 容量 0：一个都不路由，输出必须逐位等于输入
    zero_out, zero_mask, _ = solve(hidden, routers, blocks, [0])
    assert not bool(zero_mask.any())
    assert torch.equal(zero_out, hidden.to(torch.float32))
    # 容量超过 T：全部路由
    full_out, full_mask, _ = solve(hidden, routers, blocks, [99])
    assert bool(full_mask.all())
    assert full_out.shape == hidden.shape


def check_single_token_block_scaling(solve):
    # 容量 1 时子序列只有一行 → 因果 softmax 恒为 1，block 输出就是 x·W
    hidden = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    routers = torch.ones(1, 1, dtype=torch.float64)
    blocks = torch.ones(1, 1, 1, dtype=torch.float64)
    output, route_mask, _ = solve(hidden, routers, blocks, [1])
    # 选中第 1 个 token：3 + r·(3·1) = 3 + 3·3 = 12；第 0 个保持 1
    assert torch.equal(route_mask, torch.tensor([[False, True]]))
    assert _close(output, torch.tensor([[1.0], [12.0]], dtype=torch.float64), 1e-6), output


def check_unselected_untouched(solve):
    hidden = _rand(61, 6, 3).to(torch.float32)
    routers = _rand(62, 3, 3).to(torch.float32)
    blocks = _rand(63, 3, 3, 3).to(torch.float32)
    output, route_mask, _ = solve(hidden, routers, blocks, [2, 2, 2])
    never = ~route_mask.any(dim=0)
    assert bool(never.sum() > 0), "本检查需要至少一个从未被路由的 token"
    assert torch.equal(output[never], hidden[never]), "未被路由的 token 必须逐位不变"
    assert route_mask.shape == (3, 6) and bool(route_mask.sum(dim=1).le(2).all())
    # 顺便确认被路由的行确实变了（否则上面的断言没有区分力）
    assert bool((output[~never] != hidden[~never]).any())


def check_block_scope_selected_only(solve):
    routers = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    blocks = torch.eye(2, dtype=torch.float64).reshape(1, 2, 2)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 1.0]], dtype=torch.float64)
    output, route_mask, _ = solve(hidden, routers, blocks, [2])
    assert torch.equal(route_mask, torch.tensor([[True, False, True]])), route_mask
    # 改动未被选中的第 1 个 token：第 0、2 个 token 的输出必须完全不受影响
    perturbed = hidden.clone()
    perturbed[1] = torch.tensor([0.0, 7.0], dtype=torch.float64)
    output2, route_mask2, _ = solve(perturbed, routers, blocks, [2])
    assert torch.equal(route_mask, route_mask2)
    assert torch.equal(output[[0, 2]], output2[[0, 2]]), (output, output2)
    assert torch.equal(output[1], hidden[1].to(torch.float32))
    # 选中的集合只含 {0, 2}，所以第 2 个 token 的因果前缀里必须有第 0 个
    assert _close(
        output[2], torch.tensor([11.95778751, 3.97889400], dtype=torch.float64), 1e-5
    ), output


def check_gather_preserves_order(solve):
    # 分数降序是 [1, 2]（下标 1 在前），容量 2 全选中；但子序列必须按原始下标排成 [0, 1]
    hidden = torch.tensor([[1.0], [2.0], [0.5]], dtype=torch.float64)
    routers = torch.ones(1, 1, dtype=torch.float64)
    blocks = torch.ones(1, 1, 1, dtype=torch.float64)
    output, route_mask, _ = solve(hidden, routers, blocks, [2])
    assert torch.equal(route_mask, torch.tensor([[True, True, False]]))
    # H=[[1],[2]]，A=[[1,2],[2,4]]，第二行 logits=[2,4] → P=[1, e²]/(1+e²)
    p = 1.0 / (1.0 + math.exp(2.0))
    expected_token1 = 2.0 + 2.0 * (p * 1.0 + (1.0 - p) * 2.0)
    assert _close(output[1], torch.tensor([expected_token1], dtype=torch.float64), 1e-6), output
    # 若按分数序 gather，第二个 token 就成了子序列首行，输出会恰好是 6.0
    assert abs(expected_token1 - 6.0) > 0.1, expected_token1


def check_router_uses_current_state(solve):
    hidden = torch.tensor([[0.0, 3.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
    routers = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    blocks = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    output, route_mask, router_scores = solve(hidden, routers, blocks, [2, 1])
    assert torch.equal(
        route_mask, torch.tensor([[True, False, True], [False, False, True]])
    ), route_mask
    # 第 1 层分数必须来自第 0 层更新后的状态；用初始 hidden 会得到 [3, 2, 1] → 选下标 0
    assert _close(
        router_scores[1], torch.tensor([3.0, 2.0, 3.33952308], dtype=torch.float64), 1e-6
    ), router_scores
    stale = hidden @ routers[1]
    assert int(torch.argmax(stale)) == 0 and int(torch.argmax(router_scores[1])) == 2
    assert _close(
        output[2], torch.tensor([5.77260065, 14.49193764], dtype=torch.float64), 1e-5
    ), output


def check_output_contract(solve):
    hidden = _rand(71, 4, 3)
    routers = _rand(72, 1, 3)
    blocks = _rand(73, 1, 3, 3)
    output, route_mask, router_scores = solve(hidden, routers, blocks, [3])
    assert output.shape == (4, 3) and output.dtype == torch.float32
    assert route_mask.shape == (1, 4) and route_mask.dtype == torch.bool
    assert router_scores.shape == (1, 4) and router_scores.dtype == torch.float32
    assert int(route_mask.sum()) == 3
    assert _close(router_scores[0], hidden @ routers[0], 1e-6), router_scores
    # tuple 容量与超容量裁剪都合法
    output2, mask2, _ = solve(hidden, routers, blocks, (9,))
    assert bool(mask2.all())


def check_against_oracle(solve):
    cases = [
        (_rand(81, 1, 2), _rand(82, 1, 2), _rand(83, 1, 2, 2), [1]),
        (_rand(84, 5, 3), _rand(85, 2, 3), _rand(86, 2, 3, 3), [0, 5]),
        (_rand(87, 7, 4), _rand(88, 3, 4), _rand(89, 3, 4, 4), [2, 3, 7]),
        (_rand(90, 4, 1), _rand(91, 1, 1), _rand(92, 1, 1, 1), [0]),
        (_rand(93, 6, 2), _rand(94, 2, 2), _rand(95, 2, 2, 2), [1, 4]),
    ]
    for hidden, routers, blocks, caps in cases:
        output, route_mask, router_scores = solve(hidden, routers, blocks, caps)
        ref_x, ref_mask, ref_scores = _oracle(
            hidden.numpy(), routers.numpy(), blocks.numpy(), caps
        )
        assert _close(output, ref_x, 1e-6), (caps, output, ref_x)
        assert torch.equal(route_mask, torch.from_numpy(ref_mask)), (caps, route_mask)
        assert _close(router_scores, ref_scores, 1e-6), (caps, router_scores)
    # float32 输入同样要能跑通，且输出仍是 float32
    hidden32 = cases[2][0].to(torch.float32)
    out32, mask32, scores32 = solve(
        hidden32, cases[2][1].to(torch.float32), cases[2][2].to(torch.float32), [2, 3, 7]
    )
    assert out32.dtype == torch.float32 and scores32.dtype == torch.float32
    # 与"float32 输入升到 float64 后"的 oracle 比：路由顺序因此被完全固定
    ref32_x, ref32_mask, _ = _oracle(
        hidden32.double().numpy(),
        cases[2][1].to(torch.float32).double().numpy(),
        cases[2][2].to(torch.float32).double().numpy(),
        [2, 3, 7],
    )
    assert torch.equal(mask32, torch.from_numpy(ref32_mask)), mask32
    assert _close(out32, ref32_x, 1e-5), (out32, ref32_x)


def check_invalid_inputs(solve):
    hidden = torch.zeros(4, 2, dtype=torch.float64)
    routers = torch.zeros(1, 2, dtype=torch.float64)
    blocks = torch.zeros(1, 2, 2, dtype=torch.float64)
    bad_calls = [
        ("hidden-not-tensor", lambda: solve([[0.0, 1.0]], routers, blocks, [1])),
        ("hidden-int", lambda: solve(hidden.to(torch.int64), routers, blocks, [1])),
        ("hidden-1d", lambda: solve(hidden[0], routers, blocks, [1])),
        ("hidden-3d", lambda: solve(hidden.unsqueeze(0), routers, blocks, [1])),
        ("hidden-empty", lambda: solve(torch.zeros(0, 2, dtype=torch.float64), routers, blocks, [0])),
        (
            "hidden-zero-dim",
            lambda: solve(torch.zeros(4, 0, dtype=torch.float64), routers, blocks, [1]),
        ),
        (
            "hidden-nan",
            lambda: solve(
                torch.tensor([[0.0, float("nan")]], dtype=torch.float64),
                torch.zeros(1, 2, dtype=torch.float64),
                torch.zeros(1, 2, 2, dtype=torch.float64),
                [1],
            ),
        ),
        ("router-not-tensor", lambda: solve(hidden, [[0.0, 1.0]], blocks, [1])),
        ("router-wrong-width", lambda: solve(hidden, torch.zeros(1, 3, dtype=torch.float64), blocks, [1])),
        ("router-3d", lambda: solve(hidden, routers.unsqueeze(0), blocks, [1])),
        ("router-empty", lambda: solve(hidden, torch.zeros(0, 2, dtype=torch.float64), blocks, [])),
        ("block-wrong-shape", lambda: solve(hidden, routers, torch.zeros(1, 2, 3, dtype=torch.float64), [1])),
        ("block-layer-mismatch", lambda: solve(hidden, routers, blocks.repeat(2, 1, 1), [1])),
        ("block-not-tensor", lambda: solve(hidden, routers, [[[1.0, 0.0], [0.0, 1.0]]], [1])),
        ("caps-not-sequence", lambda: solve(hidden, routers, blocks, 1)),
        ("caps-string", lambda: solve(hidden, routers, blocks, "1")),
        ("caps-tensor", lambda: solve(hidden, routers, blocks, torch.tensor([1]))),
        ("caps-length", lambda: solve(hidden, routers, blocks, [1, 1])),
        ("caps-empty", lambda: solve(hidden, routers, blocks, [])),
        ("caps-float", lambda: solve(hidden, routers, blocks, [1.0])),
        ("caps-bool", lambda: solve(hidden, routers, blocks, [True])),
        ("caps-negative", lambda: solve(hidden, routers, blocks, [-1])),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
