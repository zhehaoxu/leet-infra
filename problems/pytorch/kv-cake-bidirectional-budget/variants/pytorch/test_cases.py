import math
import torch


def _oracle(scores, budget, edges):
    """逐层 python 重算：mass/floor/余量/传递/排序全手写，结构异于参考解的向量化路径。"""
    layers, length = scores.shape
    rows = [[float(x) for x in scores[l]] for l in range(layers)]
    mass = [sum(r) for r in rows]
    total = sum(mass)
    raw = [(budget * m / total if total > 0 else 0.0) for m in mass]
    alloc = [min(int(math.floor(r)), length) for r in raw]
    rem = budget - sum(alloc)
    if rem > 0:
        order = sorted(range(layers), key=lambda l: (-(raw[l] - math.floor(raw[l])), l))
        while rem > 0:
            moved = False
            for l in order:
                if rem == 0:
                    break
                if alloc[l] < length:
                    alloc[l] += 1
                    rem -= 1
                    moved = True
            if not moved:
                break

    work = [r[:] for r in rows]
    for src, dst in edges:
        work[dst] = [a + b for a, b in zip(work[dst], work[src])]

    keep = [[False] * length for _ in range(layers)]
    for l in range(layers):
        cand = sorted(range(length), key=lambda j: (-work[l][j], j))
        for j in cand[:alloc[l]]:
            keep[l][j] = True
    return torch.tensor(alloc, dtype=torch.long), torch.tensor(keep, dtype=torch.bool)


def check_example(solve):
    torch.manual_seed(0)
    scores = torch.tensor([
        [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 9.0, 8.0, 7.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ])
    alloc, keep = solve(scores, 9, [(0, 1)])
    assert alloc.tolist() == [3, 3, 3], f"got {alloc.tolist()}"
    assert keep[0].tolist() == [True, True, True, False, False, False], f"got {keep[0].tolist()}"
    assert keep[1].tolist() == [True, True, False, True, False, False], f"got {keep[1].tolist()}"
    assert keep[2].tolist() == [False, False, False, True, True, True], f"got {keep[2].tolist()}"


check_example.example = True


def check_allocation_is_proportional(solve):
    """预算按层质量占比分配：质量 4:2:1、总预算 7 → 分配 4:2:1。"""
    torch.manual_seed(1)
    scores = torch.zeros(3, 8)
    scores[0] = 0.5
    scores[1] = 0.25
    scores[2] = 0.125
    alloc, _ = solve(scores, 7, [])
    assert alloc.tolist() == [4, 2, 1], f"got {alloc.tolist()}"
    assert int(alloc.sum()) == 7


def check_remainder_by_largest_fraction(solve):
    """余量按小数部分降序分（并列取小层号），单层不超过 L。"""
    torch.manual_seed(2)
    scores = torch.ones(3, 8)
    alloc, _ = solve(scores, 8, [])
    assert alloc.tolist() == [3, 3, 2], f"三层等质量、预算 8 应得 [3,3,2]，got {alloc.tolist()}"
    tight, _ = solve(scores, 24, [])
    assert tight.tolist() == [8, 8, 8], f"预算超容量应封顶在 L，got {tight.tolist()}"


def check_cascade_changes_selection(solve):
    """跨层传递：无边时上层选自己的 top-n，有边时下层证据把其 token 顶上来；方向决定结果。"""
    torch.manual_seed(3)
    scores = torch.tensor([
        [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 9.0, 8.0, 7.0],
    ])
    _, plain = solve(scores, 6, [])
    assert plain[1].tolist() == [False, False, False, True, True, True], f"got {plain[1].tolist()}"
    _, down = solve(scores, 6, [(0, 1)])
    assert down[1].tolist() == [True, True, False, True, False, False], f"got {down[1].tolist()}"
    _, up = solve(scores, 6, [(1, 0)])
    assert up[0].tolist() == [True, True, False, True, False, False], f"got {up[0].tolist()}"
    assert down[0].tolist() == plain[0].tolist(), "传递只影响 dst，src 层不应变"
    assert up[1].tolist() == plain[1].tolist(), "反向边不应影响 layer1"


def check_matches_oracle_and_errors(solve):
    """随机用例与逐层重算对拍（含传递链），并校验非法参数与零质量层。"""
    torch.manual_seed(21)
    scores = torch.rand(4, 7) ** 2
    for budget, edges in ((10, [(0, 1), (1, 2)]), (5, [(3, 0)]), (0, [])):
        exp_a, exp_k = _oracle(scores, budget, edges)
        a, k = solve(scores, budget, edges)
        assert torch.equal(a, exp_a), f"budget={budget},edges={edges} 分配不符"
        assert torch.equal(k, exp_k), f"budget={budget},edges={edges} 掩码不符"
    zero = scores.clone()
    zero[2] = 0.0
    a0, k0 = solve(zero, 7, [])
    assert int(a0[2]) == 0 and not bool(k0[2].any()), "零质量层应分到 0 且不保留"
    for bad in ((scores, -1, []), (scores, 3, [(0, 0)]), (scores, 3, [(0, 9)])):
        try:
            solve(*bad)
            raise AssertionError(f"{bad[1:]} 应抛 ValueError")
        except ValueError:
            pass
