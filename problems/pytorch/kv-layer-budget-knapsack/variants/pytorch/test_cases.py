import itertools
import torch


def _oracle(gain, budget):
    """暴力枚举全部分配向量（层数少、M 小），按「收益大、字典序小」选最优。"""
    layers, max_keep = gain.shape
    best_val, best_alloc = None, None
    for alloc in itertools.product(range(max_keep), repeat=layers):
        if sum(alloc) > budget:
            continue
        val = sum(float(gain[l][alloc[l]]) for l in range(layers))
        if best_val is None or val > best_val + 1e-12 or (
                abs(val - best_val) <= 1e-12 and alloc < best_alloc):
            best_val, best_alloc = val, alloc
    return torch.tensor(best_alloc, dtype=torch.long), torch.tensor(best_val, dtype=torch.float32)


def check_example(solve):
    torch.manual_seed(0)
    gain = torch.tensor([[0.0, 5.0, 8.0, 9.0],
                         [0.0, 4.0, 7.0, 9.0]])
    alloc, total = solve(gain, 4)
    assert alloc.tolist() == [2, 2], f"got {alloc.tolist()}"
    assert abs(float(total) - 15.0) < 1e-6, f"got {float(total)}"


check_example.example = True


def check_budget_is_fully_used_when_profitable(solve):
    """收益单调时名额会用满：单层 gain=[0,1,2,3] 且 budget=2 → 分 2 个。"""
    torch.manual_seed(1)
    gain = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    alloc, total = solve(gain, 2)
    assert alloc.tolist() == [2] and abs(float(total) - 2.0) < 1e-6
    alloc3, total3 = solve(gain, 9)
    assert alloc3.tolist() == [3], f"预算超容量应封顶 M-1，got {alloc3.tolist()}"
    assert abs(float(total3) - 3.0) < 1e-6


def check_tie_break_lexicographic(solve):
    """收益并列取字典序最小：三层完全线性对称、预算 2 → [0,0,2] 而非 [2,0,0]。"""
    torch.manual_seed(2)
    gain = torch.tensor([[0.0, 1.0, 2.0],
                         [0.0, 1.0, 2.0],
                         [0.0, 1.0, 2.0]])
    alloc, total = solve(gain, 2)
    assert alloc.tolist() == [0, 0, 2], f"got {alloc.tolist()}"
    assert abs(float(total) - 2.0) < 1e-6


def check_allocation_respects_marginals(solve):
    """名额按边际收益分配：一个高开低走、一个低开高走，预算 2 → 各拿 1。"""
    torch.manual_seed(3)
    gain = torch.tensor([[0.0, 10.0, 10.5],
                         [0.0, 9.0, 17.0]])
    alloc, total = solve(gain, 2)
    assert alloc.tolist() == [1, 1], f"got {alloc.tolist()}"
    assert abs(float(total) - 19.0) < 1e-6
    alloc3, total3 = solve(gain, 3)
    assert alloc3.tolist() == [1, 2], f"第三个名额应给边际更高的层 1，got {alloc3.tolist()}"
    assert abs(float(total3) - 27.0) < 1e-6


def check_matches_oracle_and_errors(solve):
    """随机用例与暴力枚举对拍（含并列），并校验非法参数。"""
    torch.manual_seed(21)
    for layers, max_keep, budget in ((3, 4, 5), (2, 3, 0), (4, 3, 4)):
        gain = torch.randint(0, 4, (layers, max_keep)).float()
        gain[:, 0] = 0.0
        exp_a, exp_t = _oracle(gain, budget)
        a, t = solve(gain, budget)
        assert torch.equal(a, exp_a), f"{(layers, max_keep, budget)} 分配不符：{a.tolist()} vs {exp_a.tolist()}"
        assert abs(float(t) - float(exp_t)) < 1e-6, f"收益不符 {float(t)} vs {float(exp_t)}"
    for bad in ((torch.zeros(2, 3), -1), (torch.tensor([[0.0, -1.0]]), 1)):
        try:
            solve(*bad)
            raise AssertionError(f"{bad[1]} 应抛 ValueError")
        except ValueError:
            pass
