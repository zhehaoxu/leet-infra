import math

import torch


def _oracle(weight, calib, sparsity):
    w = weight.tolist()
    c = calib.tolist()
    l, h = len(w), len(w[0])
    imp = []
    for i in range(l):
        for j in range(h):
            wn = math.sqrt(sum(v * v for row in w[i][j] for v in row))
            xn = math.sqrt(sum(v * v for row in c[i][j] for v in row))
            imp.append(wn * xn)
    k = int(round(sparsity * l * h))
    order = sorted(range(l * h), key=lambda t: (imp[t], t))
    drop = set(order[:k])
    pruned = []
    for i in range(l):
        layer = []
        for j in range(h):
            if i * h + j in drop:
                layer.append([[0.0] * len(w[i][j][0]) for _ in range(len(w[i][j]))])
            else:
                layer.append([row[:] for row in w[i][j]])
        pruned.append(layer)
    return pruned, imp


W = torch.tensor([[[[1.0, 2.0], [2.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]]])
C = torch.tensor([[[[3.0, 4.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]])


def check_example(solve):
    pruned, imp = solve(W, C, 0.5)
    assert torch.allclose(
        imp, torch.tensor([[15.81138801574707, 1.0]]), atol=1e-5, rtol=1e-6
    ), f"got {imp.tolist()}"
    exp = torch.tensor([[[[1.0, 2.0], [2.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    assert torch.allclose(pruned, exp, atol=1e-6), f"got {pruned.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for l, h, dh, d, n in [(2, 4, 4, 8, 5), (3, 2, 2, 3, 4), (1, 8, 8, 16, 3)]:
        w = torch.randn(l, h, dh, d)
        c = torch.randn(l, h, n, dh)
        for s in (0.25, 0.5, 0.75):
            pruned, imp = solve(w, c, s)
            exp_pruned, exp_imp = _oracle(w, c, s)
            assert torch.allclose(imp.flatten(), torch.tensor(exp_imp), atol=1e-5, rtol=1e-5)
            assert torch.allclose(pruned, torch.tensor(exp_pruned), atol=1e-6), f"l={l} h={h} s={s}"


def check_pruned_head_count(solve):
    torch.manual_seed(3)
    w = torch.randn(2, 4, 2, 2)
    c = torch.randn(2, 4, 3, 2)
    pruned, _ = solve(w, c, 0.5)
    zero_heads = [
        (i, j)
        for i in range(2)
        for j in range(4)
        if bool((pruned[i, j] == 0).all())
    ]
    assert len(zero_heads) == 4, zero_heads


def check_sparsity_zero_keeps_all(solve):
    torch.manual_seed(4)
    w = torch.randn(2, 3, 2, 2)
    c = torch.randn(2, 3, 2, 2)
    pruned, imp = solve(w, c, 0.0)
    assert torch.equal(pruned, w), "sparsity=0 不应改动任何头"
    assert torch.all(imp >= 0)


def check_sparsity_one_zeros_all(solve):
    torch.manual_seed(5)
    w = torch.randn(1, 4, 3, 2)
    c = torch.randn(1, 4, 2, 3)
    pruned, _ = solve(w, c, 1.0)
    assert torch.all(pruned == 0), "sparsity=1 应剪掉全部头"


def check_tie_break_by_index(solve):
    # 所有权重/激活相同 => importance 全相等 => 剪展平索引最小的 k 个
    w = torch.ones(2, 2, 2, 2)
    c = torch.ones(2, 2, 2, 2)
    pruned, imp = solve(w, c, 0.5)
    flat = pruned.reshape(2, 2, 2, 2)
    assert bool((flat[0, 0] == 0).all()) and bool((flat[0, 1] == 0).all())
    assert bool((flat[1, 0] != 0).any()) and bool((flat[1, 1] != 0).any())


def check_only_pruned_heads_change(solve):
    torch.manual_seed(6)
    w = torch.randn(1, 4, 2, 2)
    c = torch.randn(1, 4, 3, 2)
    pruned, _ = solve(w, c, 0.25)
    for j in range(4):
        if bool((pruned[0, j] == 0).all()):
            continue
        assert torch.equal(pruned[0, j], w[0, j]), f"头 {j} 未被剪却发生了变化"


def check_zero_calib_head_pruned_first(solve):
    # 第 0 个头权重范数很大但校准激活全零 => importance=0，必被剪
    w = torch.zeros(1, 2, 2, 2)
    w[0, 0] = 100.0
    w[0, 1] = 0.01
    c = torch.zeros(1, 2, 2, 2)
    c[0, 1] = 1.0
    pruned, imp = solve(w, c, 0.5)
    assert float(imp[0, 0]) == 0.0, imp.tolist()
    assert bool((pruned[0, 0] == 0).all())
    assert bool((pruned[0, 1] != 0).any())
