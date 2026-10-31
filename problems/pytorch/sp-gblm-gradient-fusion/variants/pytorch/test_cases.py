import numpy as np
import torch


def _oracle(weight, activations, grad_samples, sparsity, scale=1.0,
            gradient_inv=False, prune_n=0, prune_m=0):
    """独立 numpy 重算 GBLM 融合显著性（与 solve 不同代码路径）。"""
    w = weight.detach().cpu().numpy().astype(np.float64)
    a = activations.detach().cpu().numpy().astype(np.float64)
    gs = grad_samples.detach().cpu().numpy().astype(np.float64)

    col_ms = (a ** 2).mean(axis=0)                      # (in,)
    G = np.sqrt(((gs * scale) ** 2).sum(axis=0))        # (out, in) L2 聚合
    base = np.abs(w) * np.sqrt(col_ms)[None, :]
    if not gradient_inv:
        metric = base + np.abs(w) * G
    else:
        metric = base / (G + 1e-8)

    out_f, in_f = w.shape
    mask = np.zeros_like(metric, dtype=bool)
    if prune_n > 0:
        for start in range(0, in_f, prune_m):
            block = metric[:, start:start + prune_m]
            order = np.argsort(block, axis=-1, kind="stable")[:, :prune_n]
            np.put_along_axis(mask, order + start, True, axis=1)
    else:
        k = int(in_f * float(sparsity))
        order = np.argsort(metric, axis=-1, kind="stable")[:, :k]
        np.put_along_axis(mask, order, True, axis=1)

    pruned = np.where(mask, 0.0, w)
    return {
        "metric": torch.tensor(metric, dtype=weight.dtype),
        "grad_l2": torch.tensor(G, dtype=weight.dtype),
        "mask": torch.tensor(mask),
        "pruned_weight": torch.tensor(pruned, dtype=weight.dtype),
        "sparsity": float(mask.sum()) / float(mask.size),
    }


def check_example_hand_computed(solve):
    """手算用例：scale=1 融合显著性 + 逐行 0.5 剪枝。"""
    W = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    A = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    # 两个样本的梯度；G row0 = [0.1√2, 0, 0, 0]，row1 = [0, 0, 0.3, 0.2]
    G1 = torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2]])
    G2 = torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0]])
    grads = torch.stack([G1, G2])  # (2, 2, 4)
    got = solve(W, A, grads, 0.5, scale=1.0)
    exp = _oracle(W, A, grads, 0.5, scale=1.0)
    # 手算（√s = [1, 1/√2, 1/√2, 0]）：
    # metric row0 = [1+0.141421, 1.414214, 2.121320, 0] → 剪 {3, 0}
    # metric row1 = [4, 2.121320, 2.014214, 0.2]        → 剪 {3, 2}
    assert torch.allclose(got["metric"], exp["metric"], atol=1e-9)
    assert torch.equal(got["mask"], torch.tensor(
        [[True, False, False, True], [False, False, True, True]]))
    assert torch.equal(got["pruned_weight"], torch.tensor(
        [[0.0, 2.0, 3.0, 0.0], [4.0, 3.0, 0.0, 0.0]]))
    assert abs(got["sparsity"] - 0.5) < 1e-12
    assert torch.allclose(got["grad_l2"], torch.tensor(
        [[0.1414213562373095, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.3, 0.2]]), atol=1e-9)


def check_scale_changes_mask(solve):
    """scale 放大梯度项 → 两项权重比改变 → 掩码随之改变（融合是真考点）。"""
    W = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    A = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    grads = torch.stack([
        torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2]]),
        torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0]]),
    ])
    got10 = solve(W, A, grads, 0.5, scale=10.0)
    exp10 = _oracle(W, A, grads, 0.5, scale=10.0)
    # scale=10 时 row0 metric = [2.428427, 1.414214, 2.121320, 0] → 剪 {3, 1}
    assert torch.allclose(got10["metric"], exp10["metric"], atol=1e-9)
    assert torch.equal(got10["mask"], torch.tensor(
        [[False, True, False, True], [False, True, False, True]]))
    assert torch.equal(got10["pruned_weight"], torch.tensor(
        [[1.0, 0.0, 3.0, 0.0], [4.0, 0.0, 2.0, 0.0]]))


def check_gradient_inv(solve):
    """gradient_inv：metric = |W|·√s / (G + 1e-8)；梯度大处反而被保护/剪除反转。"""
    W = torch.tensor([[10.0, 8.0]])
    A = torch.tensor([[1.0, 1.0]])
    grads = torch.tensor([[[1.0, 20.0]]])  # G = [1, 20]
    got = solve(W, A, grads, 0.5, scale=1.0, gradient_inv=True)
    # 非 inv：[11, 28] → 剪 0；inv：[10, 0.4] → 剪 1（反转）
    assert torch.equal(got["mask"], torch.tensor([[False, True]]))
    assert torch.equal(got["pruned_weight"], torch.tensor([[10.0, 0.0]]))
    # 同输入非 inv 剪另一个
    got_normal = solve(W, A, grads, 0.5, scale=1.0, gradient_inv=False)
    assert torch.equal(got_normal["mask"], torch.tensor([[True, False]]))
    # 零梯度位：10/(0+1e-8) = 1e9 巨值 → 不剪；G=20 处 8/20 = 0.4 → 剪
    grads0 = torch.tensor([[[0.0, 20.0]]])
    got0 = solve(W, A, grads0, 0.5, scale=1.0, gradient_inv=True)
    assert torch.equal(got0["mask"], torch.tensor([[False, True]]))
    assert torch.allclose(got0["metric"], torch.tensor(
        [[10.0 / 1e-8, 8.0 / 20.00000001]]), rtol=1e-6)


def check_structured_2_4_tiebreak(solve):
    """2:4 块内窗口 + metric 并列值。"""
    W = torch.arange(1.0, 9.0).reshape(1, 8)
    A = torch.ones(1, 8)
    grads = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    got = solve(W, A, grads, 0.5, prune_n=2, prune_m=4)
    # metric = [2,2,3,4,5,6,7,8]；块 [0:4] 剪 0、1（并列 2 按 stable 索引序），块 [4:8] 剪 4、5
    exp = _oracle(W, A, grads, 0.5, prune_n=2, prune_m=4)
    assert torch.allclose(got["metric"], exp["metric"], atol=1e-9)
    assert torch.equal(got["mask"], torch.tensor(
        [[True, True, False, False, True, True, False, False]]))


def check_multi_sample_l2_aggregation(solve):
    """多样本梯度 L2 聚合：sqrt(Σ(g·scale)²)，与逐样本取 max/L1 不同。"""
    torch.manual_seed(7)
    W = torch.randn(5, 6)
    A = torch.randn(9, 6)
    grads = torch.randn(4, 5, 6)
    for inv in (False, True):
        got = solve(W, A, grads, 0.5, scale=3.0, gradient_inv=inv)
        exp = _oracle(W, A, grads, 0.5, scale=3.0, gradient_inv=inv)
        assert torch.allclose(got["metric"], exp["metric"], rtol=1e-9, atol=1e-12)
        assert torch.equal(got["mask"], exp["mask"])
        assert torch.allclose(got["pruned_weight"], exp["pruned_weight"],
                              rtol=1e-9, atol=0.0)


def check_random_against_oracle(solve):
    """随机形状对拍（非结构化与 2:4、双 inv 开关）。"""
    torch.manual_seed(11)
    for _ in range(6):
        out_f, in_f = int(torch.randint(2, 9, (1,))), int(torch.randint(2, 12, (1,)))
        W = torch.randn(out_f, in_f)
        A = torch.randn(int(torch.randint(1, 6, (1,))), in_f)
        grads = torch.randn(int(torch.randint(1, 4, (1,))), out_f, in_f)
        inv = bool(torch.randint(0, 2, (1,)))
        if in_f % 2 == 0 and in_f >= 4:
            got = solve(W, A, grads, 0.5, scale=2.0, gradient_inv=inv,
                        prune_n=1, prune_m=2)
            exp = _oracle(W, A, grads, 0.5, scale=2.0, gradient_inv=inv,
                          prune_n=1, prune_m=2)
        else:
            got = solve(W, A, grads, 0.25, scale=2.0, gradient_inv=inv)
            exp = _oracle(W, A, grads, 0.25, scale=2.0, gradient_inv=inv)
        assert torch.allclose(got["metric"], exp["metric"], rtol=1e-9, atol=1e-12)
        assert torch.equal(got["mask"], exp["mask"])
        assert torch.allclose(got["pruned_weight"], exp["pruned_weight"],
                              rtol=1e-9, atol=0.0)


def check_validation_errors(solve):
    """非法输入全部抛 ValueError。"""
    W = torch.randn(3, 4)
    A = torch.randn(2, 4)
    G = torch.randn(1, 3, 4)
    bad = [
        (lambda: solve(torch.randn(3, 4, 1), A, G, 0.5), "weight 三维"),
        (lambda: solve(W, torch.randn(2, 3), G, 0.5), "activations 列不匹配"),
        (lambda: solve(W.to(torch.float32), A.to(torch.float64), G, 0.5), "dtype 不一致"),
        (lambda: solve(W, A, torch.randn(1, 3, 5), 0.5), "grad 形状不匹配"),
        (lambda: solve(W, A, torch.randn(0, 3, 4), 0.5), "grad 零样本"),
        (lambda: solve(W, A, G, 1.5), "sparsity 越界"),
        (lambda: solve(W, A, G, 0.5, scale=0.0), "scale 非正"),
        (lambda: solve(W, A, G, 0.5, gradient_inv="yes"), "gradient_inv 非布尔"),
        (lambda: solve(W, A, G, 0.25, prune_n=2, prune_m=4), "N:M sparsity≠0.5"),
        (lambda: solve(W, A, G, 0.5, prune_n=4, prune_m=4), "N:M n>=m"),
        (lambda: solve(W, A, G, 0.5, prune_n=1, prune_m=3), "in%m≠0"),
    ]
    for fn, _ in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{bad[[f for f, _ in bad].index(fn)][1]}")
