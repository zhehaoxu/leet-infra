"""独立 oracle：非 GS 分支直接按「反射逐个相乘」还原 H，GS 分支用投影式，均不经过紧凑 WY。"""

import random

import torch


def _oracle_h(u, apply_gs, reverse=False):
    u = u.to(torch.float64)
    n, r = u.shape
    eye_n = torch.eye(n, dtype=torch.float64)
    eye_r = torch.eye(r, dtype=torch.float64)
    if apply_gs:
        v = torch.linalg.qr(u, mode="reduced").Q
        h = eye_n - 2 * (v @ v.T)
        return h if not reverse else h.T
    v = torch.stack([u[:, k] / u[:, k].norm() for k in range(r)], dim=1)
    h = eye_n
    for k in range(r):
        vk = v[:, k : k + 1]
        h = h @ (eye_n - 2 * (vk @ vk.T))
    return h if not reverse else h.T


def check_example(solve):
    """两列不正交时出现交叉项（T 非对角）；正交化后退化为 diag(-1,-1,1)。"""
    u = torch.tensor([[1.0, 2.0], [0.0, 1.0], [0.0, 0.0]])
    w0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v, t, h, merged = solve(u, w0, False)
    assert v.shape == (3, 2) and t.shape == (2, 2) and h.shape == (3, 3), (v.shape, t.shape, h.shape)
    assert torch.allclose(h, torch.tensor([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6), h
    assert torch.allclose(t, torch.tensor([[2.0, -3.5777087], [0.0, 2.0]]), atol=1e-6), t
    assert torch.allclose(
        merged,
        torch.tensor([[-1.0, 2.0, 3.0], [-1.6, 6.2, 6.0]]),
        atol=1e-6,
    ), merged
    _, _, h_gs, merged_gs = solve(u, w0, True)
    assert torch.allclose(h_gs, torch.diag(torch.tensor([-1.0, -1.0, 1.0])), atol=1e-6), h_gs
    assert torch.allclose(merged_gs, torch.tensor([[-1.0, -2.0, 3.0], [-4.0, -5.0, 6.0]]), atol=1e-6)


check_example.example = True


def check_gs_is_projection(solve):
    """GS 分支：H = I - 2QQ^T，正交、且把 span(u) 里的向量整体取反。"""
    u = torch.tensor([[1.0, 0.5], [2.0, -1.0], [0.0, 3.0], [1.0, 1.0]])
    w0 = torch.eye(4)
    v, t, h, _ = solve(u, w0, True)
    q = torch.linalg.qr(u.to(torch.float64), mode="reduced").Q.float()
    assert torch.allclose(h, torch.eye(4) - 2 * (q @ q.T), atol=1e-5), h
    assert torch.allclose(t, 2 * torch.eye(2), atol=1e-6), t
    assert torch.allclose(h @ h.T, torch.eye(4), atol=1e-5), "H 必须正交"
    assert torch.allclose(h @ u, -u, atol=1e-5), "span(u) 内向量应被整体取反"
    assert torch.allclose(v, q, atol=1e-6)


def check_nogs_is_product_of_reflections(solve):
    """非 GS 分支：紧凑 WY 必须等于反射的逐个乘积，交叉项不能丢。"""
    u = torch.tensor([[1.0, 2.0], [0.0, 1.0], [0.0, 0.0]])
    _, _, h, _ = solve(u, torch.eye(3), False)
    want = _oracle_h(u, False)
    assert torch.allclose(h, want.float(), atol=1e-5), (h, want)
    u2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert torch.allclose(solve(u2, torch.eye(3), False)[2], torch.diag(torch.tensor([-1.0, -1.0, 1.0]))), (
        "正交列时两个分支应给出同一个 H"
    )


def check_r1_single_reflection(solve):
    """r=1：H = I - 2 v v^T，T=[[2]]。"""
    u = torch.tensor([[3.0], [0.0], [4.0]])
    w0 = torch.tensor([[1.0, 2.0, 3.0]])
    v, t, h, merged = solve(u, w0, False)
    assert torch.allclose(v, torch.tensor([[0.6], [0.0], [0.8]]), atol=1e-6), v
    assert torch.allclose(t, torch.tensor([[2.0]]), atol=1e-6), t
    want = torch.tensor([[0.28, 0.0, -0.96], [0.0, 1.0, 0.0], [-0.96, 0.0, -0.28]])
    assert torch.allclose(h, want, atol=1e-6), h
    assert torch.allclose(merged, w0 @ want, atol=1e-6), merged


def check_reverse_is_transpose(solve):
    """reverse=True 给 H.T；与 W0 @ H.T 一致。"""
    u = torch.tensor([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    w0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _, _, h, _ = solve(u, w0, False)
    _, t_rev, h_rev, merged_rev = solve(u, w0, False, True)
    assert torch.allclose(h_rev, h.T, atol=1e-5), (h_rev, h.T)
    assert torch.allclose(t_rev, solve(u, w0, False)[1].T, atol=1e-5), t_rev
    assert torch.allclose(merged_rev, w0 @ h.T, atol=1e-5), merged_rev


def check_merge_matches_forward_path(solve):
    """合并权重与「先对输入做 x H.T 再过基座层」等价。"""
    u = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, -1.0], [0.5, 0.0]])
    w0 = torch.randn(2, 4, generator=torch.Generator().manual_seed(7))
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(11))
    _, _, h, merged = solve(u, w0, False)
    assert torch.allclose(x @ merged.T, (x @ h.T) @ w0.T, atol=1e-5)


def check_bad_input(solve):
    """零列、形状不匹配、GS 下 r > in_features 都要报错。"""
    for bad in (
        lambda: solve(torch.zeros(3, 1), torch.eye(3), False),
        lambda: solve(torch.ones(3, 1), torch.eye(4), False),
        lambda: solve(torch.ones(2, 0), torch.eye(2), False),
        lambda: solve(torch.ones(2, 3), torch.eye(3), True),
        lambda: solve(torch.ones(3), torch.eye(3), False),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("应当抛出 ValueError")


def check_random_against_oracle(solve):
    """随机 30 组与独立 oracle 对拍（两种分支、正反两个方向）。"""
    rng = random.Random(92)
    gen = torch.Generator().manual_seed(92)
    for _ in range(30):
        n = rng.randint(2, 6)
        r = rng.randint(1, n)
        u = torch.randn(n, r, generator=gen)
        w0 = torch.randn(rng.randint(1, 4), n, generator=gen)
        for gs in (False, True):
            for rev in (False, True):
                _, _, h, merged = solve(u, w0, gs, rev)
                want = _oracle_h(u, gs, rev).float()
                assert torch.allclose(h, want, atol=1e-4), (n, r, gs, rev, h, want)
                assert torch.allclose(merged, w0 @ want, atol=1e-4), (n, r, gs, rev)
                assert torch.allclose(h @ h.T, torch.eye(n), atol=1e-4), "H 必须正交"
