import torch

QF = torch.tensor([[[1.0, -2.0]], [[-0.5, 0.25]]])          # (N=2, H=1, D=2)
KEYS = torch.tensor([
    [[[0.5, 1.0], [0.0, -1.0], [2.0, 0.5], [-1.0, -0.5]]],
    [[[1.0, 1.0], [2.0, 0.0], [0.0, 4.0], [-1.0, 2.0]]],
])                                                          # (N=2, H=1, L=4, D=2)


def _oracle(q_filters, keys, cr):
    """独立纯 Python 四重循环 oracle（与 solution 的广播/向量化路径不同）。"""
    N, H, D = q_filters.shape
    L = keys.shape[2]
    scores = []
    kept = []
    n_kept = max(1, int(L * (1 - cr)))
    for n in range(N):
        layer_s = []
        layer_k = []
        for h in range(H):
            row = []
            for t in range(L):
                acc = 0.0
                for d in range(D):
                    acc += float(q_filters[n, h, d]) * float(keys[n, h, t, d])
                row.append(-acc)
            order = sorted(range(L), key=lambda j: (-row[j], j))[:n_kept]
            layer_s.append(row)
            layer_k.append(sorted(order))
        scores.append(layer_s)
        kept.append(layer_k)
    return (torch.tensor(scores, dtype=torch.float32),
            torch.tensor(kept, dtype=torch.long))


def check_example(solve):
    scores, kept = solve(QF, KEYS, 0.5)
    exp_scores = torch.tensor([[[1.5, -2.0, -1.0, -0.0]], [[0.25, 1.0, -1.0, -1.0]]])
    exp_kept = torch.tensor([[[0, 3]], [[0, 1]]])
    assert torch.allclose(scores, exp_scores, atol=1e-5, rtol=1e-5), scores
    assert torch.equal(kept, exp_kept), kept


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(5)
    for _ in range(30):
        N = int(torch.randint(1, 4, (1,)))
        H = int(torch.randint(1, 4, (1,)))
        D = int(torch.randint(1, 9, (1,)))
        L = int(torch.randint(1, 15, (1,)))
        cr = float(torch.rand(1)) * 0.9
        qf = torch.randn(N, H, D)
        keys = torch.randn(N, H, L, D)
        gs, gk = solve(qf, keys, cr)
        es, ek = _oracle(qf, keys, cr)
        assert torch.allclose(gs, es, atol=1e-5, rtol=1e-5), (gs, es)
        assert torch.equal(gk, ek), (cr, gk, ek)


def check_zero_filter_ties(solve):
    # 零滤波器 → 全 0 分：平局规则保留最左 n_kept 个
    scores, kept = solve(torch.zeros(1, 1, 2), torch.randn(1, 1, 4, 2), 0.5)
    assert torch.allclose(scores, torch.zeros(1, 1, 4), atol=1e-5, rtol=1e-5), scores
    assert torch.equal(kept, torch.tensor([[[0, 1]]])), kept


def check_sign_semantics(solve):
    # f = e0：score = -k[...,0]；保留的应是第 0 维最小（点积最负）的 key
    qf = torch.tensor([[[1.0, 0.0]]])
    keys = torch.tensor([[[[3.0, 9.0], [-2.0, 1.0], [0.5, -7.0], [-1.0, 2.0]]]])
    scores, kept = solve(qf, keys, 0.5)
    assert torch.allclose(scores, torch.tensor([[[-3.0, 2.0, -0.5, 1.0]]]), atol=1e-5, rtol=1e-5), scores
    assert torch.equal(kept, torch.tensor([[[1, 3]]])), kept


def check_ratio_zero_keeps_all(solve):
    scores, kept = solve(QF, KEYS, 0.0)
    assert kept.shape == (2, 1, 4), kept.shape
    assert torch.equal(kept, torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]]])), kept


def check_single_layer_invariance(solve):
    # 单层退化：与 _oracle 逐元素一致，且 n_kept=1 时保留全局最反向 key
    torch.manual_seed(13)
    qf = torch.randn(1, 3, 5)
    keys = torch.randn(1, 3, 12, 5)
    scores, kept = solve(qf, keys, 11 / 12)
    es, ek = _oracle(qf, keys, 11 / 12)
    assert torch.allclose(scores, es, atol=1e-5, rtol=1e-5), (scores, es)
    assert torch.equal(kept, ek) and kept.shape == (1, 3, 1), (kept, ek)
    # 独立验证：kept 恰为该 (n,h) 行 argmin(点积)
    dots = (qf[0, :, None, :] * keys[0]).sum(dim=-1)   # (3, 12)
    assert torch.equal(kept[0, :, 0], dots.argmin(dim=-1)), kept
