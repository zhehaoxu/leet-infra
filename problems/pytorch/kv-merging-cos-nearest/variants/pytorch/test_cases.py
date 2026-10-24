import math

import torch


_EPS = 1e-6


def _oracle(keys, values, scores, n_kept):
    """独立：纯 Python 逐 head / evict 累加。"""
    H, L, D = keys.shape
    result = values.clone().float()
    for h in range(H):
        order = sorted(range(L), key=lambda j: (-float(scores[h, j]), j))[:n_kept]
        evict = [j for j in range(L) if j not in order]
        if not evict or not order:
            continue

        def norm_row(vec):
            n = math.sqrt(sum(float(x) * float(x) for x in vec))
            n = max(n, _EPS)
            return [float(x) / n for x in vec]

        kk = [norm_row(keys[h, j]) for j in order]
        ek = [norm_row(keys[h, j]) for j in evict]

        vacc = [[0.0] * D for _ in order]
        wacc = [0.0] * len(order)
        for ei in range(len(evict)):
            e_pos = evict[ei]
            best_s = -1e30
            best_k = 0
            for ki in range(len(order)):
                s = sum(ek[ei][d] * kk[ki][d] for d in range(D))
                if s > best_s:
                    best_s = s
                    best_k = ki
            if best_s < 0:
                continue
            en = math.sqrt(sum(float(values[h, e_pos, d]) ** 2 for d in range(D)))
            tp = order[best_k]
            tn = math.sqrt(sum(float(values[h, tp, d]) ** 2 for d in range(D)))
            w = best_s * en / (en + tn + _EPS)
            for d in range(D):
                vacc[best_k][d] += w * float(values[h, e_pos, d])
            wacc[best_k] += w

        for ki, kp in enumerate(order):
            if wacc[ki] > 0:
                for d in range(D):
                    orig = float(values[h, kp, d])
                    result[h, kp, d] = (orig + vacc[ki][d]) / (1.0 + wacc[ki])
    return result


def check_example(solve):
    keys = torch.tensor([
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.2, 0.8]],
        [[1.0, 1.0], [0.5, 0.5], [2.0, 0.0], [0.0, 2.0]],
    ])
    values = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        [[0.5, 0.5], [1.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
    ])
    scores = torch.tensor([[0.9, 0.1, 0.7, 0.2], [0.3, 0.4, 0.8, 0.1]])
    out = solve(keys, values, scores, 2)
    exp = torch.tensor([
        [[1.814293, 2.814292], [3.0, 4.0], [5.717330, 6.717330], [7.0, 8.0]],
        [[0.5, 0.5], [1.821373, 2.124212], [2.0, 3.0], [4.0, 5.0]],
    ])
    assert torch.allclose(out, exp, atol=1e-4), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 3, (1,)))
        L = int(torch.randint(3, 7, (1,)))
        D = int(torch.randint(2, 5, (1,)))
        n_kept = int(torch.randint(1, L, (1,)))
        keys = torch.randn(H, L, D)
        values = torch.randn(H, L, D)
        scores = torch.rand(H, L)
        got = solve(keys, values, scores, n_kept)
        exp = _oracle(keys, values, scores, n_kept)
        assert torch.allclose(got, exp, atol=1e-4), (H, L, D, n_kept)


def check_all_kept_no_merge(solve):
    keys = torch.randn(2, 3, 4)
    values = torch.randn(2, 3, 4)
    scores = torch.rand(2, 3)
    out = solve(keys, values, scores, 3)  # n_kept = L → 无 evict
    assert torch.allclose(out, values.float(), atol=1e-6), out


def check_evicted_positions_unchanged(solve):
    # 剪掉的 slot 上 value 应保持原值
    keys = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]])
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    scores = torch.tensor([[0.9, 0.1, 0.7]])
    out = solve(keys, values, scores, 2)
    # evict 是 idx 1 → 原值不变
    assert torch.allclose(out[0, 1], torch.tensor([3.0, 4.0]), atol=1e-6), out


def check_shape_dtype(solve):
    keys = torch.randn(2, 5, 3)
    values = torch.randn(2, 5, 3)
    scores = torch.rand(2, 5)
    out = solve(keys, values, scores, 3)
    assert out.shape == (2, 5, 3), out.shape
    assert out.dtype == torch.float32, out.dtype


check_example.example = True
