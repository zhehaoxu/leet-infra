import math

import torch


def _oracle(mu, cov, keys, values, n_sink, epsilon):
    """独立：三重 for 循环手算高斯期望 + softmax + vnorm。"""
    H, L, D = keys.shape
    Lp = L - n_sink
    body = torch.zeros(H, Lp, dtype=torch.float32)
    for h in range(H):
        raw = []
        for n in range(Lp):
            kn = [float(keys[h, n_sink + n, d]) for d in range(D)]
            s1 = sum(float(mu[h, d]) * kn[d] for d in range(D)) / math.sqrt(D)
            t = 0.0
            for i in range(D):
                for j in range(D):
                    t += kn[i] * float(cov[h, i, j]) * kn[j]
            s2 = t / D / 2
            raw.append(s1 + s2)
        mx = max(raw)
        ex = [math.exp(r - mx) for r in raw]
        sm = sum(ex)
        for n in range(Lp):
            p = ex[n] / sm
            vn = math.sqrt(sum(float(values[h, n_sink + n, d]) ** 2 for d in range(D)))
            body[h, n] = (p + epsilon) * vn
    m = float(body.max()) + 1.0
    out = torch.zeros(H, L, dtype=torch.float32)
    for h in range(H):
        for i in range(n_sink):
            out[h, i] = m
        for n in range(Lp):
            out[h, n_sink + n] = body[h, n]
    return out


def check_example(solve):
    mu = torch.tensor([[0.5, 0.2, -0.1], [0.1, 0.4, 0.3]])
    cov = torch.tensor([
        [[1.0, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.6]],
        [[0.7, 0.0, 0.2], [0.0, 0.9, 0.1], [0.2, 0.1, 0.5]],
    ])
    keys = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 1.0], [0.2, 0.3, 0.4], [0.0, 0.0, 2.0]],
        [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.5], [0.3, 0.3, 0.3], [1.5, 0.0, 0.0]],
    ])
    values = torch.tensor([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.5]],
        [[0.2, 0.1, 0.0], [0.3, 0.3, 0.3], [2.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 2.0, 0.0], [0.5, 1.0, 1.5]],
    ])
    out = solve(mu, cov, keys, values, 2, 0.1)
    exp = torch.tensor([
        [1.930834, 1.930834, 0.322050, 0.470150, 0.404386, 0.278735],
        [1.930834, 1.930834, 0.603884, 0.930834, 0.530053, 0.553043],
    ])
    assert torch.allclose(out, exp, atol=1e-4), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 3, (1,)))
        L = int(torch.randint(3, 8, (1,)))
        D = int(torch.randint(1, 4, (1,)))
        n_sink = int(torch.randint(1, min(3, L - 1) + 1, (1,)))
        mu = torch.randn(H, D) * 0.3
        A = torch.randn(H, D, D) * 0.3
        cov = A @ A.transpose(-2, -1) + torch.eye(D).expand(H, D, D) * 0.1
        keys = torch.randn(H, L, D)
        values = torch.randn(H, L, D)
        eps = float(torch.rand(1) * 0.3)
        got = solve(mu, cov, keys, values, n_sink, eps)
        exp = _oracle(mu, cov, keys, values, n_sink, eps)
        assert torch.allclose(got, exp, atol=1e-4), (H, L, D, n_sink, eps)


def check_sink_is_global_max_plus_one(solve):
    mu = torch.randn(2, 3) * 0.2
    A = torch.randn(2, 3, 3) * 0.2
    cov = A @ A.transpose(-2, -1) + torch.eye(3).expand(2, 3, 3) * 0.5
    keys = torch.randn(2, 5, 3)
    values = torch.randn(2, 5, 3)
    out = solve(mu, cov, keys, values, 2, 0.0)
    sink_val = float(out[0, 0])
    body_max = float(out[:, 2:].max())
    assert abs(sink_val - body_max - 1.0) < 1e-5, (sink_val, body_max)


def check_epsilon_shift(solve):
    # epsilon=0 与 epsilon>0 的差别：body 部分每位置加 epsilon * ||v||
    mu = torch.tensor([[0.3, 0.1, 0.0]])
    cov = torch.eye(3).unsqueeze(0)
    keys = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]])
    values = torch.tensor([[[2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]]])
    a = solve(mu, cov, keys, values, 0, 0.0)
    b = solve(mu, cov, keys, values, 0, 0.1)
    # 差 = 0.1 * ||v||；||v|| = [2, 3, 4]
    exp_diff = torch.tensor([[0.2, 0.3, 0.4]])
    assert torch.allclose(b - a, exp_diff, atol=1e-5), (a, b, b - a)


def check_shape(solve):
    mu = torch.randn(2, 3)
    cov = torch.eye(3).expand(2, 3, 3)
    keys = torch.randn(2, 5, 3)
    values = torch.randn(2, 5, 3)
    out = solve(mu, cov, keys, values, 1, 0.0)
    assert out.shape == (2, 5), out.shape


check_example.example = True
