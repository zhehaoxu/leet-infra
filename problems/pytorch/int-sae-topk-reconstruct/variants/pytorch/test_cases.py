import math
import random

import torch


def _oracle(x, W_enc, b_enc, W_dec, b_dec, k):
    B = len(x)
    D = len(x[0])
    F_ = len(W_enc)
    acts = []
    for b in range(B):
        pre = []
        for f in range(F_):
            v = sum((x[b][d] - b_dec[d]) * W_enc[f][d] for d in range(D)) + b_enc[f]
            pre.append(v)
        order = sorted(range(F_), key=lambda f: (-pre[f], f))[:k]
        row = [0.0] * F_
        for f in order:
            row[f] = pre[f]
        acts.append(row)
    recon = [[sum(acts[b][f] * W_dec[f][d] for f in range(F_)) + b_dec[d] for d in range(D)] for b in range(B)]
    err = [[x[b][d] - recon[b][d] for d in range(D)] for b in range(B)]
    l0 = [sum(1 for f in range(F_) if acts[b][f] != 0.0) for b in range(B)]
    mse = [sum(e * e for e in err[b]) / D for b in range(B)]
    max_feat = []
    for b in range(B):
        best, bi = -1.0, 0
        for f in range(F_):
            a = abs(acts[b][f])
            if a > best:
                best, bi = a, f
        max_feat.append(bi)
    return acts, recon, err, l0, mse, max_feat


def _py(t):
    return t.detach().float().tolist()


def _close(a, b, tol=1e-4):
    fa, fb = [], []

    def rec(x, y):
        if isinstance(x, list):
            for u, v in zip(x, y):
                rec(u, v)
        else:
            fa.append(x)
            fb.append(y)
    rec(a, b)
    return all(abs(p - q) <= tol * max(1.0, abs(q)) for p, q in zip(fa, fb))


def _rand_case(rng, B=2, D=6, F_=10):
    g = torch.Generator().manual_seed(rng.randrange(10**9))
    x = torch.randn(B, D, generator=g)
    W_enc = torch.randn(F_, D, generator=g) * 0.6
    b_enc = torch.randn(F_, generator=g) * 0.1
    W_dec = torch.randn(F_, D, generator=g) * 0.6
    b_dec = torch.randn(D, generator=g) * 0.1
    k = rng.randint(1, F_)
    return x, W_enc, b_enc, W_dec, b_dec, k


def check_random(solve):
    rng = random.Random(652)
    for _ in range(25):
        x, W_enc, b_enc, W_dec, b_dec, k = _rand_case(rng)
        led = solve(x, W_enc, b_enc, W_dec, b_dec, k)
        acts, recon, err, l0, mse, mf = _oracle(_py(x), _py(W_enc), _py(b_enc), _py(W_dec), _py(b_dec), k)
        assert _close(_py(led["feature_acts"]), acts)
        assert _close(_py(led["recon"]), recon)
        assert _close(_py(led["recon_error"]), err)
        assert _py(led["l0"]) == l0
        assert _close(_py(led["mse"]), mse)
        assert _py(led["max_feat"]) == mf


def check_anchor_toy(solve):
    # 手算锚点：D=2, F=3, k=2, W_dec = I（特征即维度方向）
    x = torch.tensor([[1.0, 2.0]])
    W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    b_enc = torch.tensor([0.0, 0.0, 1.0])
    W_dec = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b_dec = torch.zeros(2)
    led = solve(x, W_enc, b_enc, W_dec, b_dec, 2)
    # pre = [1, 2, 1+1+1=2.5] -> top2 = feat2 (2.5), feat1 (2.0)
    acts = _py(led["feature_acts"])[0]
    assert acts[0] == 0.0
    assert abs(acts[1] - 2.0) < 1e-6
    assert abs(acts[2] - 2.5) < 1e-6
    # recon = 2*e1 + 2.5*e2_dir? W_dec[2]=[0,0] -> recon = [0, 2]
    assert abs(_py(led["recon"])[0][0] - 0.0) < 1e-6
    assert abs(_py(led["recon"])[0][1] - 2.0) < 1e-6
    assert _py(led["l0"])[0] == 2
    # mse = ((1-0)^2 + (2-2)^2)/2 = 0.5
    assert abs(_py(led["mse"])[0] - 0.5) < 1e-6
    assert _py(led["max_feat"])[0] == 2


def check_k_equal_F(solve):
    # k=F 时激活 = 完整 pre_acts（无清零）
    x, W_enc, b_enc, W_dec, b_dec, _ = _rand_case(random.Random(1652))
    F_ = W_enc.shape[0]
    led = solve(x, W_enc, b_enc, W_dec, b_dec, F_)
    pre = (x - b_dec) @ W_enc.T + b_enc
    assert torch.allclose(led["feature_acts"], pre, atol=1e-5)
    assert torch.allclose(led["recon"], pre @ W_dec + b_dec, atol=1e-5)


def check_dead_features(solve):
    # k < 非零特征数时恰好 k 个非零；pre-act 相同的并列取较小 index
    x = torch.zeros(1, 3)
    W_enc = torch.eye(4, 3)  # 4 特征
    b_enc = torch.tensor([1.0, 1.0, 1.0, 1.0])
    W_dec = torch.zeros(4, 3)
    b_dec = torch.zeros(3)
    led = solve(x, W_enc, b_enc, W_dec, b_dec, 2)
    acts = _py(led["feature_acts"])[0]
    nz = [f for f, v in enumerate(acts) if v != 0.0]
    assert nz == [0, 1], nz
    assert _py(led["max_feat"])[0] == 0


def check_negative_pre_acts_kept(solve):
    # TopK 不做 ReLU：负的 pre-act 也入选（全负输入时 top-k 仍是最大的 k 个负值）
    x = torch.tensor([[0.0, 0.0]])
    W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b_enc = torch.tensor([-1.0, -2.0])
    W_dec = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b_dec = torch.zeros(2)
    led = solve(x, W_enc, b_enc, W_dec, b_dec, 1)
    acts = _py(led["feature_acts"])[0]
    assert abs(acts[0] - (-1.0)) < 1e-6 and acts[1] == 0.0
    # recon = -1 * e0
    assert abs(_py(led["recon"])[0][0] - (-1.0)) < 1e-6


def check_recon_error_identity(solve):
    x, W_enc, b_enc, W_dec, b_dec, k = _rand_case(random.Random(2652))
    led = solve(x, W_enc, b_enc, W_dec, b_dec, k)
    assert torch.allclose(led["recon_error"], x.float() - led["recon"], atol=1e-5)
    mse_manual = (led["recon_error"] ** 2).mean(-1)
    assert torch.allclose(led["mse"], mse_manual, atol=1e-6)


def check_noncontiguous(solve):
    x, W_enc, b_enc, W_dec, b_dec, k = _rand_case(random.Random(3652))
    D = x.shape[1]
    big = torch.randn(2, 2 * D)
    nc = big[:, ::2][:, :D]
    assert not nc.is_contiguous()
    led = solve(nc, W_enc, b_enc, W_dec, b_dec, k)
    led_ref = solve(nc.contiguous(), W_enc, b_enc, W_dec, b_dec, k)
    assert torch.allclose(led["feature_acts"], led_ref["feature_acts"], atol=1e-6)


def check_invalid_k(solve):
    x, W_enc, b_enc, W_dec, b_dec, _ = _rand_case(random.Random(4652))
    for bad in (0, -1, W_enc.shape[0] + 1):
        try:
            solve(x, W_enc, b_enc, W_dec, b_dec, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for k={bad}")
