import math

import torch


def _argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def _oracle(logits, tokens, step, steps, use_loo):
    """独立 oracle：alpha 账、LOO 平移、三路质量、归一化全部用 python 标量重做。"""
    batch, length, vocab = logits.shape
    alpha_t = step / steps
    alpha_s = (step + 1) / steps
    survival = alpha_t / alpha_s
    delta = math.log1p(vocab * alpha_t / (1.0 - alpha_t)) if (use_loo and 0.0 < alpha_t < 1.0) else 0.0

    routes = []
    preds = []
    for b in range(batch):
        row_routes = []
        row_preds = []
        for i in range(length):
            row = [float(v) for v in logits[b, i].tolist()]
            token = int(tokens[b, i].item())
            if delta:
                row[token] -= delta
            pred = _argmax(row)
            same = 1.0 if token == pred else 0.0
            clean = alpha_s * (1.0 - survival) / vocab + survival * alpha_s * same
            stay = survival * (1.0 - alpha_s) / vocab
            noise = (1.0 - survival) * (1.0 - alpha_s) / vocab
            total = clean + stay + noise
            row_routes.append([clean / total, stay / total, noise / total])
            row_preds.append(pred)
        routes.append(row_routes)
        preds.append(row_preds)

    return (
        torch.tensor(routes, dtype=torch.float64).reshape(batch, length, 3),
        torch.tensor(preds, dtype=torch.int64).reshape(batch, length),
    )


LOGITS = torch.tensor([[[0.0, 1.0, 3.0, 0.5], [0.0, 2.5, 1.0, 0.0]]])
TOKENS = torch.tensor([[0, 1]])
LOO_LOGITS = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
LOO_TOKENS = torch.tensor([[1]])


def check_example(solve):
    routes, pred = solve(LOGITS, TOKENS, 1, 4)
    assert pred.tolist() == [[2, 1]], pred.tolist()
    assert routes.dtype == torch.float32 and routes.shape == (1, 2, 3)
    assert pred.dtype == torch.int64
    # 第一个位置 x_t != x0：三路权重相等；第二个位置命中 x0，clean 占大头
    assert [[round(float(v), 6) for v in row] for row in routes[0].tolist()] == [
        [0.333333, 0.333333, 0.333333],
        [0.714286, 0.142857, 0.142857],
    ], routes.tolist()

    first, _ = solve(LOGITS, TOKENS, 0, 4)
    assert [[round(float(v), 6) for v in row] for row in first[0].tolist()] == [
        [0.25, 0.0, 0.75],
        [0.25, 0.0, 0.75],
    ], first.tolist()

    last, last_pred = solve(LOGITS, TOKENS, 3, 4)
    assert last_pred.tolist() == [[2, 1]]
    assert torch.allclose(last, torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]), atol=1e-6), last.tolist()


check_example.example = True


def check_last_step_commits(solve):
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    tokens = torch.randint(0, 5, (2, 4))
    for steps in (2, 4, 9):
        routes, _ = solve(logits, tokens, steps - 1, steps)
        assert torch.allclose(routes[..., 0], torch.ones_like(routes[..., 0]), atol=1e-6)
        assert torch.allclose(routes[..., 1:], torch.zeros_like(routes[..., 1:]), atol=1e-6)


def check_first_step(solve):
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 6)
    tokens = torch.randint(0, 6, (2, 3))
    for steps in (1, 3, 8):
        routes, _ = solve(logits, tokens, 0, steps)
        b = 1.0 / steps
        # alpha_t=0：clean = b/K，stay = 0，noise = (1-b)/K，归一化后与 same 无关
        assert torch.allclose(routes[..., 0], torch.full_like(routes[..., 0], b), atol=1e-6)
        assert torch.allclose(routes[..., 1], torch.zeros_like(routes[..., 1]), atol=1e-6)
        assert torch.allclose(routes[..., 2], torch.full_like(routes[..., 2], 1.0 - b), atol=1e-6)


def check_route_normalization(solve):
    torch.manual_seed(2)
    logits = torch.randn(2, 3, 4)
    tokens = torch.randint(0, 4, (2, 3))
    vocab = 4
    for steps in (3, 5):
        for step in range(steps):
            routes, pred = solve(logits, tokens, step, steps)
            routes = routes.to(torch.float64)
            assert torch.allclose(routes.sum(-1), torch.ones_like(routes[..., 0]), atol=1e-6)
            a = (step / steps) / ((step + 1) / steps)
            b = (step + 1) / steps
            same = (tokens == pred).to(torch.float64)
            clean = b * (1 - a) / vocab + a * b * same
            stay = a * (1 - b) / vocab
            noise = (1 - a) * (1 - b) / vocab
            total = clean + stay + noise
            # 未归一化的三元组在 same=0 处和为 (1 - a*b)/K，因此归一化不是空操作
            if step > 0:
                want = torch.full_like(total[~same.bool()], (1 - a * b) / vocab)
                assert torch.allclose(total[~same.bool()], want, atol=1e-6)
            assert torch.allclose(routes[..., 0], (clean / total).to(torch.float32).to(torch.float64), atol=1e-5)


def check_loo(solve):
    # 观测 token 的 logit 为 1.0，其余为 0；step=2/N=4 时 delta = log(5) ≈ 1.609
    plain_routes, plain_pred = solve(LOO_LOGITS, LOO_TOKENS, 2, 4)
    assert plain_pred.tolist() == [[1]]
    loo_routes, loo_pred = solve(LOO_LOGITS, LOO_TOKENS, 2, 4, use_loo=True)
    assert loo_pred.tolist() == [[0]], "LOO 修正后观测 token 不应再是 argmax"
    assert not torch.allclose(plain_routes, loo_routes, atol=1e-3)

    # delta = log1p(K * alpha_t / (1 - alpha_t))，把观测位置的 logit 压到 -delta 之后
    K = 4
    alpha_t = 2 / 4
    delta = math.log1p(K * alpha_t / (1 - alpha_t))
    assert abs(delta - math.log(5.0)) < 1e-12
    assert 1.0 - delta < 0.0, "该步修正应把观测 logit 压到负数"

    # step=0 时 alpha_t=0，不做修正
    at0, pred0 = solve(LOO_LOGITS, LOO_TOKENS, 0, 4, use_loo=True)
    assert pred0.tolist() == [[1]]
    assert torch.allclose(at0, solve(LOO_LOGITS, LOO_TOKENS, 0, 4)[0], atol=1e-7)

    # 最后一步同样会修正
    _, pred_last = solve(LOO_LOGITS, LOO_TOKENS, 3, 4, use_loo=True)
    assert pred_last.tolist() == [[0]]


def check_against_oracle(solve):
    torch.manual_seed(3)
    for vocab, length in ((3, 2), (5, 3), (7, 4)):
        logits = torch.randn(2, length, vocab, dtype=torch.float64)
        tokens = torch.randint(0, vocab, (2, length))
        for steps in (2, 4, 6):
            for step in range(steps):
                for use_loo in (False, True):
                    routes, pred = solve(logits, tokens, step, steps, use_loo=use_loo)
                    want_routes, want_pred = _oracle(logits, tokens, step, steps, use_loo)
                    assert torch.equal(pred, want_pred), f"pred 不一致: K={vocab}, step={step}, loo={use_loo}"
                    assert torch.allclose(routes, want_routes.to(torch.float32), atol=1e-6), (
                        f"routes 不一致: K={vocab}, step={step}, loo={use_loo}"
                    )


def check_invalid(solve):
    logits = torch.zeros(2, 3, 4)
    tokens = torch.zeros(2, 3, dtype=torch.int64)
    for bad in (
        lambda: solve([[0.0, 1.0]], tokens, 0, 4),
        lambda: solve(torch.zeros(2, 4), tokens, 0, 4),
        lambda: solve(logits, [[0, 1, 2]], 0, 4),
        lambda: solve(logits, torch.zeros(2, 4, dtype=torch.int64), 0, 4),
        lambda: solve(logits, torch.zeros(2, 3), 0, 4),
        lambda: solve(logits, torch.full((2, 3), 4, dtype=torch.int64), 0, 4),
        lambda: solve(logits, torch.full((2, 3), -1, dtype=torch.int64), 0, 4),
        lambda: solve(logits, tokens, -1, 4),
        lambda: solve(logits, tokens, 4, 4),
        lambda: solve(logits, tokens, 0.5, 4),
        lambda: solve(logits, tokens, True, 4),
        lambda: solve(logits, tokens, 0, 0),
        lambda: solve(torch.zeros(2, 3, 1), torch.zeros(2, 3, dtype=torch.int64), 0, 4),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
