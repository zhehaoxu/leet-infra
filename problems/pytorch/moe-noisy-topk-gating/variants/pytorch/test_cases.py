import math

import torch


def _softplus(z):
    if z > 0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _oracle(x, gate_w, noise_w, noise, k, training=True):
    rows, gw, nw, eps = x.tolist(), gate_w.tolist(), noise_w.tolist(), noise.tolist()
    t, e = len(rows), len(gw)
    gates = [[0.0] * e for _ in range(t)]
    indices, effective = [], []
    for i, row in enumerate(rows):
        logits = [math.fsum(a * b for a, b in zip(row, gw[eid])) for eid in range(e)]
        if training:
            add = [math.fsum(a * b for a, b in zip(row, nw[eid])) for eid in range(e)]
            logits = [z + eps[i][eid] * _softplus(s) for eid, (z, s) in enumerate(zip(logits, add))]
        order = sorted(range(e), key=lambda eid: (-logits[eid], eid))
        picked = order[:k]
        chosen = [logits[eid] for eid in picked]
        top = max(chosen)
        denom = math.fsum(math.exp(v - top) for v in chosen)
        for eid, v in zip(picked, chosen):
            gates[i][eid] = math.exp(v - top) / denom
        indices.append(picked)
        effective.append(logits)
    return (torch.tensor(gates, dtype=torch.float64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(effective, dtype=torch.float64))


def _case(t=4, d=3, e=4, k=2, seed=23, training=True, noise_scale=0.7):
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand((t, d), generator=rng, dtype=torch.float64) * 4 - 2
    gw = torch.rand((e, d), generator=rng, dtype=torch.float64) * 2 - 1
    nw = torch.rand((e, d), generator=rng, dtype=torch.float64) * 0.4 - 0.2
    eps = torch.randn((t, e), generator=rng, dtype=torch.float64) * noise_scale
    return x, gw, nw, eps, k, training


def _example():
    return (torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0]], dtype=torch.float64), 1, True)


def _verify(solve, args):
    before = [v.clone() for v in args[:4]]
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 3
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == target.dtype and actual.shape == target.shape
        if target.dtype == torch.int64:
            assert torch.equal(actual, target)
        else:
            torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11)
    for actual, original in zip(args[:4], before):
        assert torch.equal(actual, original)
    return result


def check_example_no_noise(solve):
    gates, ids, logits = _verify(solve, _example())
    assert ids.tolist() == [[0]]
    torch.testing.assert_close(gates, torch.tensor([[1.0, 0.0]], dtype=torch.float64))


def check_zero_noise_reduces_to_deterministic(solve):
    args = list(_case())
    args[3] = torch.zeros_like(args[3])
    noisy = _verify(solve, args)
    args[5] = False
    clean = _verify(solve, args)
    for a, b in zip(noisy, clean):
        assert torch.equal(a, b)


def check_noise_changes_selection(solve):
    args = (torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[1.0], [0.9]], dtype=torch.float64),
            torch.tensor([[0.0], [0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64), 1, True)
    _, ids, logits = _verify(solve, args)
    assert ids.tolist() == [[1]]
    torch.testing.assert_close(logits, torch.tensor([[1.0, 0.9 + math.log(2.0)]], dtype=torch.float64))


def check_softplus_scale_not_raw_abs(solve):
    args = (torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[0.0], [0.0]], dtype=torch.float64),
            torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
            torch.tensor([[1.0, 1.0]], dtype=torch.float64), 1, True)
    _, _, logits = _verify(solve, args)
    torch.testing.assert_close(
        logits, torch.tensor([[math.log1p(math.e), math.log1p(math.exp(-1))]],
                             dtype=torch.float64))


def check_weights_softmax_only_selected(solve):
    _, _, logits = _verify(solve, _case(k=3))
    gates, ids, _ = solve(*_case(k=3))
    top = logits.gather(1, ids)
    expected = torch.softmax(top, dim=-1)
    torch.testing.assert_close(gates.gather(1, ids), expected)
    assert torch.count_nonzero(gates) == ids.numel()
    assert torch.allclose(gates.sum(dim=1), torch.ones(gates.shape[0], dtype=torch.float64))


def check_k_equals_experts(solve):
    gates, ids, logits = _verify(solve, _case(e=5, k=5))
    torch.testing.assert_close(gates, torch.softmax(logits, dim=-1))
    assert sorted(int(v) for v in ids[0]) == list(range(5))


def check_tie_breaks_to_lower_index(solve):
    args = (torch.tensor([[1.0]], dtype=torch.float64), torch.ones((3, 1), dtype=torch.float64),
            torch.zeros((3, 1), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64), 2, True)
    _, ids, _ = _verify(solve, args)
    assert ids.tolist() == [[0, 1]]


def check_training_false_ignores_noise(solve):
    args = list(_case(training=False))
    baseline = _verify(solve, args)
    args[3] = args[3] * 100 + 50
    altered = _verify(solve, args)
    for a, b in zip(baseline, altered):
        assert torch.equal(a, b)


def check_training_uses_noise(solve):
    args = list(_case(training=True))
    baseline = _verify(solve, args)
    args[3] = args[3] + 3
    altered = _verify(solve, args)
    assert not torch.equal(baseline[2], altered[2])


def check_negative_scale_stays_positive(solve):
    args = (torch.tensor([[1.0]], dtype=torch.float64),
            torch.zeros((2, 1), dtype=torch.float64),
            torch.tensor([[-2.0], [2.0]], dtype=torch.float64),
            torch.ones((1, 2), dtype=torch.float64), 1, True)
    _, ids, logits = _verify(solve, args)
    assert ids.tolist() == [[1]]
    assert logits[0, 0] < logits[0, 1]
    torch.testing.assert_close(logits[0, 0],
                               torch.tensor(math.log1p(math.exp(-2)), dtype=torch.float64))


def check_noncontiguous_inputs(solve):
    args = list(_case())
    for j in range(4):
        tensor = args[j]
        buffer = torch.empty((*tensor.shape, 2), dtype=tensor.dtype)
        buffer[..., 0] = tensor
        args[j] = buffer[..., 0]
        assert not args[j].is_contiguous()
    _verify(solve, args)


def check_zero_rows(solve):
    args = list(_case())
    args[0] = torch.zeros_like(args[0])
    args[3] = torch.zeros_like(args[3])
    gates, ids, logits = _verify(solve, args)
    assert torch.count_nonzero(gates.gather(1, ids) - 1.0 / args[4]) == 0
    assert torch.count_nonzero(gates) == args[4] * args[0].shape[0]


def check_randomized_scalar_oracle(solve):
    for seed in range(30):
        e = 1 + seed % 8
        _verify(solve, _case(t=1 + seed % 8, d=1 + seed % 5, e=e,
                             k=1 + seed % e, seed=seed,
                             training=bool(seed % 3)))


def check_invalid_k(solve):
    for k in (0, 5, -1, 2.0, True):
        try:
            solve(*_case(e=4, k=2)[:4], k, False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"k={k!r} should raise ValueError")


def check_maximum_shape(solve):
    _verify(solve, _case(t=8, d=8, e=8, k=8))


check_example_no_noise.example = True
check_noise_changes_selection.example = True
