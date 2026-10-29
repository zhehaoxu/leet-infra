import math

import torch


def _mlp(row, gate, up, down):
    hidden = []
    for gw, uw in zip(gate, up):
        g = math.fsum(a * b for a, b in zip(row, gw))
        u = math.fsum(a * b for a, b in zip(row, uw))
        hidden.append(g / (1.0 + math.exp(-g)) * u)
    return [math.fsum(a * b for a, b in zip(hidden, dw)) for dw in down]


def _oracle(*args):
    x, ids, weights, gw, uw, dw, sg, su, sd = [v.tolist() for v in args]
    routed, shared = [], []
    for row, selected, coeffs in zip(x, ids, weights):
        outputs = [_mlp(row, gw[e], uw[e], dw[e]) for e in selected]
        routed.append([math.fsum(c * out[d] for c, out in zip(coeffs, outputs))
                       for d in range(len(row))])
        shared.append(_mlp(row, sg, su, sd))
    r = torch.tensor(routed, dtype=torch.float64)
    s = torch.tensor(shared, dtype=torch.float64)
    total = sum(t.numel() for t in args[3:])
    active = len(ids[0]) * sum(args[j][0].numel() for j in (3, 4, 5))
    active += sum(t.numel() for t in args[6:])
    return r + s, r, s, total, active


def _case(t=3, e=4, d=3, h=2, hs=5, k=2, seed=41):
    rng = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(shape, dtype=torch.float64, generator=rng) * 2 - 1

    ids = torch.stack([torch.randperm(e, generator=rng)[:k] for _ in range(t)])
    return (rand(t, d), ids, torch.rand(t, k, dtype=torch.float64, generator=rng) * 2,
            rand(e, h, d), rand(e, h, d), rand(e, d, h),
            rand(hs, d), rand(hs, d), rand(d, hs))


def _example():
    return (torch.ones((1, 1), dtype=torch.float64), torch.tensor([[1, 0]]),
            torch.tensor([[2., 0.5]], dtype=torch.float64),
            torch.ones((2, 1, 1), dtype=torch.float64),
            torch.tensor([[[2.]], [[4.]]], dtype=torch.float64),
            torch.ones((2, 1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            torch.tensor([[3.]], dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64))


def _verify(solve, args):
    snapshots = [v.clone() for v in args]
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 5
    for actual, target in zip(result[:3], expected[:3]):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == torch.float64 and actual.device.type == "cpu"
        torch.testing.assert_close(actual, target, rtol=2e-10, atol=2e-10)
    for actual, target in zip(result[3:], expected[3:]):
        assert type(actual) is int and actual == target
    for actual, original in zip(args, snapshots):
        assert torch.equal(actual, original), "inputs were mutated"
    return result


def check_example(solve):
    out, routed, shared, total, active = _verify(solve, _example())
    a = 1 / (1 + math.exp(-1))
    torch.testing.assert_close(out, torch.tensor([[12 * a]], dtype=torch.float64))
    assert total == active == 9


def check_zero_route_keeps_shared(solve):
    args = list(_case())
    args[2] = torch.zeros_like(args[2])
    out, routed, shared, _, _ = _verify(solve, args)
    assert torch.equal(out, shared) and torch.count_nonzero(routed) == 0
    assert torch.count_nonzero(shared) > 0


def check_zero_shared_keeps_route(solve):
    args = list(_case())
    args[8] = torch.zeros_like(args[8])
    out, routed, shared, _, _ = _verify(solve, args)
    assert torch.equal(out, routed) and torch.count_nonzero(shared) == 0


def check_weights_not_renormalized(solve):
    args = list(_case())
    original = _verify(solve, args)
    args[2] = args[2] * 2
    changed = _verify(solve, args)
    torch.testing.assert_close(changed[1], original[1] * 2)
    torch.testing.assert_close(changed[2], original[2])


def check_selected_slot_permutation(solve):
    args = list(_case(k=3))
    original = _verify(solve, args)
    args[1] = args[1].flip(1)
    args[2] = args[2].flip(1)
    changed = _verify(solve, args)
    for a, b in zip(original[:3], changed[:3]):
        torch.testing.assert_close(a, b)


def check_all_experts_selected(solve):
    out = _verify(solve, _case(e=5, k=5))
    assert out[3] == out[4]


def check_single_expert_and_dimensions(solve):
    _verify(solve, _case(t=1, e=1, d=1, h=1, hs=1, k=1))


def check_different_shared_width(solve):
    for h, hs in ((1, 7), (7, 1), (3, 3)):
        _verify(solve, _case(h=h, hs=hs))


def check_unselected_experts_do_not_contribute(solve):
    args = list(_case(e=5, k=1))
    args[1] = torch.zeros_like(args[1])
    original = _verify(solve, args)
    for i in (3, 4, 5):
        args[i] = args[i].clone()
        args[i][1:] = 3.0
    changed = _verify(solve, args)
    assert torch.equal(original[0], changed[0])


def check_active_count_not_batch_union(solve):
    args = list(_case(t=4, e=4, k=1))
    args[1] = torch.arange(4).reshape(4, 1)
    _, _, _, total, active = _verify(solve, args)
    assert active == 3 * 3 * (2 + 5) and total == 3 * 3 * (4 * 2 + 5)


def check_noncontiguous_inputs(solve):
    args = []
    for tensor in _case(t=4, e=5, k=3):
        buffer = torch.empty((*tensor.shape, 2), dtype=tensor.dtype)
        buffer[..., 0] = tensor
        args.append(buffer[..., 0])
    assert all(not t.is_contiguous() for t in args)
    _verify(solve, args)


def check_zero_input_no_residual_or_bias(solve):
    args = list(_case())
    args[0] = torch.zeros_like(args[0])
    result = _verify(solve, args)
    assert all(torch.count_nonzero(t) == 0 for t in result[:3])


def check_randomized_independent_scalar_oracle(solve):
    for seed in range(30):
        e = 1 + seed % 8
        _verify(solve, _case(t=1 + seed % 8, e=e, d=1 + seed % 6,
                            h=1 + seed % 7, hs=1 + (seed * 3) % 7,
                            k=1 + seed % e, seed=seed))


def check_maximum_shapes(solve):
    _verify(solve, _case(t=8, e=8, d=8, h=8, hs=8, k=8))


check_example.example = True
check_zero_route_keeps_shared.example = True
