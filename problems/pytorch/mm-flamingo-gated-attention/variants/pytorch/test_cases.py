"""Per-token, selected-media SDPA oracle; no global masked-softmax oracle."""

import math

import torch
import torch.nn.functional as F


def _normalize_token(token, eps):
    centered = token - token.mean()
    return centered / (centered.square().mean() + eps).sqrt()


def _oracle(x, media, media_locations, wq, wk, wv, wo, gate, n_heads, eps=1e-5):
    b, t, d = x.shape
    m, k = media.shape[1:3]
    width = d // n_heads
    out = x.clone()
    attention = torch.zeros((b, n_heads, t, m * k), dtype=x.dtype)
    visible = torch.zeros((b, t, m * k), dtype=torch.bool)
    for batch in range(b):
        media_count = 0
        for token in range(t):
            # Count markers procedurally; a marker sees its own media immediately.
            if bool(media_locations[batch, token]):
                media_count += 1
            if media_count == 0:
                continue  # No SDPA invocation, and exact zero probabilities/output branch.
            start = (media_count - 1) * k
            visible[batch, token, start:start + k] = True
            selected = media[batch, media_count - 1]
            query = F.linear(_normalize_token(x[batch, token], eps), wq)
            keys, values = F.linear(selected, wk), F.linear(selected, wv)
            pieces = []
            for head in range(n_heads):
                span = slice(head * width, (head + 1) * width)
                qh, kh, vh = query[span], keys[:, span], values[:, span]
                pieces.append(F.scaled_dot_product_attention(
                    qh[None, None, None], kh[None, None], vh[None, None], dropout_p=0.0
                )[0, 0, 0])
                logits = (kh * qh).sum(dim=-1) / math.sqrt(width)
                numerators = (logits - logits.max()).exp()
                attention[batch, head, token, start:start + k] = numerators / numerators.sum()
            out[batch, token] = x[batch, token] + math.tanh(gate) * F.linear(torch.cat(pieces), wo)
    return out, attention, visible


def _case(b=2, t=6, m=3, k=3, d=6, h=2, dtype=torch.float64, seed=19, gate=0.7, eps=1e-5):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(shape, dtype=dtype, generator=generator) * 0.6

    locations = torch.zeros((b, t), dtype=torch.bool)
    for batch in range(b):
        candidates = list(range(batch % 2, t, 2))[:m]
        locations[batch, candidates] = True
    return (rand(b, t, d), rand(b, m, k, d), locations,
            rand(d, d), rand(d, d), rand(d, d), rand(d, d), gate, h, eps)


def _example():
    x = torch.tensor([[[1., -1.], [0., 0.], [0., 0.], [0., 0.]]], dtype=torch.float64)
    media = torch.tensor([[[[2., 0.], [0., 2.]], [[4., 0.], [0., 4.]],
                           [[9., 9.], [9., 9.]]]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    return (x, media, torch.tensor([[False, True, False, True]]),
            identity.clone(), identity.clone(), identity.clone(), identity.clone(), math.log(3) / 2, 1)


def _negative_example():
    x = torch.tensor([[[2., -2.], [0., 0.]]], dtype=torch.float64)
    media = torch.tensor([[[[2., 4.]]]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    return (x, media, torch.tensor([[False, True]]),
            identity.clone(), identity.clone(), identity.clone(), identity.clone(), -math.log(3) / 2, 1)


def _close(actual, expected):
    tolerance = 4e-5 if expected.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _verify(solve, args):
    expected = _oracle(*args)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 3
    x, media = args[:2]
    b, t, _ = x.shape
    m, k, h = media.shape[1], media.shape[2], args[8]
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == target.shape and actual.dtype == target.dtype
        assert actual.device == x.device
        if target.dtype == torch.bool:
            assert torch.equal(actual, target)
        else:
            assert torch.isfinite(actual).all()
            _close(actual, target)
    out, attention, visible = result
    assert attention.shape == (b, h, t, m * k)
    assert (attention >= 0).all()
    invalid = (~expected[2])[:, None].expand_as(attention)
    assert torch.count_nonzero(attention[invalid]) == 0
    sums = expected[2].any(-1)[:, None].expand(b, h, t).to(x.dtype)
    _close(attention.sum(-1), sums)
    before_first = ~expected[2].any(-1)
    assert torch.equal(out[before_first], x[before_first]), "pre-media residual must be exact"
    return result


def check_example_immediate_media(solve):
    _verify(solve, _example())


def check_example_negative_gate(solve):
    _verify(solve, _negative_example())


def check_zero_gate_exact_identity(solve):
    for dtype in (torch.float32, torch.float64):
        args = _case(dtype=dtype, gate=0.0)
        out, attention, visible = _verify(solve, args)
        assert torch.equal(out, args[0])
        assert attention.sum() > 0 and visible.any()


def check_positive_negative_scalar_gate(solve):
    args = list(_case(gate=0.8))
    positive = _verify(solve, args)
    args[7] = -0.8
    negative = _verify(solve, args)
    _close(positive[0] + negative[0], 2 * args[0])
    _close(positive[1], negative[1])
    assert not torch.allclose(positive[0], args[0])
    for gate in (-20.0, 20.0, 1e-6, -1e-6):
        args[7] = gate
        _verify(solve, args)


def check_marker_is_visible_immediately(solve):
    args = list(_case(b=1, t=5, m=3, k=2, d=4, h=2))
    args[2] = torch.tensor([[False, True, True, False, True]])
    _, attention, visible = _verify(solve, args)
    for token, media_index in ((1, 0), (2, 1), (3, 1), (4, 2)):
        assert visible[0, token, 2 * media_index:2 * media_index + 2].all()
        _close(attention[0, :, token, 2 * media_index:2 * media_index + 2].sum(-1),
               torch.ones(2, dtype=args[0].dtype))


def check_no_markers_exact_zero(solve):
    args = list(_case(b=3))
    args[2].fill_(False)
    out, attention, visible = _verify(solve, args)
    assert torch.equal(out, args[0])
    assert torch.count_nonzero(attention) == 0
    assert not visible.any()


def check_batch_dependent_prefixes(solve):
    args = list(_case(b=3, t=6, m=3))
    args[2] = torch.tensor([[False, False, True, False, True, False],
                            [True, False, False, True, False, True],
                            [False, False, False, False, False, False]])
    _verify(solve, args)


def check_previous_media_cannot_leak(solve):
    args = list(_case(b=1, t=6, m=3, k=3))
    args[2] = torch.tensor([[True, False, True, False, True, False]])
    original = _verify(solve, args)
    changed = list(args)
    changed[1] = args[1].clone()
    changed[1][:, 0] = -3 * changed[1][:, 0] + 4
    perturbed = _verify(solve, changed)
    _close(perturbed[0][:, 2:], original[0][:, 2:])
    _close(perturbed[1][:, :, 2:], original[1][:, :, 2:])
    assert not torch.allclose(perturbed[0][:, :2], original[0][:, :2])


def check_unreferenced_media_has_no_effect(solve):
    args = list(_case(b=2, t=5, m=4))
    args[2] = torch.tensor([[False, True, False, True, False],
                            [False, False, True, False, False]])
    original = _verify(solve, args)
    changed = list(args)
    changed[1] = args[1].clone()
    changed[1][0, 2:] = -3 * changed[1][0, 2:] + 5
    changed[1][1, 1:] = 2 * changed[1][1, 1:] - 4
    perturbed = _verify(solve, changed)
    _close(perturbed[0], original[0])
    _close(perturbed[1], original[1])
    assert torch.equal(perturbed[2], original[2])


def check_raw_media_not_normalized(solve):
    args = list(_case(b=1, t=2, m=1, k=1, d=3, h=3, gate=0.5))
    args[0].zero_()
    args[1] = torch.tensor([[[[2., 2., 2.]]]], dtype=torch.float64)
    args[2] = torch.tensor([[True, False]])
    args[5] = torch.eye(3, dtype=torch.float64)
    args[6] = torch.eye(3, dtype=torch.float64)
    out, _, _ = _verify(solve, args)
    _close(out, torch.full_like(out, 2 * math.tanh(0.5)))


def check_token_ln_eps_and_nonsymmetric_weights(solve):
    args = list(_case(b=2, d=8, h=4, eps=0.3))
    for weight in args[3:7]:
        assert not torch.equal(weight, weight.T)
    original = _verify(solve, args)
    changed = list(args)
    offset = torch.arange(6, dtype=args[0].dtype).reshape(1, 6, 1) - 2
    changed[0] = args[0] + offset
    shifted = _verify(solve, changed)
    _close(shifted[0], original[0] + offset)
    _close(shifted[1], original[1])
    default = list(args)
    default[-1] = 1e-5
    other = _verify(solve, default)
    assert not torch.allclose(other[0], original[0])


def check_finite_extreme_logits(solve):
    for dtype in (torch.float32, torch.float64):
        args = list(_case(b=2, t=4, m=2, k=2, d=2, h=1, dtype=dtype, gate=20.0))
        args[0] = torch.tensor([1., -1.], dtype=dtype).expand(2, 4, 2).clone()
        args[1] = torch.tensor([[[[1000., 1.], [-1000., -1.]],
                                 [[-1000., 2.], [1000., -2.]]]], dtype=dtype).expand(2, 2, 2, 2).clone()
        args[2] = torch.tensor([[False, True, False, True], [False, False, False, False]])
        args[3] = torch.diag(torch.tensor([math.sqrt(2 * (1 + 1e-5)), 0.], dtype=dtype))
        args[4] = torch.eye(2, dtype=dtype)
        args[5] = torch.tensor([[0., 1.], [0., -1.]], dtype=dtype)
        args[6] = torch.eye(2, dtype=dtype)
        out, attention, _ = _verify(solve, args)
        assert torch.isfinite(out).all() and torch.isfinite(attention).all()
        assert attention[0, 0, 1, 0] == 1 and attention[0, 0, 3, 3] == 1


def check_minimal_and_single_feature(solve):
    for marked in (False, True):
        args = list(_case(b=1, t=1, m=1, k=1, d=1, h=1))
        args[2].fill_(marked)
        _verify(solve, args)
    _verify(solve, _case(b=2, t=5, m=3, k=4, d=1, h=1))


def check_zero_projection_residual(solve):
    for index in (5, 6):
        args = list(_case())
        args[index].zero_()
        out, attention, _ = _verify(solve, args)
        assert torch.equal(out, args[0])
        assert attention.sum() > 0


def check_float32_float64_multihead(solve):
    for dtype in (torch.float32, torch.float64):
        for h in (1, 2, 3, 6):
            _verify(solve, _case(b=3, t=7, m=4, k=5, d=6, h=h, dtype=dtype))
        _verify(solve, _case(b=3, t=7, m=4, k=5, d=12, h=4, dtype=dtype))


def check_noncontiguous_no_mutation(solve):
    for dtype in (torch.float32, torch.float64):
        args = list(_case(dtype=dtype))
        for index in range(7):
            tensor = args[index]
            args[index] = torch.stack((tensor, torch.zeros_like(tensor)), dim=-1)[..., 0]
            assert not args[index].is_contiguous()
        copies = [tensor.clone() for tensor in args[:7]]
        _verify(solve, args)
        for tensor, copy in zip(args[:7], copies):
            assert torch.equal(tensor, copy), "input mutated"


def check_seeded_shape_sweep(solve):
    for d in range(1, 13):
        divisors = [h for h in range(1, d + 1) if d % h == 0]
        _verify(solve, _case(b=1 + d % 3, t=1 + d % 7, m=1 + d % 4,
                             k=1 + d % 5, d=d, h=divisors[d % len(divisors)], seed=80 + d,
                             gate=(-1) ** d * 1.3,
                             dtype=torch.float32 if d % 2 else torch.float64))


def check_invalid_heads(solve):
    for h in (0, -2, 4, 7, 2.0, True, None):
        args = list(_case(d=6))
        args[8] = h
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for n_heads={h!r}")


check_example_immediate_media.example = True
check_example_negative_gate.example = True
