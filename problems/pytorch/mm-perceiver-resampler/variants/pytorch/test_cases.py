"""Independent selected-key SDPA oracle plus explicit attention probabilities."""

import math

import torch
import torch.nn.functional as F


def _normalize_rows(x, eps):
    centered = x - x.mean(dim=-1, keepdim=True)
    return centered / (centered.square().mean(dim=-1, keepdim=True) + eps).sqrt()


def _oracle(visual, latents, visual_mask, wq, wk, wv, wo, n_heads, eps=1e-5):
    # Select visible keys first: no global masked-softmax shared with the solution.
    b, k, d = latents.shape
    n = visual.shape[1]
    width = d // n_heads
    probabilities = torch.zeros((b, n_heads, k, n + k), dtype=visual.dtype)
    outputs = []
    for batch in range(b):
        nv = _normalize_rows(visual[batch], eps)
        nl = _normalize_rows(latents[batch], eps)
        indices = torch.cat((visual_mask[batch].nonzero().flatten(), torch.arange(n, n + k)))
        source = torch.cat((nv, nl), dim=0).index_select(0, indices)
        q = F.linear(nl, wq)
        keys, values = F.linear(source, wk), F.linear(source, wv)
        pieces = []
        for head in range(n_heads):
            span = slice(head * width, (head + 1) * width)
            qh, kh, vh = q[:, span], keys[:, span], values[:, span]
            pieces.append(F.scaled_dot_product_attention(
                qh[None, None], kh[None, None], vh[None, None], dropout_p=0.0
            )[0, 0])
            # Probabilities are computed independently, not recovered from SDPA output.
            for query in range(k):
                scores = (kh * qh[query]).sum(dim=-1) / math.sqrt(width)
                numerators = (scores - scores.max()).exp()
                probabilities[batch, head, query, indices] = numerators / numerators.sum()
        outputs.append(latents[batch] + F.linear(torch.cat(pieces, dim=-1), wo))
    return torch.stack(outputs), probabilities


def _case(b=2, n=5, k=3, d=6, h=2, dtype=torch.float64, seed=7, eps=1e-5):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(shape, dtype=dtype, generator=generator) * 0.6

    return (rand(b, n, d), rand(b, k, d),
            torch.rand((b, n), generator=generator) > 0.35,
            rand(d, d), rand(d, d), rand(d, d), rand(d, d), h, eps)


def _example():
    visual = torch.tensor([[[1., -1.], [-1., 1.]]], dtype=torch.float64)
    latents = torch.tensor([[[0., 0.], [1., -1.]]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    return (visual, latents, torch.tensor([[True, False]]),
            identity.clone(), identity.clone(), identity.clone(), identity.clone(), 1)


def _empty_example():
    return (torch.empty((1, 0, 1), dtype=torch.float64),
            torch.tensor([[[3.]]], dtype=torch.float64), torch.empty((1, 0), dtype=torch.bool),
            *(torch.ones((1, 1), dtype=torch.float64) for _ in range(4)), 1)


def _close(actual, expected):
    tolerance = 3e-5 if expected.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _verify(solve, args):
    expected = _oracle(*args)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 2
    visual, latents, mask = args[:3]
    b, k, d = latents.shape
    n, h = visual.shape[1], args[7]
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == latents.dtype and actual.device == latents.device
        assert actual.shape == target.shape
        assert torch.isfinite(actual).all()
        _close(actual, target)
    attention = result[1]
    assert attention.shape == (b, h, k, n + k)
    assert (attention >= 0).all()
    invalid = (~mask)[:, None, None, :].expand(b, h, k, n)
    assert torch.count_nonzero(attention[..., :n][invalid]) == 0
    _close(attention.sum(dim=-1), torch.ones((b, h, k), dtype=latents.dtype))
    return result


def check_example_masked_latents(solve):
    _verify(solve, _example())


def check_example_empty_visual_scalar(solve):
    out, attention = _verify(solve, _empty_example())
    assert torch.equal(out, torch.tensor([[[3.]]], dtype=torch.float64))
    assert torch.equal(attention, torch.ones((1, 1, 1, 1), dtype=torch.float64))


def check_latent_keys_are_included(solve):
    args = list(_case(b=1, n=2, k=3, d=4, h=2))
    args[2].fill_(True)
    args[3].zero_()
    _, attention = _verify(solve, args)
    _close(attention, torch.full_like(attention, 1 / 5))
    assert (attention[..., 2:] > 0).all()


def check_all_visual_masked_keeps_latents(solve):
    args = list(_case(b=2, n=7, k=4, d=6, h=3))
    args[2].fill_(False)
    original = _verify(solve, args)
    changed = list(args)
    changed[0] = -args[0] * 2.0 + 3.0
    new = _verify(solve, changed)
    for first, second in zip(original, new):
        _close(first, second)
    _close(original[1][..., 7:].sum(-1), torch.ones((2, 3, 4), dtype=args[0].dtype))


def check_masked_padding_perturbation(solve):
    args = list(_case(b=2, n=6, k=2, d=8, h=4))
    args[2] = torch.tensor([[True, False, True, False, False, True],
                            [False, True, False, True, True, False]])
    original = _verify(solve, args)
    changed = list(args)
    changed[0] = torch.where(args[2][..., None], args[0], args[0] * -3 + 4)
    perturbed = _verify(solve, changed)
    for first, second in zip(original, perturbed):
        _close(first, second)


def check_empty_visual_multiple_heads(solve):
    for dtype in (torch.float32, torch.float64):
        _verify(solve, _case(b=3, n=0, k=6, d=12, h=4, dtype=dtype))


def check_scalar_features_biased_variance(solve):
    for n in (0, 5):
        args = _case(b=2, n=n, k=3, d=1, h=1)
        out, _ = _verify(solve, args)
        assert torch.equal(out, args[1])


def check_single_latent(solve):
    _verify(solve, _case(b=1, n=12, k=1, d=8, h=2))


def check_last_dimension_ln_and_eps(solve):
    args = list(_case(b=2, n=4, k=3, d=4, h=2, eps=0.25))
    original = _verify(solve, args)
    changed = list(args)
    changed[0] = args[0] + torch.tensor([[[1.], [2.], [-3.], [0.5]]])
    offset = torch.tensor([[[2.], [-1.], [0.5]]], dtype=args[1].dtype)
    changed[1] = args[1] + offset
    shifted = _verify(solve, changed)
    _close(shifted[0], original[0] + offset)
    _close(shifted[1], original[1])
    default = list(args)
    default[-1] = 1e-5
    other = _verify(solve, default)
    assert not torch.allclose(original[0], other[0])


def check_nonsymmetric_weights_and_head_order(solve):
    for h in (1, 2, 3, 6):
        args = list(_case(b=3, n=8, k=4, d=6, h=h, seed=11))
        args[2].fill_(True)
        for weight in args[3:7]:
            assert not torch.equal(weight, weight.T)
        _verify(solve, args)


def check_zero_projection_residual(solve):
    for index in (5, 6):
        args = list(_case())
        args[index].zero_()
        out, attention = _verify(solve, args)
        assert torch.equal(out, args[1])
        assert attention.sum() > 0


def check_float32_and_float64(solve):
    for dtype in (torch.float32, torch.float64):
        _verify(solve, _case(b=3, n=12, k=6, d=12, h=3, dtype=dtype))


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
        _verify(solve, _case(b=1 + d % 3, n=d, k=1 + d % 6, d=d,
                             h=divisors[d % len(divisors)], seed=100 + d,
                             dtype=torch.float32 if d % 2 else torch.float64))


def check_invalid_heads(solve):
    for h in (0, -1, 4, 7, 2.0, True, None):
        args = list(_case(d=6))
        args[7] = h
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for n_heads={h!r}")


check_example_masked_latents.example = True
check_example_empty_visual_scalar.example = True
