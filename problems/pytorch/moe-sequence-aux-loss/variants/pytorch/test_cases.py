import math

import torch


def _oracle(scores, selected_ids, alpha=1.0):
    values, picks = scores.tolist(), selected_ids.tolist()
    b, t, e = scores.shape
    k = selected_ids.shape[-1]
    f, p, losses = [], [], []
    for rows, choices in zip(values, picks):
        fractions = [sum(expert in token for token in choices) * e / (t * k)
                     for expert in range(e)]
        averages = [math.fsum(row[expert] / math.fsum(row) for row in rows) / t
                    for expert in range(e)]
        f.append(fractions)
        p.append(averages)
        losses.append(alpha * math.fsum(x * y for x, y in zip(fractions, averages)))
    seq = math.fsum(losses) / b
    batch = alpha * math.fsum(
        math.fsum(row[i] for row in f) * math.fsum(row[i] for row in p) / (b * b)
        for i in range(e)
    )
    return tuple(torch.tensor(v, dtype=torch.float64) for v in (f, p, losses, seq, batch))


def _case(b=3, t=4, e=5, k=2, seed=17, alpha=0.3):
    rng = torch.Generator().manual_seed(seed)
    scores = torch.rand((b, t, e), generator=rng, dtype=torch.float64) * 0.9 + 0.01
    ids = torch.stack([torch.randperm(e, generator=rng)[:k]
                       for _ in range(b * t)]).reshape(b, t, k)
    return scores, ids, alpha


def _example():
    return (torch.tensor([[[0.9, 0.1], [0.9, 0.1]],
                          [[0.1, 0.9], [0.1, 0.9]]], dtype=torch.float64),
            torch.tensor([[[0], [0]], [[1], [1]]]), 0.1)


def _verify(solve, args):
    before = [t.clone() for t in args[:2]]
    target = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 5
    for actual, expected in zip(result, target):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == torch.float64 and actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    assert torch.equal(args[0], before[0]) and torch.equal(args[1], before[1])
    return result


def check_example_sequence_vs_batch(solve):
    f, p, losses, seq, batch = _verify(solve, _example())
    torch.testing.assert_close(seq, torch.tensor(0.18, dtype=torch.float64))
    torch.testing.assert_close(batch, torch.tensor(0.1, dtype=torch.float64))
    assert not torch.isclose(seq, batch)


def check_single_sequence_matches_batch(solve):
    result = _verify(solve, _case(b=1))
    torch.testing.assert_close(result[3], result[4])


def check_normalize_each_token_not_global(solve):
    scores = torch.tensor([[[0.9, 0.1], [0.001, 0.009]]], dtype=torch.float64)
    args = (scores, torch.tensor([[[0], [0]]]), 1.0)
    result = _verify(solve, args)
    torch.testing.assert_close(result[1], torch.tensor([[0.5, 0.5]], dtype=torch.float64))


def check_row_scale_invariance(solve):
    args = list(_case())
    original = _verify(solve, args)
    args[0] = args[0] * torch.linspace(0.1, 1.0, args[0].shape[1])[None, :, None]
    changed = _verify(solve, args)
    for a, b in zip(original, changed):
        torch.testing.assert_close(a, b)


def check_k_normalization(solve):
    for k in (1, 2, 4):
        f, _, _, _, _ = _verify(solve, _case(e=4, k=k))
        torch.testing.assert_close(f.sum(dim=1), torch.full((3,), 4., dtype=torch.float64))


def check_all_experts_selected(solve):
    f, _, losses, seq, batch = _verify(solve, _case(e=5, k=5, alpha=0.7))
    assert torch.equal(f, torch.ones_like(f))
    torch.testing.assert_close(losses, torch.full((3,), 0.7, dtype=torch.float64))


def check_zero_alpha_keeps_statistics(solve):
    original = _verify(solve, _case())
    zero = _verify(solve, _case(alpha=0.0))
    for a, b in zip(original[:2], zero[:2]):
        assert torch.equal(a, b)
    assert all(torch.count_nonzero(t) == 0 for t in zero[2:])


def check_single_expert(solve):
    f, p, losses, seq, batch = _verify(solve, _case(b=1, t=1, e=1, k=1, alpha=2.0))
    assert f.item() == p.item() == 1.0
    assert losses.item() == seq.item() == batch.item() == 2.0


def check_selections_are_explicit(solve):
    scores = torch.tensor([[[0.9, 0.1], [0.9, 0.1]]], dtype=torch.float64)
    ids = torch.tensor([[[1], [1]]])
    result = _verify(solve, (scores, ids, 1.0))
    torch.testing.assert_close(result[3], torch.tensor(0.2, dtype=torch.float64))


def check_zero_scores_and_unselected_experts(solve):
    scores = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]], dtype=torch.float64)
    ids = torch.tensor([[[2], [2]]])
    result = _verify(solve, (scores, ids, 1.0))
    assert result[3].item() == 0.0


def check_permutations_preserve_losses(solve):
    args = list(_case())
    original = _verify(solve, args)
    args[0] = args[0].flip(1)
    args[1] = args[1].flip(1).flip(2)
    changed = _verify(solve, args)
    for a, b in zip(original, changed):
        torch.testing.assert_close(a, b)


def check_noncontiguous_inputs(solve):
    args = list(_case())
    for j in (0, 1):
        tensor = args[j]
        buffer = torch.empty((*tensor.shape, 2), dtype=tensor.dtype)
        buffer[..., 0] = tensor
        args[j] = buffer[..., 0]
        assert not args[j].is_contiguous()
    _verify(solve, args)


def check_randomized_enumeration(solve):
    for seed in range(32):
        e = 1 + seed % 8
        _verify(solve, _case(b=1 + seed % 4, t=1 + seed % 8,
                            e=e, k=1 + seed % e, seed=seed))


def check_maximum_shape(solve):
    _verify(solve, _case(b=4, t=8, e=8, k=8, alpha=2.0))


def check_default_alpha_and_small_affinities(solve):
    scores, ids, _ = _case()
    scores = scores / scores.sum(dim=-1, keepdim=True) * 1.1e-6
    _verify(solve, (scores, ids))


check_example_sequence_vs_batch.example = True
check_all_experts_selected.example = True
