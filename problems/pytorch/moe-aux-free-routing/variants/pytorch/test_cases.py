import math

import torch


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _oracle(x, gate_w, bias, group_count, top_groups, k, gamma, update_bias=False):
    rows, gw, b = x.tolist(), gate_w.tolist(), bias.tolist()
    t, e = len(rows), len(gw)
    group_size = e // group_count
    weights, indices, scores_all, group_scores_all, groups_all, loads = [], [], [], [], [], [0] * e
    for row in rows:
        logits = [math.fsum(a * c for a, c in zip(row, gw[eid])) for eid in range(e)]
        scores = [_sigmoid(v) for v in logits]
        choice = [s + shift for s, shift in zip(scores, b)]
        group_scores = [
            math.fsum(sorted(choice[g * group_size:(g + 1) * group_size], reverse=True)[:2])
            for g in range(group_count)
        ]
        groups = sorted(range(group_count), key=lambda g: (-group_scores[g], g))[:top_groups]
        allowed = set()
        for g in groups:
            allowed.update(range(g * group_size, (g + 1) * group_size))
        picked = sorted(allowed, key=lambda eid: (-choice[eid], eid))[:k]
        raw = [scores[eid] for eid in picked]
        denom = math.fsum(raw)
        weights.append([v / denom for v in raw])
        indices.append(picked)
        scores_all.append(scores)
        group_scores_all.append(group_scores)
        groups_all.append(groups)
        for eid in picked:
            loads[eid] += 1
    if update_bias:
        average = t * k / e
        new_bias = [start + gamma * (1 if average > load else -1 if average < load else 0)
                    for start, load in zip(b, loads)]
    else:
        new_bias = list(b)
    return (torch.tensor(weights, dtype=torch.float64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(scores_all, dtype=torch.float64),
            torch.tensor(group_scores_all, dtype=torch.float64),
            torch.tensor(groups_all, dtype=torch.int64),
            torch.tensor(loads, dtype=torch.int64),
            torch.tensor(new_bias, dtype=torch.float64))


def _case(t=3, d=2, e=8, groups=2, top_groups=1, k=2, seed=31, gamma=0.1,
          update_bias=True):
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand((t, d), generator=rng, dtype=torch.float64) * 4 - 2
    gw = torch.rand((e, d), generator=rng, dtype=torch.float64) * 2 - 1
    bias = torch.rand((e,), generator=rng, dtype=torch.float64) * 0.4 - 0.2
    return x, gw, bias, groups, top_groups, k, gamma, update_bias


def _example():
    return (torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[1.0], [0.9], [-0.5], [-0.6]], dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64), 2, 1, 2, 0.1, True)


def _verify(solve, args):
    before = [v.clone() for v in args[:3]]
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 7
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == target.dtype and actual.shape == target.shape
        if target.dtype == torch.int64:
            assert torch.equal(actual, target)
        else:
            torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11)
    for actual, original in zip(args[:3], before):
        assert torch.equal(actual, original)
    return result


def check_example_bias_update(solve):
    weights, ids, scores, group_scores, groups, load, new_bias = _verify(solve, _example())
    s = [1 / (1 + math.exp(-v)) for v in (1.0, 0.9, -0.5, -0.6)]
    torch.testing.assert_close(weights, torch.tensor([[s[0] / (s[0] + s[1]),
                                                       s[1] / (s[0] + s[1])]],
                                                     dtype=torch.float64))
    assert ids.tolist() == [[0, 1]] and groups.tolist() == [[0]]
    assert load.tolist() == [1, 1, 0, 0]
    torch.testing.assert_close(new_bias, torch.tensor([-0.1, -0.1, 0.1, 0.1],
                                                      dtype=torch.float64))


def check_bias_selects_but_does_not_weight(solve):
    args = list(_example())
    plain = _verify(solve, args)
    args[2] = args[2].clone()
    args[2][2] += 2.0
    shifted = _verify(solve, args)
    assert shifted[4].tolist() == [[1]]
    assert shifted[1].tolist() == [[2, 3]]
    raw = shifted[2].gather(1, shifted[1])
    torch.testing.assert_close(shifted[0], raw / raw.sum(dim=1, keepdim=True))
    torch.testing.assert_close(plain[2], shifted[2])


def check_group_selection_uses_top2_sum(solve):
    args = (torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[1.0], [0.9], [-0.5], [-0.6]], dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64), 2, 1, 1, 0.0, False)
    result = _verify(solve, args)
    assert result[4].tolist() == [[0]]
    assert result[1].tolist() == [[0]]
    torch.testing.assert_close(result[3], torch.tensor(
        [[0.7310585786300049 + 0.7109495026250040,
          0.3775406687981454 + 0.3543436937742046]], dtype=torch.float64))


def check_all_groups_selected_equals_plain_topk(solve):
    args = list(_case(groups=4, top_groups=4, k=3))
    weights, ids, scores, _, groups, _, _ = _verify(solve, args)
    assert sorted(groups[0].tolist()) == [0, 1, 2, 3]
    expected = torch.argsort(scores + args[2], dim=-1, descending=True, stable=True)[:, :3]
    assert torch.equal(ids, expected)


def check_update_uses_global_batch_load(solve):
    args = list(_case(t=4, e=4, groups=2, top_groups=1, k=1, gamma=0.5))
    args[2] = torch.zeros(4, dtype=torch.float64)
    result = _verify(solve, args)
    average = 4 * 1 / 4
    expected = [0.5 if average > n else -0.5 if average < n else 0.0
                for n in result[5].tolist()]
    torch.testing.assert_close(result[6], torch.tensor(expected, dtype=torch.float64))


def check_update_false_returns_copy(solve):
    args = list(_case(update_bias=False))
    result = _verify(solve, args)
    assert torch.equal(result[6], args[2])
    assert result[6].data_ptr() != args[2].data_ptr()


def check_zero_gamma_keeps_bias(solve):
    args = list(_case(gamma=0.0))
    result = _verify(solve, args)
    assert torch.equal(result[6], args[2])


def check_ties_prefer_lower_group_and_expert(solve):
    args = (torch.tensor([[1.0]], dtype=torch.float64),
            torch.zeros((4, 1), dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64), 2, 1, 2, 0.0, False)
    result = _verify(solve, args)
    assert result[4].tolist() == [[0]]
    assert result[1].tolist() == [[0, 1]]


def check_k_boundaries(solve):
    _verify(solve, _case(groups=2, top_groups=2, k=1))
    _verify(solve, _case(groups=2, top_groups=2, k=8, e=8))
    _verify(solve, _case(groups=1, top_groups=1, k=3, e=3))


def check_negative_bias_and_saturation(solve):
    args = (torch.tensor([[8.0], [-8.0]], dtype=torch.float64),
            torch.tensor([[1.0], [-1.0], [2.0], [-2.0]], dtype=torch.float64),
            torch.tensor([0.5, -0.5, 0.25, -0.25], dtype=torch.float64),
            2, 1, 2, 0.2, True)
    _verify(solve, args)


def check_noncontiguous_inputs(solve):
    args = list(_case())
    for j, tensor in enumerate(args[:3]):
        buffer = torch.empty((*tensor.shape, 2), dtype=tensor.dtype)
        buffer[..., 0] = tensor
        args[j] = buffer[..., 0]
        assert not args[j].is_contiguous()
    _verify(solve, args)


def check_randomized_scalar_oracle(solve):
    for seed in range(36):
        groups = 1 + seed % 4
        size = 2 + seed % 3
        e = groups * size
        top_groups = 1 + seed % groups
        k = 1 + seed % (top_groups * size)
        _verify(solve, _case(t=1 + seed % 6, d=1 + seed % 5, e=e, groups=groups,
                             top_groups=top_groups, k=k, seed=seed,
                             gamma=0.05 * (seed % 4),
                             update_bias=bool(seed % 2)))


def check_invalid_structure(solve):
    base = list(_case(e=8, groups=2, top_groups=1, k=2))
    cases = [
        (1, 0, 1), (1, 1, 0), (0, 1, 1), (3, 1, 1), (2, 1, 9),
    ]
    for groups, top, k in cases:
        args = list(base)
        args[3], args[4], args[5] = groups, top, k
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError((groups, top, k))
    args = list(base)
    args[6] = -0.1
    try:
        solve(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("negative gamma")


def check_maximum_shape(solve):
    _verify(solve, _case(t=8, d=8, e=8, groups=4, top_groups=4, k=8,
                         update_bias=True))


check_example_bias_update.example = True
check_bias_selects_but_does_not_weight.example = True
