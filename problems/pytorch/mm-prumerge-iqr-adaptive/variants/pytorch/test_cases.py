"""Independent pure-Python oracle for the IQR-adaptive kept-token count."""

import math
import random

import numpy as np
import torch


def _f32(x):
    return float(np.float32(x))


def _percentile_linear(values, q):
    """numpy's default linear percentile on a float32 array, transcribed step
    by step in float32 arithmetic (verified bit-exact against np.percentile)."""
    ordered = sorted(values)
    n = len(ordered)
    pos = q / 100.0 * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    a, b = _f32(ordered[lo]), _f32(ordered[hi])
    diff = _f32(b - a)
    # numpy's _lerp switches to b - (b-a)*(1-t) when t >= 0.5 for exactness
    if frac >= 0.5:
        return _f32(b - _f32(diff * (1.0 - frac)))
    return _f32(a + _f32(diff * frac))


def _oracle(cls_attn):
    attention = np.asarray(cls_attn.tolist(), dtype=np.float32).reshape(-1).tolist()
    q1 = _percentile_linear(attention, 25)
    q3 = _percentile_linear(attention, 75)
    upper = q3 + 1.5 * (q3 - q1)
    outliers = sum(1 for value in attention if value > upper)
    ratio = outliers / len(attention)

    rows = cls_attn.tolist()
    batch = len(rows)
    tokens = len(rows[0])
    keep = int(tokens * ratio)
    kept_rows, pruned_rows = [], []
    for row in rows:
        ranked = sorted(range(tokens), key=lambda index: (-row[index], index))
        kept = sorted(ranked[:keep])
        chosen = set(kept)
        pruned = [index for index in range(tokens) if index not in chosen]
        kept_rows.append(kept)
        pruned_rows.append(pruned)
    return ratio, kept_rows, pruned_rows


def _check_partition(solve, cls_attn):
    ratio, kept, pruned = solve(cls_attn)
    exp_ratio, exp_kept, exp_pruned = _oracle(cls_attn)
    assert isinstance(ratio, float), "ratio must be a Python float"
    assert ratio == exp_ratio, f"ratio {ratio} != expected {exp_ratio}"
    assert torch.equal(kept, torch.tensor(exp_kept, dtype=torch.int64)), f"kept {kept}"
    assert torch.equal(pruned, torch.tensor(exp_pruned, dtype=torch.int64)), f"pruned {pruned}"
    assert kept.dtype == torch.int64 and pruned.dtype == torch.int64
    batch, tokens = cls_attn.shape
    assert kept.shape == (batch, kept.shape[1]) and pruned.shape[1] >= 1, "pruned must never be empty"
    for b in range(batch):
        assert kept[b].tolist() == sorted(kept[b].tolist()), "kept must ascend"
        assert pruned[b].tolist() == sorted(pruned[b].tolist()), "pruned must ascend"
        assert set(kept[b].tolist()).isdisjoint(set(pruned[b].tolist()))
        assert sorted(kept[b].tolist() + pruned[b].tolist()) == list(range(tokens))
    return ratio, kept, pruned


def check_constant_flat_zero_ratio(solve):
    attn = torch.full((2, 5), 0.1, dtype=torch.float32)
    ratio, kept, pruned = _check_partition(solve, attn)
    assert ratio == 0.0, "identical values give IQR=0 and no strictly-greater element"
    assert kept.shape == (2, 0), "k=0 keeps nothing"
    assert torch.equal(pruned, torch.arange(5).unsqueeze(0).expand(2, -1))


def check_single_outlier_textbook(solve):
    attn = torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.50]], dtype=torch.float64)
    ratio, kept, pruned = _check_partition(solve, attn)
    assert ratio == 0.125, "only 0.50 exceeds Q3 + 1.5*IQR"
    assert kept.tolist() == [[7]]
    assert pruned.tolist() == [[0, 1, 2, 3, 4, 5, 6]]


def check_flatten_across_batch(solve):
    # seven 0.1s and one 0.9 spread over two rows: ratio is GLOBAL, not per-row
    attn = torch.tensor(
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9]], dtype=torch.float32
    )
    ratio, kept, pruned = _check_partition(solve, attn)
    assert ratio == 0.125
    assert kept.shape == (2, 0), "int(4*0.125)=0 -> both rows keep nothing"
    assert torch.equal(pruned[0], torch.arange(4))
    assert torch.equal(pruned[1], torch.arange(4))


def check_ties_and_shared_ratio(solve):
    attn = torch.tensor(
        [[0.01, 0.02, 0.03, 0.50], [0.02, 0.02, 0.02, 0.02]], dtype=torch.float64
    )
    ratio, kept, pruned = _check_partition(solve, attn)
    assert ratio == 0.25, "0.03 and 0.50 exceed the shared upper bound"
    assert kept.tolist() == [[3], [0]], "row 2 ties resolve to the lowest index"
    assert pruned.tolist() == [[0, 1, 2], [1, 2, 3]]


def _mixed_values(rng, tokens):
    """Mostly a tight bulk plus injected outliers, so kept sizes > 1 are common."""
    return [
        round(rng.uniform(0.6, 1.0), 3) if rng.random() < 0.2 else round(rng.uniform(0.0, 0.4), 3)
        for _ in range(tokens)
    ]


def check_pruned_never_empty(solve):
    generator = random.Random(644)
    for _ in range(40):
        batch = generator.randint(1, 4)
        tokens = generator.randint(1, 24)
        attn = torch.tensor(
            [_mixed_values(generator, tokens) for _ in range(batch)],
            dtype=torch.float64,
        )
        _, _, pruned = _check_partition(solve, attn)
        assert pruned.shape[1] >= 1, "mathematically at least one element sits at or below Q1"


def check_float32_float64_consistency(solve):
    values = [[0.03, 0.11, 0.27, 0.48, 0.52, 0.66, 0.83, 0.97]]
    f64 = torch.tensor(values, dtype=torch.float64)
    f32 = torch.tensor(values, dtype=torch.float32)
    r64, k64, p64 = _check_partition(solve, f64)
    r32, k32, p32 = _check_partition(solve, f32)
    assert r64 == r32, "upstream casts to float32 BEFORE the percentile pass"
    assert torch.equal(k64, k32) and torch.equal(p64, p32)


def check_float32_cast_matters(solve):
    # one nextafter step above 1.0 survives in float64 but collapses to 1.0
    # in float32: the cast MUST happen before the percentile pass
    values = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000000000002]]
    attn = torch.tensor(values, dtype=torch.float64)
    ratio, kept, pruned = _check_partition(solve, attn)
    assert ratio == 0.0, "float32 cast merges the neighbor into the bound"
    assert kept.shape == (1, 0)


def check_noncontiguous_and_no_mutation(solve):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24.0
    attn = base.t().contiguous().t()  # noncontiguous view of the same values
    snapshot = attn.clone()
    _check_partition(solve, attn)
    assert torch.equal(attn, snapshot), "input must not be mutated"


def check_random_against_oracle(solve):
    generator = random.Random(20261031)
    for _ in range(12):
        batch = generator.randint(1, 4)
        tokens = generator.randint(1, 24)
        dtype = torch.float32 if generator.random() < 0.5 else torch.float64
        attn = torch.tensor(
            [_mixed_values(generator, tokens) for _ in range(batch)],
            dtype=dtype,
        )
        _check_partition(solve, attn)


def check_invalid_inputs(solve):
    good = torch.rand(2, 5)
    for bad in (
        torch.rand(5),  # 1-D
        torch.rand(2, 5, 1),  # 3-D
        torch.rand(2, 5).to(torch.int64),  # non-floating dtype
        torch.rand(0, 5),  # empty batch
        torch.rand(2, 0),  # empty tokens
    ):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for shape/dtype {bad.shape}/{bad.dtype}")
    with_nan = good.clone()
    with_nan[1, 3] = float("nan")
    try:
        solve(with_nan)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for NaN")
    with_inf = good.clone()
    with_inf[0, 0] = float("inf")
    try:
        solve(with_inf)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for inf")
    try:
        solve([[0.1, 0.2], [0.3, 0.4]])  # plain list, not a tensor
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-tensor input")
