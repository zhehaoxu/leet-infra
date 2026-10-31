"""Independent pure-Python FastV oracle: scalar head-mean loop plus manual sorting."""

import torch


def _keep_count(image_len, r):
    return min(max(round(image_len * (1.0 - float(r))), 0), image_len)


def _oracle(hidden_states, attention, image_start, image_len, r):
    """Score, select and gather with Python lists; no torch sorting, topk or mean."""
    values = hidden_states.tolist()
    weights = attention.tolist()
    batch, seq_length, feature = len(values), len(values[0]), len(values[0][0])
    heads = len(weights[0])
    keep_count = _keep_count(image_len, r)

    pruned_rows, keep_rows, score_rows = [], [], []
    for row in range(batch):
        scores = []
        for offset in range(image_len):
            column = image_start + offset
            total = 0.0
            for head in range(heads):  # scalar accumulation, not tensor mean()
                total += weights[row][head][seq_length - 1][column]
            scores.append(total / heads)
        score_rows.append(scores)
        # ties: score descending then index ascending; finally restore index order
        ranked = sorted(range(image_len), key=lambda index: (-scores[index], index))
        keep = (
            list(range(image_start))
            + sorted(index + image_start for index in ranked[:keep_count])
            + list(range(image_start + image_len, seq_length))
        )
        keep_rows.append(keep)
        pruned_rows.append([values[row][index] for index in keep])

    keep_tensor = torch.tensor(keep_rows, dtype=torch.int64)
    pruned = torch.tensor(pruned_rows, dtype=hidden_states.dtype)
    return (
        pruned.reshape(batch, len(keep_rows[0]), feature),
        keep_tensor,
        keep_tensor.clone(),
        torch.tensor(score_rows, dtype=hidden_states.dtype),
    )


def _case(b=2, seq=12, image_len=6, image_start=2, heads=3, feature=4, r=0.5,
          dtype=torch.float64, seed=7):
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn((b, seq, feature), dtype=dtype, generator=generator) * 0.7
    attention = torch.rand((b, heads, seq, seq), dtype=dtype, generator=generator) + 0.05
    return (hidden_states, attention, image_start, image_len, r)


def _example():
    """Prefix 0-1, visual span 2-5, suffix 6-7; three-way tie inside the span."""
    hidden_states = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0],
          [5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]],
        dtype=torch.float64,
    )
    attention = torch.full((1, 2, 8, 8), 0.1, dtype=torch.float64)
    attention[:, :, -1, :] = torch.tensor(
        [[0.1, 0.1, 0.4, 0.2, 0.8, 0.6, 0.1, 0.1],
         [0.1, 0.1, 0.6, 0.2, 0.2, 0.4, 0.1, 0.1]],
        dtype=torch.float64,
    )
    return (hidden_states, attention, 2, 4, 0.5)


def _batch_example():
    hidden_states = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0], [5.0]],
         [[10.0], [20.0], [30.0], [40.0], [50.0]]],
        dtype=torch.float64,
    )
    attention = torch.zeros((2, 1, 5, 5), dtype=torch.float64)
    attention[0, 0, -1, :] = torch.tensor([0.2, 0.9, 0.1, 0.5, 0.3], dtype=torch.float64)
    attention[1, 0, -1, :] = torch.tensor([0.2, 0.3, 0.7, 0.4, 0.1], dtype=torch.float64)
    return (hidden_states, attention, 1, 3, 0.5)


def _close(actual, expected):
    tolerance = 3e-5 if expected.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _verify(solve, args):
    hidden_states, attention, image_start, image_len, r = args
    expected = _oracle(*args)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 4
    batch, seq_length, feature = hidden_states.shape
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == target.shape and actual.dtype == target.dtype
        assert actual.device == hidden_states.device
        assert torch.isfinite(actual).all()
    pruned, keep_indexs, position_ids, score = result
    assert pruned.shape == (batch, seq_length - (image_len - _keep_count(image_len, r)), feature)
    assert keep_indexs.dtype == torch.int64 and score.dtype == hidden_states.dtype
    assert torch.equal(pruned, expected[0]), "surviving rows must be exact untouched copies"
    assert torch.equal(keep_indexs, expected[1]) and torch.equal(position_ids, expected[2])
    _close(score, expected[3])
    assert (keep_indexs[:, 1:] > keep_indexs[:, :-1]).all(), "keep_indexs must ascend"
    assert torch.equal(position_ids, keep_indexs)
    for row in range(batch):
        indices = keep_indexs[row].tolist()
        suffix = list(range(image_start + image_len, seq_length))
        visual = indices[image_start:len(indices) - len(suffix)]
        assert indices[:image_start] == list(range(image_start)), "prefix must survive"
        assert indices[len(indices) - len(suffix):] == suffix, "suffix must survive"
        assert all(image_start <= index < image_start + image_len for index in visual)
    return result


def check_example_head_mean_and_tie(solve):
    pruned, keep_indexs, position_ids, score = _verify(solve, _example())
    assert keep_indexs.tolist() == [[0, 1, 2, 4, 6, 7]]
    assert position_ids.tolist() == [[0, 1, 2, 4, 6, 7]]
    _close(score, torch.tensor([[0.5, 0.2, 0.5, 0.5]], dtype=torch.float64))
    assert pruned.tolist()[0] == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],
                                  [5.0, 50.0], [7.0, 70.0], [8.0, 80.0]]


def check_example_rows_select_independently(solve):
    pruned, keep_indexs, position_ids, score = _verify(solve, _batch_example())
    assert keep_indexs.tolist() == [[0, 1, 3, 4], [0, 2, 3, 4]]
    assert position_ids.tolist() == [[0, 1, 3, 4], [0, 2, 3, 4]]
    _close(score, torch.tensor([[0.9, 0.1, 0.5], [0.3, 0.7, 0.4]], dtype=torch.float64))
    assert pruned.tolist() == [[[1.0], [2.0], [4.0], [5.0]],
                               [[10.0], [30.0], [40.0], [50.0]]]


def check_bankers_rounding_boundaries(solve):
    table = {(4, 0.5): 2, (3, 0.5): 2, (5, 0.5): 2, (2, 0.5): 1,
             (2, 0.75): 0, (6, 0.75): 2, (4, 0.25): 3, (3, 0.6): 1}
    for (image_len, r), keep_count in table.items():
        assert _keep_count(image_len, r) == keep_count, (image_len, r)
        args = _case(b=1, seq=image_len + 4, image_len=image_len, image_start=2, r=r)
        pruned, _, _, _ = _verify(solve, args)
        assert pruned.shape[1] == args[0].shape[1] - (image_len - keep_count)


def check_zero_ratio_is_identity(solve):
    args = _case(r=0.0)
    pruned, keep_indexs, position_ids, _ = _verify(solve, args)
    assert torch.equal(pruned, args[0])
    expected = torch.arange(args[0].shape[1], dtype=torch.int64).expand_as(keep_indexs)
    assert torch.equal(keep_indexs, expected) and torch.equal(position_ids, expected)


def check_all_visual_tokens_pruned(solve):
    # image_len=2 with r=0.75 gives round(0.5) == 0: the whole span disappears.
    args = _case(b=2, seq=8, image_len=2, image_start=3, heads=2, feature=4, r=0.75)
    pruned, keep_indexs, position_ids, score = _verify(solve, args)
    assert pruned.shape == (2, 6, 4)
    assert keep_indexs.tolist() == [[0, 1, 2, 5, 6, 7]] * 2
    assert torch.equal(position_ids, keep_indexs)
    assert score.shape == (2, 2) and not torch.equal(pruned, args[0])


def check_prefix_and_suffix_survive(solve):
    args = _case(b=2, seq=14, image_len=8, image_start=3, heads=2, r=0.9)
    assert _keep_count(8, 0.9) == 1
    _, keep_indexs, position_ids, _ = _verify(solve, args)
    for row in range(2):
        indices = keep_indexs[row].tolist()
        assert len(indices) == 7
        assert indices[:3] == [0, 1, 2] and indices[-3:] == [11, 12, 13]
        assert 3 <= indices[3] <= 10
        _close(position_ids[row].to(torch.float64),
               torch.tensor(indices, dtype=torch.float64))


def check_ties_pin_to_lower_index(solve):
    hidden_states, attention, image_start, image_len, r = _case(
        b=1, seq=10, image_len=6, image_start=2, heads=2, feature=3, r=0.5)
    assert _keep_count(image_len, r) == 3
    tied = attention.clone()
    tied[:, :, -1, :] = 0.25  # every column tied, so the lowest indices win
    _, keep_indexs, _, score = _verify(
        solve, (hidden_states, tied, image_start, image_len, r))
    assert keep_indexs.tolist() == [[0, 1, 2, 3, 4, 8, 9]]
    _close(score, torch.full((1, image_len), 0.25, dtype=torch.float64))
    worst = tied.clone()
    worst[:, :, -1, 4] = worst[:, :, -1, 7] = -0.5  # tied last two columns
    pruned, keep_indexs, _, _ = _verify(
        solve, (hidden_states, worst, image_start, image_len, r))
    assert keep_indexs.tolist() == [[0, 1, 2, 3, 5, 8, 9]]
    assert pruned.shape[1] == 7


def check_survivors_keep_gaps(solve):
    args = _case(b=1, seq=11, image_len=7, image_start=1, heads=3, r=0.5)
    assert _keep_count(7, 0.5) == 4
    _, keep_indexs, position_ids, _ = _verify(solve, args)
    assert keep_indexs.shape[1] == 8 and position_ids.shape == keep_indexs.shape
    natural = torch.arange(8, dtype=torch.int64)
    assert not torch.equal(position_ids[0], natural), "renumbering from 0 is wrong"
    assert torch.equal(keep_indexs[0], position_ids[0])
    assert (position_ids[0] >= natural).all() and (position_ids[0] > natural).any(), \
        "gaps must be preserved, never renumbered"


def check_only_last_row_and_head_mean_matter(solve):
    hidden_states, attention, image_start, image_len, r = _example()
    baseline = _verify(solve, (hidden_states, attention, image_start, image_len, r))
    other_rows = attention.clone()
    other_rows[:, :, :-1, :] = 0.01  # every non-final query row is irrelevant
    flipped = attention.clone()
    flipped[:, :, -1, :] = attention.flip(1)[:, :, -1, :]  # head order cannot matter
    for variant in (other_rows, flipped):
        result = _verify(solve, (hidden_states, variant, image_start, image_len, r))
        for first, second in zip(baseline, result):
            if first.dtype == torch.int64:
                assert torch.equal(first, second)
            else:
                _close(first, second)


def check_kept_rows_are_untouched(solve):
    args = _case(b=2, seq=9, image_len=5, image_start=1, heads=2, feature=3, r=0.4)
    assert _keep_count(5, 0.4) == 3
    pruned, keep_indexs, _, _ = _verify(solve, args)
    values = args[0].tolist()
    for row in range(2):
        for column, index in enumerate(keep_indexs[row].tolist()):
            assert pruned[row, column].tolist() == values[row][index]


def check_ascending_reemission(solve):
    hidden_states, attention, image_start, image_len, r = _case(
        b=1, seq=9, image_len=6, image_start=2, heads=1, feature=2, r=1 / 3)
    assert _keep_count(image_len, r) == 4
    attention = attention.clone()
    attention[0, 0, -1, 2:8] = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    _, keep_indexs, _, _ = _verify(solve, (hidden_states, attention, image_start, image_len, r))
    assert keep_indexs.tolist() == [[0, 1, 3, 5, 6, 7, 8]]
    # the naive score-descending order would be 3, 5, 7, 6 inside the visual span
    assert keep_indexs[0, 2:6].tolist() != [3, 5, 7, 6]


def check_float32_matches_float64(solve):
    base = _case(b=2, seq=10, image_len=5, image_start=2, heads=2, feature=3, r=0.4, seed=21)
    wide = _verify(solve, base)
    narrow = _verify(solve, (base[0].to(torch.float32), base[1].to(torch.float32), 2, 5, 0.4))
    assert narrow[0].dtype == torch.float32 and narrow[3].dtype == torch.float32
    assert torch.equal(narrow[1], wide[1]) and torch.equal(narrow[2], wide[2])
    # the same logical data rounded to float32 may only drift by float32 resolution
    for first, second in ((narrow[0], wide[0]), (narrow[3], wide[3])):
        torch.testing.assert_close(first.to(torch.float64), second, rtol=1e-6, atol=1e-6)
    for dtype in (torch.float32, torch.float64):
        _verify(solve, _case(b=2, seq=12, image_len=6, image_start=2, heads=3,
                             r=0.5, dtype=dtype, seed=5))


def check_noncontiguous_views(solve):
    for dtype in (torch.float32, torch.float64):
        hidden_states, attention, image_start, image_len, r = _case(dtype=dtype)
        dense = _verify(solve, (hidden_states, attention, image_start, image_len, r))
        wide = torch.zeros((hidden_states.shape[0], hidden_states.shape[1],
                            hidden_states.shape[2] * 2), dtype=dtype)
        wide[..., ::2] = hidden_states
        wide[..., 1::2] = -hidden_states
        tall = torch.zeros((attention.shape[0], attention.shape[1],
                            attention.shape[2] * 2, attention.shape[3]), dtype=dtype)
        tall[:, :, ::2, :] = attention
        tall[:, :, 1::2, :] = attention * 4.0
        args = (wide[..., ::2], tall[:, :, ::2, :], image_start, image_len, r)
        assert not args[0].is_contiguous() and not args[1].is_contiguous()
        assert args[1].shape == attention.shape and args[0].shape == hidden_states.shape
        for first, second in zip(_verify(solve, args), dense):
            if first.dtype == torch.int64:
                assert torch.equal(first, second)
            else:
                _close(first, second)


def check_inputs_are_not_mutated(solve):
    cases = [_example(), _batch_example(),
             _case(b=3, seq=13, image_len=7, image_start=2, heads=4, r=0.62),
             _case(b=1, seq=16, image_len=11, image_start=5, heads=3, r=0.5)]
    for args in cases:
        copies = [args[0].clone(), args[1].clone()]
        _verify(solve, args)
        assert torch.equal(args[0], copies[0]) and torch.equal(args[1], copies[1])


def check_seeded_oracle_sweep(solve):
    # (batch, seq, image_len, image_start, heads, feature, r, dtype)
    table = [
        (2, 12, 6, 2, 3, 4, 0.5, torch.float64),
        (1, 9, 4, 0, 2, 2, 0.25, torch.float32),
        (3, 14, 7, 5, 4, 1, 0.75, torch.float64),
        (2, 8, 3, 1, 1, 3, 0.1, torch.float32),
        (1, 16, 10, 3, 2, 4, 0.9, torch.float64),
        (3, 11, 5, 2, 3, 2, 0.625, torch.float32),
        (2, 6, 2, 4, 4, 1, 0.33, torch.float64),
        (1, 13, 8, 0, 2, 3, 0.99, torch.float32),
        (3, 10, 6, 1, 1, 4, 0.5, torch.float64),
        (2, 15, 9, 6, 2, 2, 0.45, torch.float32),
    ]
    for index, (b, seq, image_len, image_start, heads, feature, r, dtype) in enumerate(table):
        _verify(solve, _case(b=b, seq=seq, image_len=image_len, image_start=image_start,
                             heads=heads, feature=feature, r=r, dtype=dtype, seed=17 + index))


def check_invalid_inputs(solve):
    hidden_states, attention, image_start, image_len, r = _example()
    broken = [
        (hidden_states, attention, 2, 4, 1.0),
        (hidden_states, attention, 2, 4, -0.1),
        (hidden_states, attention, 2, 4, float("nan")),
        (hidden_states, attention, 2, 4, float("inf")),
        (hidden_states, attention, 2, 4, "0.5"),
        (hidden_states, attention, 2, 4, True),
        (hidden_states, attention, 2, 4, torch.tensor(0.5)),
        (hidden_states, attention, -1, 4, 0.5),
        (hidden_states, attention, 2, 0, 0.5),
        (hidden_states, attention, 2, 4, None),
        (hidden_states, attention, True, 4, 0.5),
        (hidden_states, attention, 2, True, 0.5),
        (hidden_states, attention, 2, 2.0, 0.5),
        (hidden_states, attention, 6, 4, 0.5),
        (hidden_states, attention, 1, 8, 0.5),
        (hidden_states, attention[0, 0], 2, 4, 0.5),
        (hidden_states, attention[:, 0], 2, 4, 0.5),
        (hidden_states, attention[:, :, :7], 2, 4, 0.5),
        (hidden_states, attention[:, :, :, :7], 2, 4, 0.5),
        (hidden_states, attention[:, :0], 2, 4, 0.5),
        (hidden_states.to(torch.float16), attention.to(torch.float16), 2, 4, 0.5),
        (hidden_states.to(torch.int32), attention, 2, 4, 0.5),
        (hidden_states, attention.to(torch.float32), 2, 4, 0.5),
        (hidden_states.float(), hidden_states.float(), 0, 1, 0.5),
        (hidden_states, attention + float("inf"), 2, 4, 0.5),
        (hidden_states * float("nan"), attention, 2, 4, 0.5),
        (hidden_states[:, :1], attention, 2, 4, 0.5),
        (hidden_states[0], attention, 2, 4, 0.5),
        (hidden_states.repeat(2, 1, 1), attention, 2, 4, 0.5),
        ("hidden", attention, 2, 4, 0.5),
    ]
    for args in broken:
        try:
            solve(*args)
        except ValueError:
            continue
        except AssertionError as exc:
            raise AssertionError(f"expected ValueError, got AssertionError: {exc}")
        raise AssertionError(f"expected ValueError for {args[2:]}")


check_example_head_mean_and_tie.example = True
check_example_rows_select_independently.example = True
