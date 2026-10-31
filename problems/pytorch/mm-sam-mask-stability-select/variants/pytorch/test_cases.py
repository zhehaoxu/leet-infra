"""Pure-Python pixel counting, bias arithmetic and first-max argmax oracle.

Every value used here is a dyadic rational (integer, quarter or sixteenth), so float32
and float64 store it exactly and the Python threshold comparisons below agree with the
torch ones to the last bit. The oracle never touches a torch tensor operation: it walks
nested lists, counts pixels with plain ifs, divides two Python ints and scans for the
maximum index by hand.
"""

import torch


def _oracle(masks, iou_preds, num_points, mask_threshold=0.0, offset=1.0):
    batches = masks.tolist()
    preds = iou_preds.tolist()
    threshold = float(mask_threshold)
    step = float(offset)
    high = threshold + step
    low = threshold - step
    bias = (float(num_points) - 2.5) * 1000.0

    stability_rows, biased_rows, area_rows = [], [], []
    selected_masks, selected_scores, best_indices, selected_preds = [], [], [], []
    for batch_index, batch in enumerate(batches):
        scores, shifted, areas, grids = [], [], [], []
        for channel, planes in enumerate(batch):
            inter = union = area = 0
            grid = []
            for row in planes:
                line = []
                for value in row:
                    if value > high:
                        inter += 1
                    if value > low:
                        union += 1
                    if value > threshold:
                        area += 1
                        line.append(True)
                    else:
                        line.append(False)
                grid.append(line)
            # Upstream computes 0/0 = nan here; this contract defines an empty union as 0.0.
            score = 0.0 if union == 0 else inter / union
            scores.append(score)
            shifted.append(score + bias if channel == 0 else score)
            areas.append(area)
            grids.append(grid)
        best = 0
        for index in range(1, len(shifted)):
            if shifted[index] > shifted[best]:
                best = index
        stability_rows.append(scores)
        biased_rows.append(shifted)
        area_rows.append(areas)
        selected_masks.append(grids[best])
        selected_scores.append(scores[best])
        best_indices.append(best)
        selected_preds.append(preds[batch_index][best])

    dtype = masks.dtype
    return (torch.tensor(selected_masks, dtype=torch.bool),
            torch.tensor(selected_scores, dtype=dtype),
            torch.tensor(stability_rows, dtype=dtype),
            torch.tensor(biased_rows, dtype=dtype),
            torch.tensor(best_indices, dtype=torch.int64),
            torch.tensor(area_rows, dtype=torch.int64),
            torch.tensor(selected_preds, dtype=dtype))


def _case(b=2, c=3, h=3, w=4, num_points=3, threshold=0.0, offset=1.0,
          dtype=torch.float64, seed=11):
    generator = torch.Generator().manual_seed(seed)
    masks = (torch.randint(-12, 13, (b, c, h, w), generator=generator)
             .to(torch.float64) / 4.0)
    iou_preds = (torch.randint(0, 17, (b, c), generator=generator)
                 .to(torch.float64) / 16.0)
    return masks.to(dtype), iou_preds.to(dtype), num_points, threshold, offset


def _example():
    """Two points: the -500 penalty hides the perfect channel-0 mask."""
    masks = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]],
                           [[1.0, 0.0], [-1.0, 2.0]],
                           [[0.5, 2.0], [2.0, -2.0]]]], dtype=torch.float64)
    iou_preds = torch.tensor([[0.9375, 0.5, 0.875]], dtype=torch.float64)
    return masks, iou_preds, 2, 0.0, 1.0


def _bias_example():
    """Three points: channel 0 wins even where its stability is 0.0 or 0.25."""
    masks = torch.tensor([[[[1.5, -0.5], [-1.5, 0.5]],
                           [[-1.0, -2.0], [-3.0, -1.5]],
                           [[2.0, 2.0], [2.0, 2.0]]],
                          [[[-2.0, -2.0], [-2.0, -2.0]],
                           [[1.0, 1.0], [1.0, 1.0]],
                           [[0.5, -0.5], [2.0, 3.0]]]], dtype=torch.float64)
    iou_preds = torch.tensor([[0.25, 0.5, 0.75], [1.0, 0.0625, 0.5]],
                             dtype=torch.float64)
    return masks, iou_preds, 3, 0.0, 1.0


def _close(actual, expected):
    tolerance = 3e-5 if expected.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _expect_error(solve, args):
    try:
        solve(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError for the patched argument set")


def _verify(solve, args):
    expected = _oracle(*args)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 7
    masks, iou_preds, num_points, threshold, _ = args
    batch, tokens, height, width = masks.shape
    dtype = masks.dtype

    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == target.shape
        assert actual.device == masks.device
    assert result[0].shape == (batch, height, width)
    assert result[1].shape == (batch,)
    assert result[2].shape == result[3].shape == (batch, tokens)
    assert result[4].shape == (batch,)
    assert result[5].shape == (batch, tokens)
    assert result[6].shape == (batch,)
    assert result[0].dtype == torch.bool
    assert result[4].dtype == torch.int64 and result[5].dtype == torch.int64
    for index in (1, 2, 3, 6):
        assert result[index].dtype == dtype
        assert torch.isfinite(result[index]).all()

    assert torch.equal(result[0], expected[0]), "selected mask must match the counting oracle"
    assert torch.equal(result[4], expected[4]), "argmax must match the hand-scan oracle"
    assert torch.equal(result[5], expected[5]), "area must match the hand-count oracle"
    for index in (1, 2, 3, 6):
        _close(result[index], expected[index])

    assert bool(((result[2] >= 0) & (result[2] <= 1)).all())
    assert bool((result[5] <= height * width).all())
    penalty = (float(num_points) - 2.5) * 1000.0
    _close(result[3][:, 0], result[2][:, 0] + penalty)
    if tokens > 1:
        _close(result[3][:, 1:], result[2][:, 1:])
    for row in range(batch):
        best = int(result[4][row])
        _close(result[1][row], result[2][row, best])
        _close(result[6][row], iou_preds[row, best])
        assert int(result[5][row, best]) == int(result[0][row].count_nonzero())
        _close(result[0][row].to(dtype), (masks[row, best] > threshold).to(dtype))
        biased = result[3][row]
        for index in range(best):
            assert float(biased[index]) < float(biased[best]), "argmax must take the first maximum"
        for index in range(best + 1, tokens):
            assert float(biased[index]) <= float(biased[best])
    return result


def check_example_penalty_hides_best_channel(solve):
    selected, score, stability, biased, best, areas, pred = _verify(solve, _example())
    assert best.tolist() == [2]
    assert areas.tolist() == [[4, 2, 3]]
    assert pred.tolist() == [0.875]


def check_example_bias_prefers_first_token(solve):
    selected, score, stability, biased, best, areas, pred = _verify(solve, _bias_example())
    assert best.tolist() == [0, 0]
    assert torch.equal(stability[0], torch.tensor([1.0 / 3.0, 0.0, 1.0],
                                                  dtype=torch.float64))
    assert torch.equal(stability[1], torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64))
    assert areas.tolist() == [[2, 0, 4], [0, 4, 3]]
    assert not bool(selected[1].any()), "an empty selected mask is legal"


def check_num_points_crosses_the_bias_at_three(solve):
    # Channel 0 and channel 1 have the same stability, so only the sign of the bias
    # decides which of them the argmax sees.
    masks = torch.tensor([[[[2.0, 2.0], [0.0, 0.0]],
                           [[2.0, 2.0], [0.0, 0.0]]]], dtype=torch.float64)
    args = (masks, torch.zeros((1, 2), dtype=torch.float64), 1, 0.0, 1.0)
    for num_points in range(1, 13):
        patched = (args[0], args[1], num_points, args[3], args[4])
        result = _verify(solve, patched)
        penalty = (num_points - 2.5) * 1000.0
        assert float(result[3][0, 0]) - float(result[2][0, 0]) == penalty
        assert result[4].tolist() == ([1] if num_points <= 2 else [0])
        assert float(result[3][0, 1]) == float(result[2][0, 1])
        assert float(result[1][0]) == float(result[2][0, int(result[4][0])])


def check_argmax_ties_take_first_index(solve):
    # Equal ratios from different pixel counts: 1/2 for channel 1 and 2/4 for channel 2.
    masks = torch.tensor([[[[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                           [[2.0, 0.0, -2.0, -2.0], [-2.0, -2.0, -2.0, -2.0]],
                           [[2.0, 2.0, 0.0, 0.0], [-2.0, -2.0, -2.0, -2.0]],
                           [[2.0, 0.0, 0.0, -2.0], [2.0, -2.0, -2.0, -2.0]]]],
                         dtype=torch.float64)
    args = (masks, torch.zeros((1, 4), dtype=torch.float64), 2, 0.0, 1.0)
    result = _verify(solve, args)
    _close(result[2][0], torch.tensor([1.0, 0.5, 0.5, 0.5], dtype=torch.float64))
    assert result[5].tolist() == [[8, 1, 2, 2]]
    assert result[4].tolist() == [1], "the first tied index must win"

    # A 1e4 offset empties every intersection, so all stabilities tie at exactly 0.0.
    flat = list(_case(b=2, c=5, h=4, w=4, num_points=2, seed=71))
    flat[4] = 10000.0
    result = _verify(solve, flat)
    assert torch.equal(result[2], torch.zeros((2, 5), dtype=torch.float64))
    assert result[4].tolist() == [1, 1]
    flat[2] = 3
    assert _verify(solve, flat)[4].tolist() == [0, 0]


def check_thresholds_are_strict_greater(solve):
    masks = torch.tensor([[[[-1.0, 0.0], [1.0, 2.0]]]], dtype=torch.float64)
    preds = torch.zeros((1, 1), dtype=torch.float64)
    result = _verify(solve, (masks, preds, 1, 0.0, 1.0))
    # inter = {2.0}, union = {0.0, 1.0, 2.0}, area = {1.0, 2.0}: the -1.0, 0.0 and 1.0
    # pixels sit exactly on a bound and must be excluded by the strict comparison.
    _close(result[2], torch.tensor([[1.0 / 3.0]], dtype=torch.float64))
    assert result[5].tolist() == [[2]]
    assert torch.equal(result[0], torch.tensor([[[False, False], [True, True]]]))
    assert not torch.allclose(result[2], torch.tensor([[0.5]], dtype=torch.float64))

    # Shifting the threshold moves all three bounds with it, still strictly greater.
    result = _verify(solve, (masks, preds, 1, 1.0, 1.0))
    _close(result[2], torch.zeros((1, 1), dtype=torch.float64))
    assert result[5].tolist() == [[1]]
    assert torch.equal(result[0], torch.tensor([[[False, False], [False, True]]]))


def check_all_foreground_and_empty_union(solve):
    masks = torch.tensor([[[[3.0, 3.0], [3.0, 3.0]],
                           [[-3.0, -3.0], [-3.0, -3.0]],
                           [[3.0, -0.5], [-3.0, -3.0]]]], dtype=torch.float64)
    preds = torch.tensor([[0.5, 0.25, 1.0]], dtype=torch.float64)
    result = _verify(solve, (masks, preds, 4, 0.0, 1.0))
    _close(result[2], torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64))
    assert result[5].tolist() == [[4, 0, 1]]
    assert result[4].tolist() == [0]
    assert bool(result[0].all()), "the all-foreground channel wins and stays all True"
    assert torch.isfinite(result[2]).all(), "an empty union must give 0.0, never nan"

    # A tight offset on an all-background mask empties every union and intersection.
    blank = torch.full((1, 2, 3, 3), -3.0, dtype=torch.float64)
    result = _verify(solve, (blank, torch.zeros((1, 2), dtype=torch.float64), 4, 0.0, 0.25))
    assert torch.equal(result[2], torch.zeros((1, 2), dtype=torch.float64))
    assert torch.equal(result[5], torch.zeros((1, 2), dtype=torch.int64))
    assert not bool(result[0].any())
    assert result[4].tolist() == [0]


def check_areas_follow_the_selected_threshold(solve):
    args = list(_case(b=2, c=4, h=3, w=3, num_points=5, threshold=1.0, seed=19))
    original = _verify(solve, args)
    areas, selected = original[5], original[0]
    assert areas.dtype == torch.int64
    assert bool((areas >= 0).all()) and bool((areas <= 9).all())
    for row in range(2):
        assert int(selected[row].count_nonzero()) == int(areas[row, int(original[4][row])])

    # The area ledger follows mask_threshold, not the two stability bounds.
    raised = list(args)
    raised[3] = 1.25
    other = _verify(solve, raised)
    assert not torch.equal(other[5], areas)
    assert bool((other[5] <= areas).all())
    lowered = list(args)
    lowered[3] = 0.75
    wider = _verify(solve, lowered)
    assert bool((wider[5] >= areas).all())


def check_threshold_and_offset_are_free_parameters(solve):
    base = list(_case(b=2, c=4, h=4, w=4, num_points=2, seed=37))
    previous = None
    for step in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        args = list(base)
        args[4] = step
        stability = _verify(solve, args)[2]
        assert bool((stability <= 1.0).all())
        if previous is not None:
            assert bool((stability <= previous + 1e-12).all()), "wider offset cannot help"
        previous = stability

    for threshold in (0.0, 0.5, -0.75, 1.5, -3.0):
        args = list(base)
        args[3] = threshold
        _verify(solve, args)
        shifted = list(args)
        shifted[3] = threshold + 8.0
        assert bool((_verify(solve, shifted)[5] == 0).all())


def check_per_batch_rows_are_independent(solve):
    args = list(_case(b=3, c=4, h=3, w=4, num_points=4, seed=43))
    full = _verify(solve, args)
    for row in range(3):
        single = _verify(solve, (args[0][row:row + 1], args[1][row:row + 1],
                                 args[2], args[3], args[4]))
        assert torch.equal(single[0][0], full[0][row])
        assert torch.equal(single[4][0], full[4][row])
        assert torch.equal(single[5][0], full[5][row])
        for index in (1, 2, 3, 6):
            _close(single[index][0], full[index][row])

    reordered = _verify(solve, (args[0].flip(0), args[1].flip(0), args[2], args[3], args[4]))
    assert torch.equal(reordered[4], full[4].flip(0))
    assert torch.equal(reordered[5], full[5].flip(0))
    assert torch.equal(reordered[0], full[0].flip(0))


def check_iou_preds_are_a_passive_ledger(solve):
    args = list(_case(b=2, c=5, h=4, w=4, num_points=3, seed=53))
    original = _verify(solve, args)
    for replacement in (torch.zeros_like(args[1]), torch.full_like(args[1], 9997.5),
                        args[1].flip(-1), args[1] * -1.0 + 0.0625):
        patched = list(args)
        patched[1] = replacement.to(args[1].dtype)
        result = _verify(solve, patched)
        for index in (0, 2, 3, 4, 5):
            assert torch.equal(result[index], original[index])
        for row in range(2):
            _close(result[6][row], replacement[row, int(result[4][row])])


def check_minimal_shapes_and_single_token(solve):
    # A single mask token: the bias is applied but the argmax has nothing to choose.
    for num_points in (1, 2, 3, 12):
        args = list(_case(b=1, c=1, h=1, w=1, num_points=num_points, seed=61))
        result = _verify(solve, args)
        assert result[4].tolist() == [0]
        assert bool(((result[2] == 0.0) | (result[2] == 1.0)).all())
        assert bool(((result[5] == 0) | (result[5] == 1)).all())

    # Two identical channels: the penalised channel 0 can never win at 1 or 2 points.
    for value in (0.0, 0.25, -0.25, 1.0, 4.0):
        args = list(_case(b=1, c=2, h=1, w=1, num_points=2, seed=67))
        args[0] = torch.full((1, 2, 1, 1), value, dtype=torch.float64)
        result = _verify(solve, args)
        assert torch.equal(result[2][0, 0], result[2][0, 1])
        assert result[4].tolist() == [1]


def check_float32_and_float64_outputs(solve):
    for dtype in (torch.float32, torch.float64):
        args = list(_case(b=3, c=4, h=3, w=5, num_points=6, dtype=dtype, seed=73))
        result = _verify(solve, args)
        assert result[1].dtype == dtype and result[2].dtype == dtype
        assert result[3].dtype == dtype and result[6].dtype == dtype
        assert result[0].dtype == torch.bool
        assert result[4].dtype == torch.int64 and result[5].dtype == torch.int64
        assert all(tensor.device.type == "cpu" for tensor in result)
        _verify(solve, list(_case(b=1, c=1, h=1, w=1, num_points=1, dtype=dtype, seed=79)))
        _verify(solve, _case(b=2, c=2, h=2, w=2, dtype=dtype, threshold=0.5,
                             offset=0.25, num_points=9, seed=83))


def check_noncontiguous_inputs_are_not_mutated(solve):
    for dtype in (torch.float32, torch.float64):
        args = list(_case(b=2, c=3, h=4, w=5, num_points=4, dtype=dtype, seed=89))
        masks, preds = args[0], args[1]
        args[0] = torch.stack((masks, torch.zeros_like(masks)), dim=-1)[..., 0]
        args[1] = torch.stack((preds, torch.zeros_like(preds)), dim=-1)[..., 0]
        assert not args[0].is_contiguous() and not args[1].is_contiguous()
        assert torch.equal(args[0], masks) and torch.equal(args[1], preds)
        copies = (args[0].clone(), args[1].clone())
        _verify(solve, args)
        for tensor, copy in zip(args[:2], copies):
            assert torch.equal(tensor, copy), "input mutated"

        # A transposed view of the same data is a second noncontiguous layout.
        transposed = (masks.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2), preds)
        assert not transposed[0].is_contiguous()
        _verify(solve, transposed + tuple(args[2:]))


def check_seeded_shape_sweep(solve):
    shapes = ((1, 1, 1, 1, 1, 0.0, 1.0), (1, 2, 1, 3, 2, 0.0, 0.25),
              (2, 3, 2, 2, 3, 1.0, 1.0), (2, 4, 3, 5, 4, -1.0, 2.0),
              (3, 5, 4, 4, 5, 0.0, 0.5), (4, 5, 8, 8, 12, 0.25, 1.0),
              (1, 3, 8, 1, 7, -2.0, 3.0), (3, 1, 1, 8, 11, 0.0, 4.0),
              (2, 2, 5, 7, 1, 1.5, 0.125), (4, 3, 2, 6, 2, -0.5, 1.5),
              (3, 4, 6, 3, 6, 0.0, 8.0), (2, 5, 4, 2, 9, 2.5, 0.5))
    for index, (b, c, h, w, points, threshold, step) in enumerate(shapes):
        dtype = torch.float32 if index % 2 else torch.float64
        _verify(solve, _case(b=b, c=c, h=h, w=w, num_points=points, threshold=threshold,
                             offset=step, dtype=dtype, seed=100 + 7 * index))
        flipped = list(_case(b=b, c=c, h=h, w=w, num_points=points,
                             threshold=-threshold, offset=step,
                             dtype=dtype, seed=200 + 11 * index))
        flipped[0] = -flipped[0]
        _verify(solve, flipped)


def check_invalid_inputs(solve):
    base = list(_case(b=2, c=3, h=3, w=3, num_points=3))
    for replacement in (
            (0, torch.zeros((2, 3, 3), dtype=torch.float64)),      # masks must be 4-D
            (0, torch.zeros((2, 3, 3, 3, 3), dtype=torch.float64)),
            (0, torch.zeros((0, 3, 3, 3), dtype=torch.float64)),   # empty batch
            (0, torch.zeros((2, 3, 0, 3), dtype=torch.float64)),   # empty spatial axis
            (0, torch.zeros((2, 3, 3, 3), dtype=torch.int64)),     # masks must be float
            (0, [[1.0] * 24]),                                     # not a tensor
            (1, torch.zeros((2, 2), dtype=torch.float64)),         # token count mismatch
            (1, torch.zeros((3, 3), dtype=torch.float64)),         # batch mismatch
            (1, torch.zeros(6, dtype=torch.float64)),              # not 2-D
            (1, torch.zeros((2, 3), dtype=torch.float32)),         # dtype mismatch
            (1, torch.zeros((2, 3), dtype=torch.int64)),           # preds must be float
            (1, [[0.5] * 6]),                                      # not a tensor
    ):
        args = list(base)
        args[replacement[0]] = replacement[1]
        _expect_error(solve, args)

    for value in (float("nan"), float("inf"), -float("inf")):
        for index in (0, 1):
            args = list(base)
            args[index] = args[index].clone()
            args[index].reshape(-1)[0] = value
            _expect_error(solve, args)

    for num_points in (0, -1, -5, 3.0, 2.5, True, False, None, "3"):
        args = list(base)
        args[2] = num_points
        _expect_error(solve, args)

    for threshold in (True, False, None, "0", float("nan"), float("inf")):
        args = list(base)
        args[3] = threshold
        _expect_error(solve, args)

    for step in (0, 0.0, -1.0, -0.5, True, None, "1", float("nan"), float("inf")):
        args = list(base)
        args[4] = step
        _expect_error(solve, args)


check_example_penalty_hides_best_channel.example = True
check_example_bias_prefers_first_token.example = True
