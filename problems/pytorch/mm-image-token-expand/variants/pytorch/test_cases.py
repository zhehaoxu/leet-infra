import torch


_KEYS = {"inputs_embeds", "labels", "attention_mask", "position_ids", "mm_token_type_ids"}


def _oracle(text, labels, attention, image_mask, features):
    B, L, D = text.shape
    events = []
    for b in range(B):
        marked = 0
        for j in range(L):
            if bool(image_mask[b, j]):
                if not bool(attention[b, j]):
                    raise ValueError("Image marker in padding")
                marked += 1
        if marked != 1:
            raise ValueError("Expected one image marker")
        row = []
        for j in range(L):
            if not bool(attention[b, j]):
                continue
            if bool(image_mask[b, j]):
                for n in range(features[b].shape[0]):
                    row.append((features[b][n].tolist(), -100, 1))
            else:
                row.append((text[b, j].tolist(), int(labels[b, j]), 0))
        events.append(row)

    S = max(len(row) for row in events)
    out_embeds = [[[0.0 for _ in range(D)] for _ in range(S)] for _ in range(B)]
    out_labels = [[-100 for _ in range(S)] for _ in range(B)]
    out_attention = [[False for _ in range(S)] for _ in range(B)]
    out_positions = [[0 for _ in range(S)] for _ in range(B)]
    out_types = [[0 for _ in range(S)] for _ in range(B)]
    for b, row in enumerate(events):
        for j, (values, label, token_type) in enumerate(row):
            for d in range(D):
                out_embeds[b][j][d] = values[d]
            out_labels[b][j] = label
            out_attention[b][j] = True
            out_positions[b][j] = j
            out_types[b][j] = token_type
    return {
        "inputs_embeds": torch.tensor(out_embeds, dtype=text.dtype),
        "labels": torch.tensor(out_labels, dtype=torch.int64),
        "attention_mask": torch.tensor(out_attention, dtype=torch.bool),
        "position_ids": torch.tensor(out_positions, dtype=torch.int64),
        "mm_token_type_ids": torch.tensor(out_types, dtype=torch.int64),
    }


def _snapshot(args):
    tensors = list(args[:4]) + list(args[4])
    saved = [(t, t.clone(), t.shape, t.stride(), t.storage_offset()) for t in tensors]
    return saved, tuple(args[4])


def _assert_unchanged(args, saved):
    tensors, feature_refs = saved
    assert len(args[4]) == len(feature_refs), "Input feature list length changed"
    assert all(current is original for current, original in zip(args[4], feature_refs))
    for tensor, original, shape, stride, offset in tensors:
        assert tensor.shape == shape and tensor.stride() == stride and tensor.storage_offset() == offset
        assert torch.equal(tensor, original), "Input was mutated"


def _assert_case(solve, *args):
    expected = _oracle(*args)
    saved = _snapshot(args)
    outputs = solve(*args)
    _assert_unchanged(args, saved)
    assert isinstance(outputs, dict) and set(outputs) == _KEYS
    for key in _KEYS:
        value, reference = outputs[key], expected[key]
        assert isinstance(value, torch.Tensor), key
        assert value.device.type == "cpu" and value.dtype == reference.dtype, key
        assert value.shape == reference.shape, key
        assert torch.equal(value, reference), "Incorrect " + key
    return outputs


def _expect_value_error(solve, *args):
    saved = _snapshot(args)
    try:
        solve(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid image markers must raise ValueError")
    _assert_unchanged(args, saved)


def _case_data(valid, positions, counts, D=3, dtype=torch.float32):
    attention = torch.tensor(valid, dtype=torch.bool)
    B, L = attention.shape
    text = torch.arange(B * L * D, dtype=dtype).reshape(B, L, D) / 8
    labels = torch.arange(B * L, dtype=torch.int64).reshape(B, L) + 100
    image_mask = torch.zeros((B, L), dtype=torch.bool)
    for b, position in enumerate(positions):
        if position is not None:
            image_mask[b, position] = True
    features = [
        torch.arange(count * D, dtype=dtype).reshape(count, D) / 8 + (b + 1) * 1000
        for b, count in enumerate(counts)
    ]
    return text, labels, attention, image_mask, features


def check_example(solve):
    text = torch.tensor([
        [[1, 10], [99, 99], [2, 20], [3, 30]],
        [[88, 88], [4, 40], [77, 77], [5, 50]],
    ], dtype=torch.float32)
    labels = torch.tensor([[11, 91, -100, 13], [81, 14, 92, 15]], dtype=torch.int64)
    attention = torch.tensor([[True, True, True, True], [False, True, False, True]])
    image_mask = torch.tensor([[False, True, False, False], [False, False, False, True]])
    features = [torch.tensor([[101, 102], [103, 104]], dtype=torch.float32),
                torch.tensor([[201, 202]], dtype=torch.float32)]
    outputs = _assert_case(solve, text, labels, attention, image_mask, features)
    assert outputs["inputs_embeds"].tolist() == [
        [[1, 10], [101, 102], [103, 104], [2, 20], [3, 30]],
        [[4, 40], [201, 202], [0, 0], [0, 0], [0, 0]],
    ]
    assert outputs["labels"].tolist() == [[11, -100, -100, -100, 13], [14, -100, -100, -100, -100]]
    assert outputs["attention_mask"].tolist() == [[True] * 5, [True, True, False, False, False]]
    assert outputs["position_ids"].tolist() == [[0, 1, 2, 3, 4], [0, 1, 0, 0, 0]]
    assert outputs["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]]


def check_marker_at_head_middle_tail(solve):
    for position in (0, 2, 4):
        _assert_case(solve, *_case_data([[True] * 5], [position], [3]))


def check_single_feature_no_label_shift(solve):
    for position in range(4):
        args = _case_data([[True] * 4], [position], [1])
        outputs = _assert_case(solve, *args)
        expected = [100, 101, 102, 103]
        expected[position] = -100
        assert outputs["labels"].tolist() == [expected]
        assert outputs["inputs_embeds"].shape[1] == 4


def check_variable_lengths_and_padding_holes(solve):
    valid = [
        [False, False, True, True, True, True, True],
        [True, True, True, False, False, False, False],
        [False, True, False, True, False, True, False],
        [True, False, False, True, True, False, True],
    ]
    outputs = _assert_case(solve, *_case_data(valid, [2, 2, 3, 6], [1, 5, 2, 7]))
    assert outputs["attention_mask"].sum(dim=1).tolist() == [5, 7, 4, 10]
    assert outputs["inputs_embeds"].shape == (4, 10, 3)


def check_text_ignore_labels_are_not_images(solve):
    args = _case_data([[True] * 5], [2], [3])
    args[1].fill_(-100)
    outputs = _assert_case(solve, *args)
    assert outputs["mm_token_type_ids"].tolist() == [[0, 0, 1, 1, 1, 0, 0]]
    assert outputs["attention_mask"].all()


def check_padding_sentinels_do_not_leak(solve):
    args = _case_data([[False, True, False, True, False], [True, True, True, True, True]], [3, 0], [1, 4])
    text, labels, attention, image_mask, _ = args
    text[~attention] = 1e8
    labels[~attention] = 999999
    text[image_mask] = -1e8
    labels[image_mask] = 888888
    first = _assert_case(solve, *args)
    text[~attention | image_mask] = -765432
    labels[~attention | image_mask] = -987654
    second = _assert_case(solve, *args)
    for key in _KEYS:
        assert torch.equal(first[key], second[key]), "Discarded positions leaked into " + key


def _strided(tensor):
    storage = torch.zeros((*tensor.shape, 3), dtype=tensor.dtype)
    storage[..., 1] = tensor
    return storage[..., 1], storage


def check_noncontiguous_and_no_mutation(solve):
    args = _case_data([[True, False, True, True, False], [False, True, True, False, True]], [2, 4], [2, 4], dtype=torch.float64)
    pairs = [_strided(t) for t in list(args[:4]) + args[4]]
    views = [pair[0] for pair in pairs]
    storage_before = [pair[1].clone() for pair in pairs]
    assert all(not t.is_contiguous() and t.storage_offset() > 0 for t in views)
    _assert_case(solve, *views[:4], views[4:])
    for (_, storage), original in zip(pairs, storage_before):
        assert torch.equal(storage, original), "Backing storage was mutated"


def check_dtypes(solve):
    for dtype in (torch.float32, torch.float64):
        args = _case_data([[True, True, True], [False, True, True]], [1, 1], [2, 5], D=4, dtype=dtype)
        args[0].sub_(3.25)
        if dtype == torch.float64:
            args[0].add_(2**-35)
            for feature in args[4]:
                feature.add_(2**-30)
        _assert_case(solve, *args)


def check_minimum_and_maximum_sizes(solve):
    _assert_case(solve, *_case_data([[True]], [0], [1], D=1))
    _assert_case(solve, *_case_data([[True]], [0], [12], D=8))
    args = _case_data([[True] * 16 for _ in range(4)], [0, 7, 8, 15], [12, 1, 8, 12], D=8)
    outputs = _assert_case(solve, *args)
    assert outputs["inputs_embeds"].shape == (4, 27, 8)
    assert outputs["attention_mask"].sum(dim=1).tolist() == [27, 16, 23, 27]


def check_randomized(solve):
    generator = torch.Generator().manual_seed(8602)
    for trial in range(20):
        B = int(torch.randint(1, 5, (), generator=generator))
        L = int(torch.randint(1, 17, (), generator=generator))
        D = int(torch.randint(1, 9, (), generator=generator))
        valid = torch.rand((B, L), generator=generator) > 0.4
        positions = []
        for b in range(B):
            position = int(torch.randint(L, (), generator=generator))
            valid[b, position] = True
            positions.append(position)
        counts = torch.randint(1, 13, (B,), generator=generator).tolist()
        dtype = torch.float32 if trial % 2 else torch.float64
        args = _case_data(valid.tolist(), positions, counts, D=D, dtype=dtype)
        args[1][torch.rand((B, L), generator=generator) < 0.3] = -100
        _assert_case(solve, *args)


def check_missing_markers(solve):
    _expect_value_error(solve, *_case_data([[True, True]], [None], [1]))
    _expect_value_error(solve, *_case_data([[False, False]], [None], [1]))


def check_multiple_markers(solve):
    args = _case_data([[True, True, True]], [0], [2])
    args[3][0, 2] = True
    _expect_value_error(solve, *args)
    args = _case_data([[True, True], [True, True]], [None, 0], [1, 1])
    args[3][1, 1] = True
    _expect_value_error(solve, *args)


def check_markers_in_padding(solve):
    _expect_value_error(solve, *_case_data([[False, True, True]], [0], [1]))
    args = _case_data([[True, False, True]], [0], [1])
    args[3][0, 1] = True
    _expect_value_error(solve, *args)
    _expect_value_error(solve, *_case_data([[False]], [0], [1]))


def check_invalid_later_row_no_mutation(solve):
    args = _case_data([[True, True, True], [True, True, True]], [1, None], [3, 2])
    _expect_value_error(solve, *args)
    args = _case_data([[True, True, True], [True, False, True]], [1, 1], [2, 4])
    _expect_value_error(solve, *args)


check_example.example = True
