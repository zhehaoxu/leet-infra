import torch


def _oracle(video, tau, p, m):
    T, C, H, W = video.shape
    gt, gh, gw = (T + tau - 1) // tau, H // p, W // p
    rows, coords = [], []
    for t in range(gt):
        for block_h in range(gh // m):
            for block_w in range(gw // m):
                for inner_h in range(m):
                    for inner_w in range(m):
                        h, w = block_h * m + inner_h, block_w * m + inner_w
                        coords.append([t, h, w])
                        row = []
                        for c in range(C):
                            for dt in range(tau):
                                for ph in range(p):
                                    for pw in range(p):
                                        frame = min(t * tau + dt, T - 1)
                                        row.append(video[frame, c, h * p + ph, w * p + pw].item())
                        rows.append(row)
    return torch.tensor(rows, dtype=video.dtype), (gt, gh, gw), torch.tensor(coords, dtype=torch.int64)


def _assert_reconstruction(video, patches, grid, coords, tau, p):
    T, C, H, W = video.shape
    restored = torch.empty((grid[0] * tau, C, H, W), dtype=video.dtype)
    seen = set()
    for n in range(patches.shape[0] - 1, -1, -1):
        t, h, w = coords[n].tolist()
        assert (t, h, w) not in seen, "Duplicate patch coordinate"
        seen.add((t, h, w))
        for c in range(C):
            for dt in range(tau):
                for ph in range(p):
                    for pw in range(p):
                        k = ((c * tau + dt) * p + ph) * p + pw
                        restored[t * tau + dt, c, h * p + ph, w * p + pw] = patches[n, k]
    assert len(seen) == grid[0] * grid[1] * grid[2]
    for frame in range(restored.shape[0]):
        assert torch.equal(restored[frame], video[min(frame, T - 1)]), "Inverse reconstruction failed"


def _assert_case(solve, video, tau, p, m):
    original = video.clone()
    layout = (video.shape, video.stride(), video.storage_offset())
    expected, expected_grid, expected_coords = _oracle(original, tau, p, m)
    result = solve(video, tau, p, m)
    assert torch.equal(video, original), "Input video was mutated"
    assert layout == (video.shape, video.stride(), video.storage_offset())
    assert isinstance(result, tuple) and len(result) == 3
    patches, grid, coords = result
    assert isinstance(grid, tuple) and all(type(x) is int for x in grid)
    assert grid == expected_grid
    assert isinstance(patches, torch.Tensor) and isinstance(coords, torch.Tensor)
    assert patches.device.type == coords.device.type == "cpu"
    assert patches.dtype == video.dtype and patches.shape == expected.shape
    assert coords.dtype == torch.int64 and coords.shape == expected_coords.shape
    assert torch.equal(patches, expected), "Patch values or token/feature order differ"
    assert torch.equal(coords, expected_coords), "Coordinates differ from merge-friendly order"
    _assert_reconstruction(original, patches, grid, coords, tau, p)
    return result


def _video(T, C, H, W, dtype=torch.int64):
    return torch.arange(T * C * H * W, dtype=dtype).reshape(T, C, H, W)


def check_example(solve):
    video = _video(3, 1, 2, 4)
    patches, grid, coords = _assert_case(solve, video, 2, 1, 2)
    assert grid == (2, 2, 4)
    assert patches.tolist() == [
        [0, 8], [1, 9], [4, 12], [5, 13], [2, 10], [3, 11], [6, 14], [7, 15],
        [16, 16], [17, 17], [20, 20], [21, 21], [18, 18], [19, 19], [22, 22], [23, 23],
    ]
    assert coords.tolist() == [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [0, 0, 2], [0, 0, 3], [0, 1, 2], [0, 1, 3],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        [1, 0, 2], [1, 0, 3], [1, 1, 2], [1, 1, 3],
    ]


def check_single_frame(solve):
    for tau in (1, 2, 4):
        _assert_case(solve, torch.tensor([[[[7]]]], dtype=torch.int64), tau, 1, 1)
        _assert_case(solve, _video(1, 2, 4, 8), tau, 2, 2)


def check_odd_tail_duplication(solve):
    for T, tau in ((3, 2), (5, 2), (7, 4), (9, 4), (5, 3)):
        _assert_case(solve, _video(T, 2, 4, 4), tau, 2, 2)


def check_exact_temporal_groups(solve):
    for T, tau in ((4, 2), (8, 4), (9, 3)):
        _assert_case(solve, _video(T, 2, 4, 8), tau, 2, 2)


def check_merge_order_not_raster(solve):
    video = _video(1, 1, 4, 4)
    patches, _, coords = _assert_case(solve, video, 1, 1, 2)
    assert patches[:, 0].tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    assert coords[2].tolist() == [0, 1, 0]
    for m, H, W in ((1, 4, 6), (3, 6, 9), (4, 8, 12)):
        _assert_case(solve, _video(2, 1, H, W), 2, 1, m)


def check_rectangles(solve):
    for p, m, H, W in ((2, 2, 4, 12), (2, 2, 12, 4), (3, 2, 6, 18), (4, 1, 8, 12)):
        _assert_case(solve, _video(3, 2, H, W), 2, p, m)


def check_channels_and_inner_order(solve):
    patches, _, _ = _assert_case(solve, _video(2, 2, 2, 2), 2, 2, 1)
    assert patches[0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
    _assert_case(solve, _video(4, 4, 6, 12), 3, 3, 2)


def check_noncontiguous_and_no_mutation(solve):
    base = _video(5, 3, 8, 16, torch.float64)
    original = base.clone()
    video = base[1::2, :, 1::2, ::2].transpose(2, 3)
    assert not video.is_contiguous() and video.storage_offset() > 0
    _assert_case(solve, video, 3, 2, 2)
    assert torch.equal(base, original)


def check_dtypes(solve):
    for dtype in (torch.int64, torch.float32, torch.float64):
        video = _video(3, 3, 4, 8, dtype) - 200
        if dtype == torch.int64:
            video = video + 2**40
        else:
            video = video / 8
        _assert_case(solve, video, 2, 2, 2)


def check_padding_equivalence_and_value_shift(solve):
    video = _video(3, 2, 4, 8, torch.float64) / 8
    patches, grid, coords = _assert_case(solve, video, 4, 2, 2)
    explicit = torch.cat((video, video[-1:]), dim=0)
    padded, padded_grid, padded_coords = _assert_case(solve, explicit, 4, 2, 2)
    shifted, shifted_grid, shifted_coords = _assert_case(solve, video + 6, 4, 2, 2)
    assert torch.equal(patches, padded) and grid == padded_grid == shifted_grid
    assert torch.equal(shifted, patches + 6)
    assert torch.equal(coords, padded_coords) and torch.equal(coords, shifted_coords)


def check_maximum_dimensions(solve):
    _assert_case(solve, _video(9, 4, 32, 32, torch.float32), 4, 4, 4)


def check_randomized(solve):
    generator = torch.Generator().manual_seed(86)
    for _ in range(12):
        T = int(torch.randint(1, 10, (), generator=generator))
        C = int(torch.randint(1, 5, (), generator=generator))
        tau = int(torch.randint(1, 5, (), generator=generator))
        p = int(torch.randint(1, 5, (), generator=generator))
        m = int(torch.randint(1, 5, (), generator=generator))
        H = p * m * int(torch.randint(1, 3, (), generator=generator))
        W = p * m * int(torch.randint(1, 3, (), generator=generator))
        video = torch.randint(-50, 51, (T, C, H, W), generator=generator)
        _assert_case(solve, video, tau, p, m)


def check_invalid_spatial_divisibility(solve):
    for H, W, p, m in ((3, 8, 2, 2), (8, 3, 2, 2), (6, 8, 2, 2), (8, 6, 2, 2), (2, 2, 2, 2)):
        video = _video(3, 1, H, W)
        original = video.clone()
        try:
            solve(video, 2, p, m)
        except ValueError:
            pass
        else:
            raise AssertionError("Nondivisible spatial dimensions must raise ValueError")
        assert torch.equal(video, original)


check_example.example = True
