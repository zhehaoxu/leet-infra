"""Independent pure-Python oracle for Qwen2.5-VL's window reorder accounting."""

import random

import torch


def _oracle(grid_thw, merge, patch, window):
    vws = window // merge // patch
    mu = merge**2
    wi, cu, offset = [], [0], 0
    for t, h, w in grid_thw:
        lh, lw = h // merge, w // merge
        pad_h = vws - lh % vws  # equals vws when divisible: an empty window row
        pad_w = vws - lw % vws
        nwh = (lh + pad_h) // vws
        nww = (lw + pad_w) // vws
        for ti in range(t):
            for wh in range(nwh):
                for ww in range(nww):
                    count = 0
                    for r in range(vws):
                        for c in range(vws):
                            lr, lc = wh * vws + r, ww * vws + c
                            if lr < lh and lc < lw:
                                wi.append(offset + ti * lh * lw + lr * lw + lc)
                                count += 1
                    cu.append(cu[-1] + count * mu)
        offset += t * lh * lw
    deduped = [cu[0]]
    for x in cu[1:]:
        if x != deduped[-1]:
            deduped.append(x)
    reverse = [0] * len(wi)
    for pos, tok in enumerate(wi):
        reverse[tok] = pos
    return wi, deduped, reverse


def _check(solve, grid_thw, merge, patch, window):
    wi, cu, rev = solve(torch.tensor(grid_thw, dtype=torch.int64), merge, patch, window)
    exp_wi, exp_cu, exp_rev = _oracle(grid_thw, merge, patch, window)
    assert wi.dtype == torch.int64 and rev.dtype == torch.int64 and cu.dtype == torch.int32
    assert wi.tolist() == exp_wi, f"window_index mismatch for {grid_thw}"
    assert cu.tolist() == exp_cu, f"cu mismatch for {grid_thw}"
    assert rev.tolist() == exp_rev, f"reverse mismatch for {grid_thw}"
    total = sum(t * (h // merge) * (w // merge) for t, h, w in grid_thw)
    assert wi.shape == (total,) and rev.shape == (total,), "LLM-token level lengths"
    assert cu[0].item() == 0
    assert all(b > a for a, b in zip(cu.tolist(), cu.tolist()[1:])), "cu strictly increasing"
    assert all(0 <= x < total for x in wi.tolist()), "window_index is a permutation of 0..T-1"
    gathered = wi[rev.tolist()]
    assert gathered.tolist() == list(range(total)), "reverse must be the inverse permutation"
    return wi, cu, rev


def check_single_window_anchor(solve):
    # 4x4 patches -> 2x2 llm grid, one window after padding to 4x4
    wi, cu, rev = _check(solve, [[1, 4, 4]], 2, 14, 112)
    assert wi.tolist() == [0, 1, 2, 3]
    assert cu.tolist() == [0, 16], "4 llm tokens x merge_unit 4"
    assert rev.tolist() == [0, 1, 2, 3]


def check_divisible_creates_empty_windows(solve):
    # 16x16 patches -> 8x8 llm grid; pad = vws (NOT 0) creates empty rows/cols
    wi, cu, _ = _check(solve, [[1, 16, 16]], 2, 14, 112)
    assert wi[:8].tolist() == [0, 1, 2, 3, 8, 9, 10, 11], "window keeps (row, col) order"
    assert len(wi) == 64, "empty windows contribute no tokens"
    assert cu.tolist() == [0, 64, 128, 192, 256], "5 boundaries survive unique_consecutive"
    assert len(cu) == 5, "the 5 all-empty windows of the 3x3 grid are deduplicated"


def check_two_images_offset(solve):
    wi, cu, _ = _check(solve, [[1, 16, 16], [1, 8, 8]], 2, 14, 112)
    assert len(wi) == 80
    assert all(64 <= x < 80 for x in wi.tolist()[64:]), "second image offset by 64 llm tokens"
    assert cu.tolist() == [0, 64, 128, 192, 256, 320]


def check_video_frames_nondivisible(solve):
    # t=2, llm 5x3: window rows r0-3 (12 tokens) and r4 (3 tokens) per frame
    wi, cu, _ = _check(solve, [[2, 10, 6]], 2, 14, 112)
    assert wi.tolist() == list(range(30)), "frame-major windows happen to stay ascending"
    assert cu.tolist() == [0, 48, 60, 108, 120], "12*4 and 3*4 accumulate per frame"


def check_merge_unit_scales_cu(solve):
    # one llm token maps to merge^2 patch tokens in the cu account
    _, cu, _ = _check(solve, [[1, 2, 2]], 2, 14, 112)
    assert cu.tolist() == [0, 4], "1 llm token x merge_unit 4"
    # merge=1: a llm token IS a patch token; vws=4 and a 4x4 grid is divisible,
    # so the padding formula still opens an empty window row/column pair
    wi, cu, _ = _check(solve, [[1, 4, 4]], 1, 1, 4)
    assert wi.tolist() == list(range(16))
    assert cu.tolist() == [0, 16], "16 patch tokens; empty windows deduplicated"


def check_random_against_oracle(solve):
    generator = random.Random(645)
    for _ in range(25):
        merge = generator.choice([1, 2, 4])
        patch = generator.choice([1, 2, 7, 14])
        window = generator.choice([28, 56, 112, 224])
        if window // merge // patch < 1:
            continue
        grid = [
            [
                generator.randint(1, 3),
                merge * generator.randint(1, 8),
                merge * generator.randint(1, 8),
            ]
            for _ in range(generator.randint(1, 4))
        ]
        _check(solve, grid, merge, patch, window)


def check_noncontiguous_and_no_mutation(solve):
    base = torch.tensor([[1, 8, 16], [1, 4, 4]], dtype=torch.int64)
    grid = base.t().contiguous().t()  # noncontiguous view, same values
    snapshot = grid.clone()
    _check(solve, grid.tolist(), 2, 14, 112)
    assert torch.equal(grid, snapshot), "input must not be mutated"


def check_invalid_inputs(solve):
    good = torch.tensor([[1, 8, 8]], dtype=torch.int64)
    for bad_grid in (
        torch.tensor([[1, 8, 8]], dtype=torch.int32),  # wrong dtype
        torch.tensor([1, 8, 8], dtype=torch.int64),  # 1-D
        torch.tensor([[1, 8]], dtype=torch.int64),  # [G,2]
        torch.empty(0, 3, dtype=torch.int64),  # empty batch
        torch.tensor([[0, 8, 8]], dtype=torch.int64),  # t=0
        torch.tensor([[1, 7, 8]], dtype=torch.int64),  # h not divisible by merge
    ):
        try:
            solve(bad_grid, 2, 14, 112)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for grid {bad_grid.tolist()}")
    for bad_scalar in ((0, 14, 112), (2, 0, 112), (2, 14, 0), (2.0, 14, 112), (True, 14, 112)):
        try:
            solve(good, *bad_scalar)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for scalars {bad_scalar}")
    try:
        solve(good, 2, 14, 14)  # vit_merger_window_size = 14//2//14 = 0
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when vit_merger_window_size == 0")
