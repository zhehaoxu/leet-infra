import torch

_RATIOS = [
    (1, 1),
    (4, 5),
    (5, 4),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (9, 16),
    (16, 9),
    (1, 2),
    (2, 1),
]


def _oracle_buckets(resolution, divisibility, base_resolutions):
    """独立 oracle：先从 w = sqrt(target * a) 求宽、再 h = w / a 求高，与主解的 h 先行相反。"""
    target = resolution * resolution
    if base_resolutions is None:
        ratios = _RATIOS
    else:
        ratios = [(w, h) for (h, w) in base_resolutions]
    out = []
    seen = set()
    for rw, rh in ratios:
        a = rw / rh
        w = (target * a) ** 0.5
        h = w / a
        h = max(divisibility, round(h / divisibility) * divisibility)
        w = max(divisibility, round(w / divisibility) * divisibility)
        if (h, w) not in seen:
            seen.add((h, w))
            out.append([h, w])
    return out


def _oracle_assign(h, w, buckets):
    best = None
    best_metric = None
    for i, (bh, bw) in enumerate(buckets):
        metric = abs(h * bw - w * bh)
        if best_metric is None or metric <= best_metric:
            best_metric = metric
            best = i
    return best


def check_example(solve):
    images = torch.tensor(
        [[512, 512], [768, 512], [512, 768], [1088, 976], [720, 1280]], dtype=torch.int64
    )
    buckets, idx = solve(512, images)
    assert buckets.dtype == torch.int64 and idx.dtype == torch.int64
    assert buckets.shape == (11, 2), buckets.shape
    assert idx.tolist() == [0, 3, 4, 1, 8], idx.tolist()
    assert buckets.tolist()[:3] == [[512, 512], [576, 464], [464, 576]]


check_example.example = True


def check_bucket_ladder_512(solve):
    images = torch.zeros(0, 2, dtype=torch.int64)
    buckets, idx = solve(512, images)
    want = [
        [512, 512],
        [576, 464],
        [464, 576],
        [624, 416],
        [416, 624],
        [592, 448],
        [448, 592],
        [688, 384],
        [384, 688],
        [720, 368],
        [368, 720],
    ]
    assert buckets.tolist() == want, buckets.tolist()
    assert idx.numel() == 0


def check_pixel_budget_and_divisibility(solve):
    images = torch.zeros(0, 2, dtype=torch.int64)
    for res in (256, 512, 1024):
        for div in (8, 16, 32, 64):
            buckets, _ = solve(res, images, divisibility=div)
            rows = buckets.tolist()
            assert len(rows) == len({tuple(r) for r in rows}), "桶必须去重"
            for h, w in rows:
                assert h % div == 0 and w % div == 0, (res, div, h, w)
                assert h >= div and w >= div
    # 对齐粒度固定为 16 时，每个桶的像素量都接近 resolution^2
    for res in (256, 512, 1024):
        buckets, _ = solve(res, images, divisibility=16)
        for h, w in buckets.tolist():
            assert abs(h * w - res * res) <= 0.05 * res * res, (res, h, w)


def check_base_resolutions(solve):
    empty = torch.zeros(0, 2, dtype=torch.int64)
    buckets, _ = solve(512, empty, base_resolutions=[(1024, 1024), (768, 1344)])
    assert buckets.tolist() == [[512, 512], [384, 672]], buckets.tolist()

    # 竖图基准 (h=1080, w=1920) 的内部 aspect 是 w/h = 16/9，产出横桶
    buckets, _ = solve(512, empty, base_resolutions=[(1080, 1920)])
    assert buckets.tolist() == [[384, 688]], buckets.tolist()

    images = torch.tensor([[1024, 1024], [768, 1344]], dtype=torch.int64)
    _, idx = solve(512, images, base_resolutions=[(1024, 1024), (768, 1344)])
    assert idx.tolist() == [0, 1], idx.tolist()


def check_tie_prefers_later_bucket(solve):
    images = torch.tensor([[1088, 976]], dtype=torch.int64)
    buckets, idx = solve(512, images)
    rows = buckets.tolist()
    m0 = abs(1088 * rows[0][1] - 976 * rows[0][0])
    m1 = abs(1088 * rows[1][1] - 976 * rows[1][0])
    assert m0 == m1 == 57344, (m0, m1)
    assert idx.tolist() == [1], "并列时源实现用 <=，必须选更靠后的桶"

    # 同一张图只要略微偏向 1:1，就应落回第 0 个桶
    _, idx = solve(512, torch.tensor([[1070, 976]], dtype=torch.int64))
    assert idx.tolist() == [0], idx.tolist()


def check_orientation_and_square(solve):
    images = torch.tensor(
        [[768, 512], [512, 768], [1280, 720], [720, 1280]], dtype=torch.int64
    )
    _, idx = solve(512, images)
    assert idx.tolist() == [3, 4, 7, 8], idx.tolist()

    squares = torch.tensor([[500, 500], [512, 512], [600, 600]], dtype=torch.int64)
    _, idx = solve(512, squares)
    assert idx.tolist() == [0, 0, 0], idx.tolist()


def check_dedupe_low_resolution(solve):
    empty = torch.zeros(0, 2, dtype=torch.int64)
    buckets, _ = solve(16, empty, divisibility=16)
    assert buckets.tolist() == [[16, 16]], buckets.tolist()

    buckets, _ = solve(32, empty, divisibility=16)
    rows = buckets.tolist()
    assert len(rows) == len({tuple(r) for r in rows}) and len(rows) == 5, rows


def check_bad_input(solve):
    ok = torch.tensor([[512, 512]], dtype=torch.int64)
    bads = [
        lambda: solve(0, ok),
        lambda: solve(-512, ok),
        lambda: solve(512.0, ok),
        lambda: solve(True, ok),
        lambda: solve(512, ok, divisibility=0),
        lambda: solve(512, ok, divisibility=-16),
        lambda: solve(512, [[512, 512]]),
        lambda: solve(512, torch.tensor([512, 512], dtype=torch.int64)),
        lambda: solve(512, torch.tensor([[512, 512, 1]], dtype=torch.int64)),
        lambda: solve(512, torch.tensor([[512.0, 512.0]])),
        lambda: solve(512, torch.tensor([[512, 0]], dtype=torch.int64)),
        lambda: solve(512, torch.tensor([[-4, 512]], dtype=torch.int64)),
        lambda: solve(512, ok, base_resolutions=[]),
        lambda: solve(512, ok, base_resolutions=(512, 512)),
        lambda: solve(512, ok, base_resolutions=[(512, 512, 768)]),
        lambda: solve(512, ok, base_resolutions=[(512, 0)]),
        lambda: solve(512, ok, base_resolutions=[(512.0, 512.0)]),
    ]
    for fn in bads:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("非法输入应抛 ValueError")


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(20261029)
    resolutions = [64, 128, 256, 320, 384, 448, 512, 640, 768, 1024]
    divisibilities = [8, 16, 32, 64]
    for _ in range(25):
        res = resolutions[int(torch.randint(0, len(resolutions), (1,), generator=g).item())]
        div = divisibilities[int(torch.randint(0, len(divisibilities), (1,), generator=g).item())]
        base = None
        if int(torch.randint(0, 2, (1,), generator=g).item()) == 1:
            py = int(torch.randint(256, 2048, (1,), generator=g).item())
            px = int(torch.randint(256, 2048, (1,), generator=g).item())
            base = [(py, px), (px, py)]
        n = int(torch.randint(1, 8, (1,), generator=g).item())
        images = torch.randint(64, 2048, (n, 2), generator=g, dtype=torch.int64)

        buckets, idx = solve(res, images, divisibility=div, base_resolutions=base)
        want_buckets = _oracle_buckets(res, div, base)
        assert buckets.tolist() == want_buckets, (res, div, base, buckets.tolist(), want_buckets)
        want_idx = [_oracle_assign(int(h), int(w), want_buckets) for h, w in images.tolist()]
        assert idx.tolist() == want_idx, (images.tolist(), idx.tolist(), want_idx)
