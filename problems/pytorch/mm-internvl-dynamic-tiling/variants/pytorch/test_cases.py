import random


def _oracle(width, height, min_num, max_num, image_size, use_thumbnail,
            patch_size=14, scale_factor=0.5):
    """纯 Python 重算（与 solution 同规则，独立实现）。"""
    ratios = []
    seen = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num and (i, j) not in seen:
                    seen.add((i, j))
                    ratios.append((i * j, i, j))
    ratios.sort()

    r = width / height
    best = None
    best_diff = None
    for _, i, j in ratios:
        diff = abs(r - i / j)
        if best is None or diff < best_diff:
            best, best_diff = (i, j), diff
        elif diff == best_diff:
            if width * height > 0.5 * image_size * image_size * i * j:
                best = (i, j)
    gi, gj = best
    blocks = gi * gj
    thumbs = 1 if (use_thumbnail and blocks != 1) else 0
    per = (image_size // patch_size) ** 2 * scale_factor ** 2
    total = per * (blocks + thumbs)
    return {
        "aspect_ratio": r, "target_ratio": [gi, gj],
        "target_width": image_size * gi, "target_height": image_size * gj,
        "blocks": blocks, "thumbnails": thumbs, "total_tiles": blocks + thumbs,
        "per_tile_tokens": per, "total_visual_tokens": total,
        "image_token_count": 2 + total,
    }


def check_anchor_square(solve):
    # 方图：ratio=1，最优 (1,1)，无缩略
    led = solve(448, 448, 1, 6, 448, False)
    assert led["aspect_ratio"] == 1.0
    assert led["target_ratio"] == [1, 1]
    assert led["blocks"] == 1
    assert led["thumbnails"] == 0
    assert led["total_tiles"] == 1
    assert led["per_tile_tokens"] == 1024 * 0.25
    assert led["total_visual_tokens"] == 256
    assert led["image_token_count"] == 258


def check_anchor_wide_thumbnail(solve):
    # 2:1 宽图：r=2 精确命中 (2,1)（diff=0）；(4,2) 的 diff 也为 0，但面积规则
    # 要求 area > 0.5*448²*8 = 802816，而 area=401408 不满足 → 不覆盖，保持 (2,1)。
    led = solve(896, 448, 1, 6, 448, True)
    assert led["target_ratio"] == [2, 1], led["target_ratio"]
    assert led["blocks"] == 2
    assert led["thumbnails"] == 1
    assert led["total_tiles"] == 3
    assert led["total_visual_tokens"] == 256 * 3
    assert led["image_token_count"] == 2 + 256 * 3


def check_tie_area_rule(solve):
    # 简单 sanity：r=1.25 的最近网格是 (1,1)（diff 0.25）
    led = solve(5, 4, 1, 2, 448, True)
    assert led["target_ratio"] == [1, 1]
    # 真并列构造（浮点精确）：r=5/4=1.25 与 (1,2)=0.5、(2,1)=2.0 的 diff 都是精确的 0.75；
    # min_num=max_num=2 时候选恰为 (1,2),(2,1)。先到 (1,2)（字典序），
    # 面积规则 area > 0.5*448^2*2 = 200704 时 (2,1) 覆盖，否则保持。
    led = solve(500, 400, 2, 2, 448, False)  # area=200000 < 200704 -> 保持 (1,2)
    assert led["target_ratio"] == [1, 2], led["target_ratio"]
    led = solve(505, 404, 2, 2, 448, False)  # area=204020 > 200704 -> 覆盖 (2,1)
    assert led["target_ratio"] == [2, 1], led["target_ratio"]


def check_min_one_tile(solve):
    led = solve(448, 448, 1, 1, 448, True)
    assert led["target_ratio"] == [1, 1] and led["thumbnails"] == 0  # blocks==1 不加缩略
    assert led["total_tiles"] == 1


def check_thumbnail_only_multi(solve):
    led = solve(448, 896, 1, 6, 448, True)  # r=0.5 -> (1,2) blocks=2 -> +1
    assert led["target_ratio"] == [1, 2]
    assert led["thumbnails"] == 1 and led["total_tiles"] == 3
    led = solve(448, 896, 1, 6, 448, False)
    assert led["total_tiles"] == 2


def check_parametrized_tokens(solve):
    led = solve(448, 448, 1, 6, 224, False, patch_size=14, scale_factor=0.5)
    assert led["per_tile_tokens"] == (224 // 14) ** 2 * 0.25  # 64
    led = solve(448, 448, 1, 6, 448, False, patch_size=7, scale_factor=1.0)
    assert led["per_tile_tokens"] == 64 * 64


def check_random_against_oracle(solve):
    rng = random.Random(648)
    for _ in range(120):
        w = rng.randint(1, 30) * 14
        h = rng.randint(1, 30) * 14
        lo = rng.randint(1, 6)
        hi = rng.randint(lo, 12)
        size = rng.choice([224, 336, 448])
        thumb = rng.random() < 0.5
        got = solve(w, h, lo, hi, size, thumb)
        exp = _oracle(w, h, lo, hi, size, thumb)
        assert got == exp, (w, h, lo, hi, size, thumb, got, exp)


def check_invalid_inputs(solve):
    for bad in ((0, 448, 1, 6, 448, True), (448, 0, 1, 6, 448, True),
                (448, 448, 0, 6, 448, True), (448, 448, 6, 1, 448, True),
                (448, 448, 1, 6, 0, True), (448.0, 448, 1, 6, 448, True),
                (448, 448, True, 6, 448, True), (448, 448, 1, 6, 448, 1)):
        try:
            solve(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
