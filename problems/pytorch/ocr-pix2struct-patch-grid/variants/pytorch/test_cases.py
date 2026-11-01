def _py(image_height, image_width, max_patches, patch_height=16, patch_width=16):
    """oracle：整数 isqrt 精确路径（floor(sqrt(A/B)) = isqrt(A//B)），与浮点实现独立。"""
    # floor(scale*H/ph) = floor(sqrt(M*pw*H / (W*ph)))，平方化消去 sqrt 后用 isqrt
    raw_rows = _isqrt(max_patches * patch_width * image_height, image_width * patch_height)
    raw_cols = _isqrt(max_patches * patch_height * image_width, image_height * patch_width)
    R = max(min(raw_rows, max_patches), 1)
    C = max(min(raw_cols, max_patches), 1)
    ids = []
    r = 0
    while r < R:
        c = 0
        while c < C:
            ids.append([r + 1, c + 1])
            c += 1
        r += 1
    return {
        "num_rows": R,
        "num_cols": C,
        "resized_height": max(R * patch_height, 1),
        "resized_width": max(C * patch_width, 1),
        "total_patches": R * C,
        "padded": max_patches - R * C,
        "patch_ids": ids,
    }


def _isqrt(num, den):
    """floor(sqrt(num / den))，纯整数（num, den 正整数）。"""
    return _int_sqrt(num // den)


def _int_sqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def _compare(solve, H, W, M, ph=16, pw=16):
    led = solve(H, W, M, ph, pw)
    ref = _py(H, W, M, ph, pw)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_square(solve):
    # 224×224、M=64：scale=4/7 → R=C=8，总 patch 恰 64，无 padding
    led = _compare(solve, 224, 224, 64)
    assert led["num_rows"] == 8 and led["num_cols"] == 8
    assert led["resized_height"] == 128 and led["resized_width"] == 128
    assert led["total_patches"] == 64 and led["padded"] == 0


def check_anchor_wide(solve):
    # 640×320（2:1）、M=64：R=11（floor 11.31）、C=5（floor 5.66）→ 55 patches，pad 9
    led = _compare(solve, 640, 320, 64)
    assert led["num_rows"] == 11 and led["num_cols"] == 5
    assert led["resized_height"] == 176 and led["resized_width"] == 80
    assert led["total_patches"] == 55 and led["padded"] == 9
    # 行优先 + 偏移 1：首个 [1,1]，行末 [1,2]，末个 [11,5]
    ids = led["patch_ids"]
    assert ids[0] == [1, 1] and ids[1] == [1, 2]
    assert ids[5] == [2, 1]
    assert ids[-1] == [11, 5]
    assert len(ids) == 55


def check_extreme_aspect_clamps(solve):
    # 1000000×1：R 被 min(...,M) clamp 到 64，C 的 floor 为 0 被 max(...,1) 救回 1
    led = _compare(solve, 1000000, 1, 64)
    assert led["num_rows"] == 64 and led["num_cols"] == 1
    assert led["resized_height"] == 1024 and led["resized_width"] == 16
    assert led["total_patches"] == 64 and led["padded"] == 0
    # 1×1 小图：scale=128 → R=C=8（未被 clamp，floor(8)=8）
    led2 = _compare(solve, 1, 1, 64)
    assert led2["num_rows"] == 8 and led2["num_cols"] == 8
    assert led2["total_patches"] == 64 and led2["padded"] == 0


def check_small_budget(solve):
    # M=4、64×64：scale=0.5 → R=C=2
    led = _compare(solve, 64, 64, 4)
    assert led["num_rows"] == 2 and led["num_cols"] == 2
    assert led["total_patches"] == 4 and led["padded"] == 0
    # M=32、7×13（比 M 更碎的小图）：R=4、C=7 → 28 patches，pad 4
    led2 = _compare(solve, 7, 13, 32)
    assert led2["num_rows"] == 4 and led2["num_cols"] == 7
    assert led2["total_patches"] == 28 and led2["padded"] == 4
    assert led2["resized_height"] == 64 and led2["resized_width"] == 112


def check_non_square_patch(solve):
    # 非方形 patch：8×32 patch 下 512×256、M=48
    led = _compare(solve, 512, 256, 48, 8, 32)
    import math
    scale = math.sqrt(48 * (8 / 512) * (32 / 256))
    R = max(min(math.floor(scale * 512 / 8), 48), 1)
    C = max(min(math.floor(scale * 256 / 32), 48), 1)
    assert led["num_rows"] == R and led["num_cols"] == C
    assert led["resized_height"] == R * 8 and led["resized_width"] == C * 32
    assert led["total_patches"] == R * C and led["padded"] == 48 - R * C


def check_default_patch_size(solve):
    # 默认 16×16 与显式传 16×16 一致
    a = solve(300, 200, 100)
    b = solve(300, 200, 100, 16, 16)
    assert a == b


def check_rc_le_max_patches(solve):
    # 不变量：R*C ≤ M（floor 乘积不等式），padded 非负
    import random
    rng = random.Random(688)
    for _ in range(500):
        H = rng.randint(1, 4000)
        W = rng.randint(1, 4000)
        M = rng.randint(1, 512)
        ph = rng.choice([1, 4, 8, 16, 32])
        pw = rng.choice([1, 4, 8, 16, 32])
        led = solve(H, W, M, ph, pw)
        assert led["total_patches"] <= M, (H, W, M, ph, pw, led)
        assert led["padded"] == M - led["total_patches"] >= 0
        assert led["num_rows"] >= 1 and led["num_cols"] >= 1
        assert led["resized_height"] == led["num_rows"] * ph
        assert led["resized_width"] == led["num_cols"] * pw


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    ve(0, 100, 64)            # 高为 0
    ve(100, 0, 64)
    ve(100, 100, 0)           # max_patches 为 0
    ve(100, 100, -4)          # 负数
    ve(True, 100, 64)         # bool 冒充 int
    ve(100, 100, 64, 0, 16)   # patch 高为 0
    ve(100, 100, 64, 16, -1)
    ve(10.5, 100, 64)         # 非 int
    ve(100, 100, 64, 16.0)


def check_oracle_random(solve):
    import random
    rng = random.Random(20260688)
    for _ in range(200):
        H = rng.randint(1, 5000)
        W = rng.randint(1, 5000)
        M = rng.randint(1, 128)
        ph = rng.choice([8, 16, 32])
        pw = rng.choice([8, 16, 32])
        led = solve(H, W, M, ph, pw)
        ref = _py(H, W, M, ph, pw)
        assert led == ref, f"random mismatch at {(H, W, M, ph, pw)}"
