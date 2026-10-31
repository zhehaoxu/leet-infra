"""Qwen2-Audio 音频占位展开数账（test cases）。"""

import random


def _py(input_ids, mask):
    """独立 oracle：逐行重放。"""
    assert all(t in (0, 2) for t in input_ids)
    pooled = []
    audio_lens = []
    for row in mask:
        s = 0
        for v in row:
            assert v in (0, 1)
            s += v
        pooled.append(s)
        l = s
        l = (l - 1) // 2 + 1
        l = (l - 2) // 2 + 1
        audio_lens.append(l)
    n_ph = 0
    for t in input_ids:
        if t == 2:
            n_ph += 1
    assert n_ph == len(mask), "placeholder count mismatch"
    return {"pooled_lens": pooled, "audio_lens": audio_lens,
            "expanded_len": len(input_ids) - n_ph + sum(audio_lens)}


def _close(actual, expect):
    if isinstance(actual, dict):
        for k in expect:
            assert _close(actual[k], expect[k]), f"mismatch at {k}"
        return True
    if isinstance(actual, list):
        assert isinstance(expect, list) and len(actual) == len(expect), \
            f"len {len(actual)} != {len(expect)}"
        return all(_close(a, e) for a, e in zip(actual, expect))
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, ids, mask):
    led = solve(list(ids), [list(r) for r in mask])
    ref = _py(list(ids), [list(r) for r in mask])
    _close(led, ref)
    return led


def check_anchor_single(solve):
    # 200 有效帧 + 56 pad → pooled 200 → audio_len 50
    mask = [[1] * 200 + [0] * 56]
    ids = [0, 0, 2, 0, 0]
    led = _compare(solve, ids, mask)
    assert led["pooled_lens"] == [200]
    assert led["audio_lens"] == [50]
    assert led["expanded_len"] == 5 - 1 + 50


def check_padding_invariance(solve):
    # 同一音频不同 pad 长度 → 结果相同（池化消 pad）
    ids = [2]
    m1 = [[1] * 100]
    m2 = [[1] * 100 + [0] * 300]
    led1 = _compare(solve, ids, m1)
    led2 = _compare(solve, ids, m2)
    assert led1["pooled_lens"] == led2["pooled_lens"] == [100]
    assert led1["audio_lens"] == led2["audio_lens"] == [25]
    assert led1["expanded_len"] == led2["expanded_len"] == 1 - 1 + 25


def check_conv_boundaries(solve):
    # mel ≤ 2 → audio_len 0（占位替换成空）；mel=3 → 1；mel=7 → 2
    for L, want in [(1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2)]:
        led = _compare(solve, [2], [[1] * L])
        assert led["audio_lens"] == [want], f"L={L}"


def check_zero_len_placeholder(solve):
    # audio_len=0：占位被替换为空，expanded_len 相应缩短
    ids = [0, 2, 0, 0]
    led = _compare(solve, ids, [[1, 1]])
    assert led["audio_lens"] == [0]
    assert led["expanded_len"] == 4 - 1 + 0


def check_multi_audio_order(solve):
    # 多条音频按占位顺序对应 mask 行序（与 input_ids 中占位顺序无关，
    # 对应关系按音频条目序）
    ids = [2, 0, 2]
    mask = [[1] * 200, [1] * 8]  # 8 帧 → (3+1-2)//2+1 = 2
    led = _compare(solve, ids, mask)
    assert led["audio_lens"] == [50, 2]
    assert led["expanded_len"] == 3 - 2 + 52


def check_oracle_random(solve):
    rng = random.Random(675)
    for _ in range(150):
        n_audio = rng.randint(1, 4)
        mask = []
        for _ in range(n_audio):
            L = rng.randint(1, 400)
            pad = rng.randint(0, 100)
            mask.append([1] * L + [0] * pad)
        n_ph = n_audio
        n_text = rng.randint(0, 8)
        slots = n_text + n_ph
        ids = [0] * n_text + [2] * n_ph
        rng.shuffle(ids)
        led = _compare(solve, ids, mask)
        exp_len = len(ids) - n_ph + sum(led["audio_lens"])
        assert led["expanded_len"] == exp_len


def check_invalid_inputs(solve):
    for bad_ids, bad_mask in [
        ([2], []),                    # 占位数 != 音频数
        ([0, 0], [[1]]),              # 占位数 0 != 1
        ([1], [[1]]),                 # 非法 token id
        ([2], [[2]]),                 # mask 值非法
        ([2], [[0, 0]]),              # pooled = 0
        ([0, 1, 2, 0], [[1]]),        # token id 1 非法（占位数恰好匹配）
        ([2], []),                    # 空 mask 列表
    ]:
        try:
            solve(list(bad_ids), [list(r) for r in bad_mask] if bad_mask else [])
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {bad_ids} {bad_mask}")
