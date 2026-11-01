"""order 693: FunASR `mask_along_axis` 段生成与合并账。

oracle 用逐段区间集合（区间合并算法）独立实现，与 solve 对拍；
另加手算锚点钉死布尔并/截断/均值填充。
"""
import random

SPEC = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]


def _oracle(spec, masks, axis="time", replace_with_zero=True):
    T = len(spec)
    F = len(spec[0]) if T else 0
    D = T if axis == "time" else F

    # 独立实现：截断成区间列表 → 经典区间合并 → 展开布尔
    clipped = []
    for pos, width in masks:
        if width <= 0:
            continue
        lo = max(0, pos)
        hi = min(D, pos + width)
        if lo < hi:
            clipped.append((lo, hi))
    clipped.sort()
    merged = []
    for lo, hi in clipped:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    mask = [False] * D
    for lo, hi in merged:
        for d in range(lo, hi):
            mask[d] = True

    if replace_with_zero:
        value = 0.0
    else:
        value = sum(v for row in spec for v in row) / (T * F)

    masked = [row[:] for row in spec]
    for lo, hi in merged:
        for d in range(lo, hi):
            if axis == "time":
                masked[d] = [value] * F
            else:
                for t in range(T):
                    masked[t][d] = value

    return {"masked": masked, "segments": [[lo, hi] for lo, hi in merged]}


def _compare(solve, spec, masks, axis="time", replace_with_zero=True):
    led = solve([row[:] for row in spec], [list(m) for m in masks], axis,
                replace_with_zero)
    exp = _oracle([row[:] for row in spec], [list(m) for m in masks], axis,
                  replace_with_zero)
    assert led == exp, f"solve={led}\noracle={exp}\n{masks} {axis}"
    return led


def check_single_band_time(solve):
    led = _compare(solve, SPEC, [[1, 2]], "time")
    assert led["masked"][1] == [0.0] * 3 and led["masked"][2] == [0.0] * 3
    assert led["masked"][0] == SPEC[0] and led["masked"][3] == SPEC[3]
    assert led["segments"] == [[1, 3]]


def check_adjacent_bands_merge(solve):
    # 相邻段 [0,2) 与 [2,4) 在布尔并上连成同一区段（any 的效果）
    led = _compare(solve, SPEC, [[0, 2], [2, 2]], "time")
    assert led["segments"] == [[0, 4]]
    assert all(row == [0.0] * 3 for row in led["masked"])


def check_overlapping_bands(solve):
    led = _compare(solve, SPEC, [[0, 3], [1, 1]], "time")
    assert led["segments"] == [[0, 3]]
    # 完全重复段并集不变
    led2 = _compare(solve, SPEC, [[1, 2], [1, 2]], "time")
    assert led2["segments"] == [[1, 3]]


def check_out_of_range_clipped(solve):
    # 越界截断：[2,5) 在 D=4 上只到 [2,4)
    led = _compare(solve, SPEC, [[2, 3]], "time")
    assert led["segments"] == [[2, 4]] and led["masked"][3] == [0.0] * 3
    # 完全越界段为空
    led2 = _compare(solve, SPEC, [[10, 3]], "time")
    assert led2["segments"] == [] and led2["masked"] == SPEC
    # 负起点截到 0
    led3 = _compare(solve, SPEC, [[-2, 3]], "time")
    assert led3["segments"] == [[0, 1]]


def check_zero_width_empty(solve):
    led = _compare(solve, SPEC, [[1, 0]], "time")
    assert led["segments"] == [] and led["masked"] == SPEC
    # width=0 与真实段共存时真实段保留
    led2 = _compare(solve, SPEC, [[1, 0], [2, 1]], "time")
    assert led2["segments"] == [[2, 3]]


def check_freq_axis(solve):
    led = _compare(solve, SPEC, [[1, 1]], "freq")
    assert [row[1] for row in led["masked"]] == [0.0] * 4
    assert [row[0] for row in led["masked"]] == [1.0, 4.0, 7.0, 10.0]
    assert led["segments"] == [[1, 2]]
    # freq 轴 D=F=3，pos=2 width=3 越界截到 [2,3)
    led2 = _compare(solve, SPEC, [[2, 3]], "freq")
    assert led2["segments"] == [[2, 3]]


def check_mean_fill(solve):
    # replace_with_zero=False：填全谱均值 6.5（含将被 mask 的元素）
    led = _compare(solve, SPEC, [[0, 1]], "time", False)
    assert led["masked"][0] == [6.5] * 3
    # freq 轴 mean 相同（全谱标量）
    led2 = _compare(solve, SPEC, [[0, 3]], "freq", False)
    assert all(row == [6.5] * 3 for row in led2["masked"])


def check_empty_masks_and_spec(solve):
    led = _compare(solve, SPEC, [], "time")
    assert led["segments"] == [] and led["masked"] == SPEC
    led2 = _compare(solve, [], [], "time")
    assert led2 == {"masked": [], "segments": []}


def check_segments_sorted_disjoint(solve):
    rng = random.Random(693)
    for _ in range(300):
        T = rng.randint(1, 12)
        F = rng.randint(1, 8)
        spec = [[float(rng.randint(0, 50)) for _ in range(F)] for _ in range(T)]
        axis = rng.choice(["time", "freq"])
        D = T if axis == "time" else F
        n = rng.randint(0, 5)
        masks = [[rng.randint(-2, D + 2), rng.randint(0, max(1, D))]
                 for _ in range(n)]
        rwz = rng.random() < 0.5
        led = solve([row[:] for row in spec], [list(m) for m in masks], axis, rwz)
        exp = _oracle([row[:] for row in spec], [list(m) for m in masks], axis, rwz)
        assert led == exp, f"solve={led}\noracle={exp}\n{masks} {axis}"
        # 区段升序、互不重叠、右开
        segs = led["segments"]
        for k in range(len(segs) - 1):
            assert segs[k][1] < segs[k + 1][0]
        for lo, hi in segs:
            assert 0 <= lo < hi <= D
