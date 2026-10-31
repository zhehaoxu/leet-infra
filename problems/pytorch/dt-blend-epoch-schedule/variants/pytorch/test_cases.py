import math
from bisect import bisect_right
from functools import reduce


def _oracle(sizes, weights, num_samples, temperature):
    """结构独立：probs 用 reduce 左折叠并把条目先转成 float（参考解是就地 for 循环）；
    配额用「反复在未补位的下标里取最大余额」而不是排序（平局天然留下标小者）；
    排布用并查式后继数组跳到第一个空闲槽，理想槽走浮点 floor（参考解是向后线性扫描 +
    整数除法）；文档号用每源发射计数器；全局号用 sum(sizes[:src]) 现算前缀和（参考解
    预先建 offsets 表）。四条路径不共用中间结果。
    """
    expo = 1.0 / float(temperature)
    entries = []
    for index in range(len(sizes)):
        if sizes[index] == 0:
            entries.append(0.0)
        else:
            entries.append(float(weights[index]) ** expo)
    total = reduce(lambda acc, value: acc + value, entries, 0.0)
    probs = [value / total for value in entries]

    ideal = [probs[i] * num_samples for i in range(len(probs))]
    floors = [math.floor(value) for value in ideal]
    fractions = [ideal[i] - floors[i] for i in range(len(ideal))]
    counts = list(floors)
    pool = list(range(len(counts)))
    for _ in range(num_samples - sum(counts)):
        best = pool[0]
        for candidate in pool[1:]:
            if fractions[candidate] > fractions[best]:
                best = candidate
        counts[best] += 1
        pool.remove(best)

    successor = list(range(num_samples + 1))

    def find(slot):
        while successor[slot] != slot:
            successor[slot] = successor[successor[slot]]
            slot = successor[slot]
        return slot

    schedule = [-1] * num_samples
    doc_indices = [-1] * num_samples
    emitted = [0] * len(sizes)
    for source in range(len(sizes)):
        quota = counts[source]
        for _ in range(quota):
            ideal_slot = math.floor(emitted[source] * num_samples / quota)
            slot = find(ideal_slot)
            schedule[slot] = source
            doc_indices[slot] = emitted[source] % sizes[source]
            emitted[source] += 1
            successor[slot] = find(slot + 1)

    flat_indices = [sum(sizes[:source]) + doc for source, doc in zip(schedule, doc_indices)]
    return probs, counts, schedule, doc_indices, flat_indices


def _ascending_placement(counts, num_samples):
    """把「按配额降序放置」这一错误做法算出来，用来证明题面顺序不是无害的细节。"""
    schedule = [-1] * num_samples
    for source in sorted(range(len(counts)), key=lambda i: (-counts[i], i)):
        quota = counts[source]
        for emission in range(quota):
            slot = emission * num_samples // quota
            while schedule[slot] != -1:
                slot += 1
            schedule[slot] = source
    return schedule


def _verify(solve, args):
    sizes, weights, num_samples, temperature = args
    sizes_before = list(sizes)
    weights_before = list(weights)
    result = solve(list(sizes), list(weights), num_samples, temperature)
    assert isinstance(result, tuple) and len(result) == 5, "must return a 5-tuple"
    probs, counts, schedule, doc_indices, flat_indices = result

    exp_probs, exp_counts, exp_schedule, exp_docs, exp_flat = _oracle(
        sizes, weights, num_samples, temperature)

    assert isinstance(probs, list) and len(probs) == len(sizes)
    for value, expected in zip(probs, exp_probs):
        assert isinstance(value, float), "probs entries must be floats"
        assert abs(value - expected) <= 1e-12, f"probs {probs} vs {exp_probs}"
    assert abs(sum(probs) - 1.0) <= 1e-9, f"probs must sum to one: {probs}"

    assert isinstance(counts, list) and len(counts) == len(sizes)
    for value in counts:
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value >= 0
    assert counts == exp_counts, f"counts {counts} vs {exp_counts}"
    assert sum(counts) == num_samples, f"counts must total the budget: {counts}"
    for index, value in enumerate(counts):
        ideal = exp_probs[index] * num_samples
        assert math.floor(ideal) <= value <= math.floor(ideal) + 1
        assert abs(value - ideal) < 1.0, f"quota {counts} drifts from {ideal}"
        if sizes[index] == 0:
            assert value == 0, "an empty source may not receive any sample"

    assert isinstance(schedule, list) and len(schedule) == num_samples
    assert schedule == exp_schedule, f"schedule {schedule} vs {exp_schedule}"
    seen = [0] * len(sizes)
    for source in schedule:
        assert isinstance(source, int) and not isinstance(source, bool)
        assert 0 <= source < len(sizes)
        seen[source] += 1
    assert seen == counts, f"schedule multiplicity {seen} vs quota {counts}"

    assert isinstance(doc_indices, list) and len(doc_indices) == num_samples
    assert doc_indices == exp_docs, f"doc_indices {doc_indices} vs {exp_docs}"
    for slot, doc in enumerate(doc_indices):
        assert 0 <= doc < sizes[schedule[slot]], "document id out of its source range"

    assert isinstance(flat_indices, list) and len(flat_indices) == num_samples
    assert flat_indices == exp_flat, f"flat_indices {flat_indices} vs {exp_flat}"
    cumulative = list(reduce(lambda acc, value: acc + [acc[-1] + value], sizes, [0]))
    total_docs = cumulative[-1]
    for slot, flat in enumerate(flat_indices):
        assert 0 <= flat < total_docs, "global id outside the concatenated corpus"
        source = bisect_right(cumulative[1:], flat)
        base = cumulative[source]
        assert source == schedule[slot], f"flat {flat} decodes to source {source}"
        assert flat - base == doc_indices[slot], f"flat {flat} decodes to doc {flat - base}"

    assert sizes == sizes_before and weights == weights_before, "inputs must not be mutated"


def _lcg(seed):
    state = [seed]

    def next_value(modulus):
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return state[0] % modulus

    return next_value


_FIXTURES = [
    ([2, 3, 5], [1, 1, 2], 7, 1.0),
    ([4, 0, 9], [1, 5, 4], 5, 2.0),
    ([5, 5], [1, 3], 8, 0.5),
    ([1, 1, 1, 1], [1, 1, 1, 1], 6, 1.0),
    ([2], [1], 5, 1.0),
    ([3, 4, 5], [1, 1, 1], 0, 1.0),
    ([10, 10], [0.0, 1.0], 4, 1.0),
    ([2, 2, 2], [1, 1, 1], 7, 1.0),
    ([3, 7], [2, 2], 5, 100.0),
    ([1, 2, 3], [0.5, 0.25, 1.0], 9, 1.0),
    ([7], [3], 7, 1.0),
    ([0, 5], [2, 2], 4, 1.0),
    ([1, 1, 1], [1, 2, 3], 10, 1.0),
    ([5, 5, 5], [1, 2, 4], 13, 1.0),
    ([100, 1], [1, 100], 10, 0.5),
    ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], 15, 2.0),
    ([8, 8], [1, 1], 9, 1.0),
    ([3, 3, 3], [2, 2, 2], 10, 1.0),
    ([1, 1000], [1, 1], 1, 1.0),
    ([2, 4, 8], [1, 1, 1], 3, 3.0),
    ([10, 20, 30], [0.5, 0.5, 0.5], 17, 1.0),
    ([1, 2, 3, 4], [4, 3, 2, 1], 20, 1.0),
    ([6, 6, 6], [1, 1, 1], 1, 1.0),
    ([0, 3, 0, 7], [5, 5, 5, 1], 11, 1.5),
    ([2, 2, 2, 2, 2], [1, 1, 1, 1, 1], 13, 1.0),
    ([10 ** 6, 10 ** 6], [1, 3], 2000, 0.5),
    ([4, 9, 16, 25], [2.0, 0.0, 8.0, 0.5], 23, 4.0),
]


def check_example(solve):
    probs, counts, schedule, docs, flat = solve([2, 3, 5], [1, 1, 2], 7, 1.0)
    assert probs == [0.25, 0.25, 0.5]
    assert counts == [2, 2, 3]
    assert schedule == [0, 1, 2, 0, 1, 2, 2]
    assert docs == [0, 0, 0, 1, 1, 1, 2]
    assert flat == [0, 2, 5, 1, 3, 6, 7]

    probs, counts, schedule, docs, flat = solve([4, 0, 9], [1, 5, 4], 5, 2.0)
    assert abs(probs[0] - 1.0 / 3.0) < 1e-15 and probs[1] == 0.0
    assert abs(probs[2] - 2.0 / 3.0) < 1e-15
    assert counts == [2, 0, 3]
    assert schedule == [0, 2, 0, 2, 2]
    assert docs == [0, 0, 1, 1, 2]
    assert flat == [0, 4, 1, 5, 6]

    probs, counts, schedule, docs, flat = solve([5, 5], [1, 3], 8, 0.5)
    assert probs == [0.1, 0.9]
    assert counts == [1, 7]
    assert schedule == [0, 1, 1, 1, 1, 1, 1, 1]
    assert docs == [0, 0, 1, 2, 3, 4, 0, 1]
    assert flat == [0, 5, 6, 7, 8, 9, 5, 6]

    probs, counts, schedule, docs, flat = solve([1, 1, 1, 1], [1, 1, 1, 1], 6, 1.0)
    assert counts == [2, 2, 1, 1], f"ties must go to the smaller indices: {counts}"
    assert schedule == [0, 1, 2, 0, 1, 3]
    assert docs == [0, 0, 0, 0, 0, 0]
    assert flat == [0, 1, 2, 0, 1, 3]


def check_matches_oracle(solve):
    for sizes, weights, num_samples, temperature in _FIXTURES:
        _verify(solve, (sizes, weights, num_samples, temperature))


def check_quota_is_largest_remainder(solve):
    _verify(solve, ([1, 1000], [1, 1], 1, 1.0))
    probs, counts, schedule, _, _ = solve([1, 1000], [1, 1], 1, 1.0)
    assert probs == [0.5, 0.5]
    assert counts == [1, 0], f"round-half-to-even would give [0, 0]: {counts}"
    assert schedule == [0]

    _, counts, _, _, _ = solve([3, 3], [1, 1], 3, 1.0)
    assert counts == [2, 1], f"rounding each entry would overshoot to [2, 2]: {counts}"

    _, counts, _, _, _ = solve([2, 2, 2], [1, 1, 1], 7, 1.0)
    assert counts == [3, 2, 2], f"flooring every entry would stop at [2, 2, 2]: {counts}"

    sizes = [10, 20, 30, 5]
    weights = [1, 1, 1, 1]
    probs, counts, schedule, _, _ = solve(sizes, weights, 37, 1.0)
    assert sum(counts) == 37
    for index, value in enumerate(counts):
        assert abs(value - 37 / 4.0) < 1.0
        assert math.floor(37 / 4.0) <= value <= math.floor(37 / 4.0) + 1
    assert counts[0] == 10 and counts[1] == 9 and counts[2] == 9 and counts[3] == 9
    _verify(solve, (sizes, weights, 37, 1.0))


def check_temperature_flattens_and_sharpens(solve):
    sizes = [10, 10, 10]
    weights = [1, 2, 4]

    probs, counts, _, _, _ = solve(sizes, weights, 14, 1.0)
    assert probs == [1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0], "T = 1 must be the raw ratio"
    assert counts == [2, 4, 8]

    flat_probs, flat_counts, _, _, _ = solve(sizes, weights, 14, 2.0)
    assert flat_probs[0] > probs[0] and flat_probs[2] < probs[2], "T > 1 flattens"
    assert flat_counts[0] > counts[0] and flat_counts[2] < counts[2]

    sharp_probs, sharp_counts, _, _, _ = solve(sizes, weights, 14, 0.5)
    assert sharp_probs[0] < probs[0] and sharp_probs[2] > probs[2], "T < 1 sharpens"
    assert sharp_counts[0] < counts[0] and sharp_counts[2] > counts[2]

    assert sharp_probs == [1.0 / 21.0, 4.0 / 21.0, 16.0 / 21.0], "T = 0.5 squares the weights"
    assert sharp_counts == [1, 3, 10], f"largest remainder on [0.667, 2.667, 10.667]: {sharp_counts}"
    assert flat_counts == [3, 5, 6], f"T = 2 mixture quota: {flat_counts}"

    huge_probs, huge_counts, _, _, _ = solve([4, 9], [1, 10], 13, 1e6)
    assert abs(huge_probs[0] - 0.5) < 1e-5 and abs(huge_probs[1] - 0.5) < 1e-5
    assert huge_counts == [6, 7], f"a huge temperature nearly evens the budget: {huge_counts}"

    for temperature in (0.5, 1.0, 2.0, 7.0, 1e6):
        _verify(solve, (sizes, weights, 14, temperature))


def check_cyclic_oversampling(solve):
    probs, counts, schedule, docs, flat = solve([2, 2], [1, 1], 9, 1.0)
    assert counts == [5, 4]
    first = [docs[t] for t in range(9) if schedule[t] == 0]
    second = [docs[t] for t in range(9) if schedule[t] == 1]
    assert first == [0, 1, 0, 1, 0], f"emission j must read j % sizes[i]: {first}"
    assert second == [0, 1, 0, 1]
    assert len(set(flat)) == 4, "only four distinct documents exist"
    assert sorted(flat) == [0, 0, 0, 1, 1, 2, 2, 3, 3]
    _verify(solve, ([2, 2], [1, 1], 9, 1.0))

    _, counts, _, docs, flat = solve([1], [1], 4, 1.0)
    assert counts == [4] and docs == [0, 0, 0, 0] and flat == [0, 0, 0, 0]

    _, counts, schedule, docs, _ = solve([3, 3], [1, 1], 25, 1.0)
    assert counts == [13, 12]
    assert [docs[t] for t in range(25) if schedule[t] == 0] == [0, 1, 2, 0, 1, 2,
                                                              0, 1, 2, 0, 1, 2, 0]
    _verify(solve, ([3, 3], [1, 1], 25, 1.0))


def check_placement_is_by_source_order(solve):
    sizes = [10, 10]
    weights = [1, 2]
    _, counts, schedule, _, _ = solve(sizes, weights, 5, 1.0)
    assert counts == [2, 3]
    assert schedule == [0, 1, 0, 1, 1], f"sources must be placed in index order: {schedule}"
    assert schedule != _ascending_placement(counts, 5), "the two orders must disagree here"
    _verify(solve, (sizes, weights, 5, 1.0))

    sizes = [5, 5, 5]
    weights = [1, 2, 1]
    _, counts, schedule, _, _ = solve(sizes, weights, 6, 1.0)
    assert counts == [2, 3, 1]
    assert schedule == [0, 1, 1, 0, 1, 2], f"index-order placement: {schedule}"
    assert schedule != _ascending_placement(counts, 6)
    _verify(solve, (sizes, weights, 6, 1.0))


def check_first_source_sits_on_ideal_slots(solve):
    for sizes, weights, num_samples, temperature in _FIXTURES:
        if num_samples == 0:
            continue
        _, counts, schedule, _, _ = solve(sizes, weights, num_samples, temperature)
        source = next(i for i in range(len(sizes)) if counts[i] > 0)
        quota = counts[source]
        expected = [math.floor(emission * num_samples / quota) for emission in range(quota)]
        assert [slot for slot, value in enumerate(schedule) if value == source] == expected, \
            f"the first placed source must land on its ideal slots: {schedule}"
        assert len(expected) == len(set(expected)), "ideal slots must be distinct"


def check_zero_size_source_is_never_used(solve):
    sizes = [0, 5, 0, 5]
    weights = [7.0, 1.0, 9.0, 1.0]
    probs, counts, schedule, docs, flat = solve(sizes, weights, 6, 1.0)
    assert probs[0] == 0.0 and probs[2] == 0.0, "empty sources are forced to zero"
    assert counts[0] == 0 and counts[2] == 0
    assert counts[1] == 3 and counts[3] == 3
    assert set(schedule) == {1, 3}
    assert sorted(flat) == [0, 1, 2, 5, 6, 7], f"offsets skip the empty ranges: {sorted(flat)}"
    _verify(solve, (sizes, weights, 6, 1.0))

    probs, counts, _, _, _ = solve([3, 0], [1.0, 5.0], 4, 1.0)
    assert probs == [1.0, 0.0] and counts == [4, 0]

    probs, counts, _, _, _ = solve([4, 0, 6], [2.0, 3.0, 0.0], 5, 1.0)
    assert counts == [5, 0, 0], "a weight on an empty source buys nothing"
    _verify(solve, ([4, 0, 6], [2.0, 3.0, 0.0], 5, 1.0))


def check_flat_indices_are_global_ids(solve):
    sizes = [3, 1, 4]
    weights = [3, 1, 4]
    _, counts, schedule, docs, flat = solve(sizes, weights, 8, 1.0)
    assert counts == [3, 1, 4]
    offsets = [0, 3, 4]
    for slot in range(8):
        source = schedule[slot]
        assert flat[slot] == offsets[source] + docs[slot]
        assert 0 <= flat[slot] < 8
    assert docs == [0, 0, 1, 0, 1, 2, 2, 3], f"per-source counters: {docs}"
    assert flat == [0, 3, 1, 4, 5, 2, 6, 7], f"global ids: {flat}"
    _verify(solve, (sizes, weights, 8, 1.0))


def check_num_samples_zero_and_tiny(solve):
    probs, counts, schedule, docs, flat = solve([3, 4, 5], [1, 1, 1], 0, 1.0)
    assert probs == [1.0 / 3.0] * 3
    assert counts == [0, 0, 0]
    assert schedule == [] and docs == [] and flat == []
    _verify(solve, ([3, 4, 5], [1, 1, 1], 0, 1.0))
    _verify(solve, ([7], [2], 0, 3.0))

    _, counts, schedule, _, _ = solve([2, 2, 2, 2], [1, 1, 1, 1], 2, 1.0)
    assert counts == [1, 1, 0, 0], f"budget below the source count: {counts}"
    assert schedule == [0, 1]
    _verify(solve, ([2, 2, 2, 2], [1, 1, 1, 1], 2, 1.0))

    _, counts, schedule, docs, _ = solve([9, 9, 9], [1, 1, 1], 1, 2.0)
    assert counts == [1, 0, 0] and schedule == [0] and docs == [0]


def check_scale(solve):
    draw = _lcg(20261027)
    sizes = []
    weights = []
    for _ in range(40):
        sizes.append(draw(100001))
        weights.append(draw(2000) / 100.0)
    sizes[7] = 0
    sizes[19] = 0
    weights[3] = 0.0

    for num_samples, temperature in ((20000, 1.0), (5000, 0.25), (12000, 3.0), (40, 1.0)):
        _verify(solve, (sizes, weights, num_samples, temperature))
    _, counts, schedule, docs, flat = solve(sizes, weights, 20000, 1.0)
    assert sum(counts) == 20000 and len(schedule) == 20000
    assert counts[7] == 0 and counts[19] == 0 and counts[3] == 0
    assert max(docs) <= 100000
    assert max(flat) <= sum(sizes)

    big = solve([10 ** 5] * 3, [1, 1, 1], 30000, 1.0)
    assert big[1] == [10000, 10000, 10000]
    assert big[2][:6] == [0, 1, 2, 0, 1, 2], f"three interleaved ideal-slot ladders: {big[2][:6]}"
    assert [doc for slot, doc in enumerate(big[3]) if big[2][slot] == 0] == list(range(10000))
    assert len(set(big[4])) == 30000, "nothing is oversampled here"
    _verify(solve, ([10 ** 5] * 3, [1, 1, 1], 30000, 1.0))


def check_invalid(solve):
    def expect_raises(args, why):
        try:
            solve(*args)
        except ValueError:
            return
        except Exception as error:
            raise AssertionError(f"{why}: raised {type(error).__name__}, not ValueError")
        raise AssertionError(f"{why}: no error")

    good_sizes = [3, 4]
    good_weights = [1.0, 2.0]

    # sizes
    expect_raises(([], [1.0], 3, 1.0), "empty sizes")
    expect_raises((None, [], 3, 1.0), "sizes is None")
    expect_raises(((3, 4), [1.0, 2.0], 3, 1.0), "sizes is a tuple")
    expect_raises(([3.0, 4], [1.0, 1.0], 3, 1.0), "sizes entry is a float")
    expect_raises(([3, True], [1.0, 1.0], 3, 1.0), "sizes entry is a bool")
    expect_raises(([-1, 4], [1.0, 1.0], 3, 1.0), "sizes entry is negative")
    expect_raises((["3", 4], [1.0, 1.0], 3, 1.0), "sizes entry is a str")
    expect_raises(([3, None], [1.0, 1.0], 3, 1.0), "sizes entry is None")

    # weights
    expect_raises((good_sizes, None, 3, 1.0), "weights is None")
    expect_raises((good_sizes, (1.0, 2.0), 3, 1.0), "weights is a tuple")
    expect_raises((good_sizes, [1.0], 3, 1.0), "weights too short")
    expect_raises((good_sizes, [1.0, 2.0, 3.0], 3, 1.0), "weights too long")
    expect_raises((good_sizes, [1.0, "2"], 3, 1.0), "weights entry is a str")
    expect_raises((good_sizes, [1.0, None], 3, 1.0), "weights entry is None")
    expect_raises((good_sizes, [1.0, True], 3, 1.0), "weights entry is a bool")
    expect_raises((good_sizes, [1.0, -0.5], 3, 1.0), "weights entry is negative")
    expect_raises((good_sizes, [float("nan"), 1.0], 3, 1.0), "weight is NaN")
    expect_raises((good_sizes, [float("inf"), 1.0], 3, 1.0), "weight is +inf")
    expect_raises((good_sizes, [float("-inf"), 1.0], 3, 1.0), "weight is -inf")
    expect_raises((good_sizes, [10 ** 400, 1.0], 3, 1.0), "weight cannot become a float")

    # num_samples
    expect_raises((good_sizes, good_weights, -1, 1.0), "negative budget")
    expect_raises((good_sizes, good_weights, 3.0, 1.0), "budget is a float")
    expect_raises((good_sizes, good_weights, True, 1.0), "budget is a bool")
    expect_raises((good_sizes, good_weights, None, 1.0), "budget is None")
    expect_raises((good_sizes, good_weights, "3", 1.0), "budget is a str")

    # temperature
    expect_raises((good_sizes, good_weights, 3, 0), "temperature is zero")
    expect_raises((good_sizes, good_weights, 3, -1.0), "temperature is negative")
    expect_raises((good_sizes, good_weights, 3, float("nan")), "temperature is NaN")
    expect_raises((good_sizes, good_weights, 3, float("inf")), "temperature is inf")
    expect_raises((good_sizes, good_weights, 3, 10 ** 400), "temperature cannot become a float")
    expect_raises((good_sizes, good_weights, 3, True), "temperature is a bool")
    expect_raises((good_sizes, good_weights, 3, "1.0"), "temperature is a str")
    expect_raises((good_sizes, good_weights, 3, None), "temperature is None")

    # degenerate mixtures
    expect_raises(([3, 4], [0.0, 0.0], 5, 1.0), "all weights are zero")
    expect_raises(([3, 4], [0.0, 0.0], 0, 1.0), "all weights are zero with no budget")
    expect_raises(([0, 0], [1.0, 2.0], 5, 1.0), "every source is empty")
    expect_raises(([0, 4], [5.0, 0.0], 5, 1.0), "only an empty source carries weight")
    expect_raises(([3], [1e200], 5, 0.5), "the power overflows")
    expect_raises(([1, 1], [1e308, 1e308], 4, 1.0), "the total overflows")

    # legal boundaries must pass
    _verify(solve, ([5], [2], 3, 1))
    _verify(solve, ([0, 5], [9.0, 1.0], 0, 2.0))
    _verify(solve, ([1, 0], [1, 1], 1, 0.5))
    _verify(solve, ([2, 3], [0.0, 1e-300], 4, 1.0))
