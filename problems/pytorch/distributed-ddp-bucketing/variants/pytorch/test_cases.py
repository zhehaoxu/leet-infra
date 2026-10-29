from itertools import product


def _prefix_buckets(param_sizes, cap):
    # Independent boundary oracle: enumerate all candidate contiguous prefixes,
    # select the longest legal one, and repeat. No mutable greedy bucket logic.
    order = list(range(len(param_sizes) - 1, -1, -1))
    prefix = [0]
    for index in order:
        prefix.append(prefix[-1] + param_sizes[index])
    result = []
    start = 0
    while start < len(order):
        end = max(
            stop for stop in range(start + 1, len(order) + 1)
            if stop == start + 1 or prefix[stop] - prefix[start] <= cap
        )
        result.append(order[start:end])
        start = end
    return result


def _assert_case(solve, sizes, cap_mb, ranks, times):
    original_sizes = sizes[:]
    original_times = [row[:] for row in times]
    buckets = _prefix_buckets(original_sizes, int(cap_mb * 1024**2))
    # Compute all expected times directly from gradient entries; do not reuse
    # the solution's rank-reduction followed by running-maximum recurrence.
    per_rank = [
        [max(row[index] for index in bucket) for bucket in buckets]
        for row in original_times
    ]
    global_times = [
        max(row[index] for row in original_times for index in bucket)
        for bucket in buckets
    ]
    enqueue = [
        max(row[index] for row in original_times
            for bucket in buckets[:k + 1] for index in bucket)
        for k in range(len(buckets))
    ]
    expected = {
        'buckets': buckets,
        'bucket_ready_times': per_rank,
        'global_ready_times': global_times,
        'enqueue_times': enqueue,
    }
    got = solve(sizes, cap_mb, ranks, times)
    assert sizes == original_sizes and times == original_times, 'inputs were mutated'
    assert isinstance(got, dict) and set(got) == set(expected)
    assert all(isinstance(got[key], list) for key in expected)
    assert all(isinstance(bucket, list) for bucket in got['buckets'])
    assert all(isinstance(row, list) for row in got['bucket_ready_times'])
    assert got == expected, f'got {got}, expected {expected}'
    return got


def check_example(solve):
    sizes = [3 * 1024**2, 2 * 1024**2, 4 * 1024**2, 1024**2, 2 * 1024**2]
    times = [[1., 2., 4., 3., 8.], [2., 3., 6., 1., 5.]]
    got = _assert_case(solve, sizes, 5.0, 2, times)
    assert got['buckets'] == [[4, 3], [2], [1, 0]]
    assert got['bucket_ready_times'] == [[8., 4., 2.], [5., 6., 3.]]
    assert got['global_ready_times'] == [8., 6., 3.]
    assert got['enqueue_times'] == [8., 8., 8.]
    print('param_sizes =', sizes)
    print('bucket_cap_mb = 5.0, world_size = 2')
    print('ready_times =', times)
    print(got)


check_example.example = True


def check_readiness_does_not_determine_membership(solve):
    sizes = [2, 1, 2, 1, 2, 1]
    times = [[12., 1., 2., 9., 3., 0.], [4., 2., 8., 1., 0., 7.]]
    got = _assert_case(solve, sizes, 3 / 1024**2, 2, times)
    changed = _assert_case(solve, sizes, 3 / 1024**2, 2,
                           [list(reversed(row)) for row in times])
    assert got['buckets'] == changed['buckets'] == [[5, 4], [3, 2], [1, 0]]
    assert got['bucket_ready_times'] == [[3., 9., 12.], [7., 8., 4.]]


def check_five_buckets_two_ranks_late_rank(solve):
    got = _assert_case(solve, [1] * 5, 1 / 1024**2, 2,
                       [[1., 2., 3., 4., 5.], [0., 0., 0., 0., 20.]])
    assert got['buckets'] == [[4], [3], [2], [1], [0]]
    assert got['global_ready_times'] == [20., 4., 3., 2., 1.]
    assert got['enqueue_times'] == [20.] * 5


def check_earlier_ready_later_buckets_are_blocked(solve):
    got = _assert_case(solve, [1] * 4, 1 / 1024**2, 2,
                       [[2., 4., 9., 1.], [3., 5., 7., 0.]])
    assert got['global_ready_times'] == [1., 9., 5., 3.]
    assert got['enqueue_times'] == [1., 9., 9., 9.]


def check_exact_capacity(solve):
    got = _assert_case(solve, [2, 3, 1, 4], 5 / 1024**2, 2,
                       [[2., 0., 1., 3.], [0., 4., 1., 2.]])
    assert got['buckets'] == [[3, 2], [1, 0]]


def check_oversized_middle_gradient(solve):
    got = _assert_case(solve, [2, 2, 9, 1, 2], 4 / 1024**2, 2,
                       [[1., 7., 3., 2., 0.], [4., 0., 2., 9., 1.]])
    assert got['buckets'] == [[4, 3], [2], [1, 0]]


def check_fractional_mib_and_floor(solve):
    got = _assert_case(solve, [1, 1, 5], 5.75 / 1024**2, 1, [[0., 2., 1.]])
    assert got['buckets'] == [[2], [1, 0]], 'floor(5.75) is 5, not 6'
    got = _assert_case(solve, [262144, 262144, 524288, 1], 0.5, 2,
                       [[0., 2., 1., 0.], [3., 0., 2., 1.]])
    assert got['buckets'] == [[3], [2], [1, 0]], 'capacity uses MiB, not decimal MB'


def check_zero_times(solve):
    got = _assert_case(solve, [2, 8, 1, 2], 3 / 1024**2, 3, [[0.] * 4 for _ in range(3)])
    assert got['global_ready_times'] == got['enqueue_times'] == [0., 0., 0.]


def check_empty_parameters(solve):
    for ranks in (1, 2, 5):
        got = _assert_case(solve, [], 0.5, ranks, [[] for _ in range(ranks)])
        assert got == {
            'buckets': [], 'bucket_ready_times': [[] for _ in range(ranks)],
            'global_ready_times': [], 'enqueue_times': [],
        }


def check_one_rank_and_one_bucket(solve):
    got = _assert_case(solve, [2, 3, 4], 1.0, 1, [[7., 0., 2.]])
    assert got['buckets'] == [[2, 1, 0]]
    assert got['bucket_ready_times'] == [[7.]]
    _assert_case(solve, [8], 1 / 1024**2, 1, [[0.25]])


def check_exhaustive_small_prefix_boundaries(solve):
    # Covers oversized first/last items, multiple oversized items, one-byte
    # capacities, partial final buckets, and rank counts unrelated to K.
    for length in range(5):
        for values in product((1, 2, 4), repeat=length):
            sizes = list(values)
            ranks = 1 + sum(sizes) % 3
            times = [
                [float((11 * size + 7 * i + 3 * rank) % 19) / 4
                 for i, size in enumerate(sizes)]
                for rank in range(ranks)
            ]
            for cap in (1, 2, 3, 5):
                _assert_case(solve, sizes, cap / 1024**2, ranks, times)
