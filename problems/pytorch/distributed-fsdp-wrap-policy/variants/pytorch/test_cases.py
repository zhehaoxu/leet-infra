from itertools import product


def _oracle(param_sizes, wrap_size, prefetch_distance):
    length = len(param_sizes)
    if not length:
        return 0, [], 0, 0
    partitions = []
    for cuts in product((False, True), repeat=length - 1):
        ends = tuple(i + 1 for i, cut in enumerate(cuts) if cut) + (length,)
        starts = (0,) + ends[:-1]
        if all(sum(param_sizes[a:b]) <= wrap_size for a, b in zip(starts, ends)):
            partitions.append(ends)
    ends = max(partitions)
    sizes = [sum(param_sizes[a:b]) for a, b in zip((0,) + ends[:-1], ends)]
    events = []
    peak = 0
    for order, backward in ((list(range(len(sizes))), False),
                            (list(reversed(range(len(sizes)))), True)):
        resident = set()
        for position, block in enumerate(order):
            for pending in order[position:position + prefetch_distance]:
                if pending not in resident:
                    events.append(("all_gather", pending))
                    resident.add(pending)
            peak = max(peak, sum(sizes[b] for b in resident))
            events.append(("compute", block))
            resident.remove(block)
            events.append(("reshard", block))
            if backward:
                events.append(("reduce_scatter", block))
        assert not resident
    calls = sum(kind in ("all_gather", "reduce_scatter") for kind, _ in events)
    return len(sizes), sizes, peak, calls


def check_example(solve):
    assert solve([30, 40, 50, 20], 80, 2) == (2, [70, 70], 140, 6)


check_example.example = True


def check_consecutive_prefetch(solve):
    assert solve([90, 20, 90], 100, 2) == (3, [90, 20, 90], 110, 9)
    assert solve([90, 20, 90], 100, 1) == (3, [90, 20, 90], 90, 9)
    assert solve([90, 20, 90], 100, 8) == (3, [90, 20, 90], 200, 9)


def check_exact_cap_and_single_block(solve):
    assert solve([30, 50, 80], 80, 2) == (2, [80, 80], 160, 6)
    assert solve([100], 100, 1) == (1, [100], 100, 3)
    assert solve([10, 20, 30], 100, 10) == (1, [60], 60, 3)


def check_exhaustive_partition_event_oracle(solve):
    for length in range(1, 5):
        for sizes in product(range(1, 4), repeat=length):
            for cap in range(max(sizes), 7):
                for distance in (1, 2, length + 2):
                    args = (list(sizes), cap, distance)
                    assert solve(*args) == _oracle(*args), args


def check_uneven_partition_event_oracle(solve):
    for sizes, cap in [([12, 23, 34, 45, 56], 60), ([8, 1, 6, 4, 2, 9, 3], 10),
                       ([15, 15, 15, 15], 20), ([5] * 7, 25)]:
        for distance in (1, 2, 3, 20):
            before = sizes[:]
            result = solve(sizes, cap, distance)
            assert isinstance(result, tuple) and len(result) == 4
            assert result == _oracle(sizes, cap, distance)
            assert sizes == before
            assert sum(result[1]) == sum(sizes)
            assert all(0 < size <= cap for size in result[1])
            assert isinstance(result[1], list)
            assert all(type(result[i]) is int for i in (0, 2, 3))
            assert all(type(size) is int for size in result[1])


def check_empty_params(solve):
    assert solve([], 100, 1) == (0, [], 0, 0)
    assert solve([], 1, 100) == (0, [], 0, 0)


def check_invalid_layer_sizes(solve):
    for sizes in ([0], [-1], [101], [1, 0, 2], [1.0], [True], ["1"],
                  [[1]], None, (1, 2)):
        try:
            solve(sizes, 100, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {sizes!r}")


def check_invalid_parameters(solve):
    for invalid in (0, -1, 1.5, True, "2", None):
        for sizes in ([], [1]):
            for cap, distance in ((invalid, 2), (100, invalid)):
                try:
                    solve(sizes, cap, distance)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"expected ValueError for {(sizes, cap, distance)!r}")
