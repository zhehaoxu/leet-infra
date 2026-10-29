def _assert_invariants(solve, world_size, failed_ranks, param_bytes):
    before = failed_ranks[:]
    result = solve(world_size, failed_ranks, param_bytes)
    assert isinstance(result, tuple) and len(result) == 4
    new_world, remap, destinations, payload = result
    failures = set(failed_ranks)
    checkpoints = sorted(failures)
    assert failed_ranks == before
    assert type(new_world) is int and new_world == world_size - len(failures)
    assert isinstance(remap, list) and len(remap) == world_size
    assert isinstance(destinations, list) and len(destinations) == len(checkpoints)
    assert all(type(value) is int for value in remap + destinations)
    for old_rank, new_rank in enumerate(remap):
        if old_rank in failures:
            assert new_rank == -1
        else:
            assert new_rank == old_rank - sum(failed < old_rank for failed in failures)
    assert sorted(rank for rank in remap if rank >= 0) == list(range(new_world))
    assert all(0 <= destination < new_world for destination in destinations)
    extra_loads = [0] * new_world
    for new_rank in range(new_world):
        restored = [old for old, dest in zip(checkpoints, destinations) if dest == new_rank]
        assert restored == checkpoints[new_rank::new_world]
        extra_loads[new_rank] = len(restored)
    quotient, remainder = divmod(len(checkpoints), new_world)
    assert extra_loads == [quotient + (rank < remainder) for rank in range(new_world)]
    assert max(extra_loads) - min(extra_loads) <= 1
    shard_bytes = param_bytes // world_size
    assert type(payload) is int and payload >= 0
    assert new_world * shard_bytes + payload == param_bytes
    assert payload == sum(shard_bytes for old in range(world_size) if old in failures)
    return result


def check_example(solve):
    assert solve(4, [1, 3], 16) == (2, [0, -1, 1, -1], [0, 1], 8)


check_example.example = True


def check_unsorted_and_duplicate_failures(solve):
    expected = (3, [-1, 0, -1, 1, -1, -1, 2, -1], [0, 1, 2, 0, 1], 40)
    assert _assert_invariants(solve, 8, [7, 2, 5, 0, 4], 64) == expected
    assert _assert_invariants(solve, 8, [7, 2, 7, 5, 0, 4, 2], 64) == expected
    assert _assert_invariants(solve, 3, [1] * 10, 9) == (2, [0, -1, 1], [0], 3)


def check_exhaustive_failure_subsets(solve):
    for world_size in range(1, 9):
        for mask in range((1 << world_size) - 1):
            failed = [rank for rank in reversed(range(world_size)) if mask & (1 << rank)]
            _assert_invariants(solve, world_size, failed, world_size * 7)


def check_only_one_survivor(solve):
    assert _assert_invariants(solve, 5, [4, 0, 3, 1], 25) == (
        1, [-1, -1, 0, -1, -1], [0, 0, 0, 0], 20
    )


def check_no_failure(solve):
    assert _assert_invariants(solve, 4, [], 16) == (4, [0, 1, 2, 3], [], 0)
    assert _assert_invariants(solve, 1, [], 5) == (1, [0], [], 0)


def check_extra_loads_include_idle_survivors(solve):
    for world_size, failed in ((8, [0, 1, 2, 3, 4]), (8, [7, 2]), (5, [4]), (4, [])):
        _assert_invariants(solve, world_size, failed, world_size * 10)


def check_zero_payload_and_large_integers(solve):
    _assert_invariants(solve, 6, [5, 1, 3], 0)
    _assert_invariants(solve, 3, [], 0)
    _assert_invariants(solve, 7, [6, 4, 0], 7 * (2**60 + 1))


def check_all_failed(solve):
    for world_size, failed in ((1, [0]), (4, [0, 1, 2, 3]), (3, [2, 0, 1, 2])):
        try:
            solve(world_size, failed, world_size * 4)
        except ValueError:
            pass
        else:
            raise AssertionError("all failed ranks must raise ValueError")


def check_invalid_rank_ids(solve):
    for failed in ([-1], [4], [0, 4], [1.0], [True], ["1"], [[1]], None, (1, 2)):
        try:
            solve(4, failed, 16)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {failed!r}")


def check_invalid_world_size_and_bytes(solve):
    cases = [(world, [], 0) for world in (0, -1, 2.0, True, "4", None)]
    cases += [(4, [], size) for size in (-4, 15, 16.0, True, "16", None)]
    for args in cases:
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {args!r}")
