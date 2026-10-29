"""Exact rational expectations and independent cost-line invariants."""

from copy import deepcopy
from fractions import Fraction
import math
import random


EXAMPLE_SIZES = [0, 8000, 16000]
EXAMPLE_LATENCIES = [[1.0, 4.0, 7.0], [4.0, 7.0, 10.0]]
EXAMPLE_BANDWIDTHS = [[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]]


def _table(values):
    return [[float(value) for value in values[:3]],
            [float(value) for value in values[3:]]]


def _close(actual, expected):
    assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
    assert math.isfinite(actual), actual
    assert math.isclose(actual, float(expected), rel_tol=2e-12, abs_tol=1e-12), (
        actual, float(expected)
    )


def _assert_rational_model(solve, sizes, latencies, bandwidths):
    """Cross-multiply exact rationals, rather than repeat float slope code."""
    saved = deepcopy((sizes, latencies, bandwidths))
    out = solve(sizes, latencies, bandwidths)
    assert (sizes, latencies, bandwidths) == saved, "Inputs were mutated"
    assert isinstance(out, dict)
    assert set(out) == {
        "times_us", "best", "thresholds_bytes", "achieved_gbps", "utilization"
    }
    for key in ("times_us", "best", "achieved_gbps", "utilization"):
        assert isinstance(out[key], list) and len(out[key]) == len(sizes), key

    # Integer and binary-fraction fixtures give exact, independent arithmetic.
    ls = [Fraction(value) for row in latencies for value in row]
    bs = [Fraction(value) for row in bandwidths for value in row]
    for row, size in enumerate(sizes):
        actual_times = out["times_us"][row]
        assert isinstance(actual_times, list) and len(actual_times) == 6
        costs = [(1000 * l * b + size) / (1000 * b) for l, b in zip(ls, bs)]
        for actual, expected in zip(actual_times, costs):
            _close(actual, expected)
        # Find the lower envelope by pairwise exact comparisons, including ties.
        winners = [i for i in range(6) if all(costs[i] <= c for c in costs)]
        winner = out["best"][row]
        assert type(winner) is int and winner == winners[0], (size, winner, winners)
        achieved = Fraction(size, 1000) / costs[winner]
        _close(out["achieved_gbps"][row], achieved)
        _close(out["utilization"][row], achieved / bs[winner])
        assert 0 <= out["utilization"][row] <= 1

    matrix = out["thresholds_bytes"]
    assert isinstance(matrix, list) and len(matrix) == 6
    assert all(isinstance(row, list) and len(row) == 6 for row in matrix)
    for i in range(6):
        for j in range(6):
            actual = matrix[i][j]
            assert actual == matrix[j][i], (i, j, "asymmetric threshold")
            if i == j or bs[i] == bs[j]:
                assert actual is None, (i, j, "parallel/diagonal")
                continue
            # Multiply the equality by 1000*Bi*Bj to avoid reciprocal slopes.
            rhs = 1000 * bs[i] * bs[j] * (ls[j] - ls[i])
            crossing = rhs / (bs[j] - bs[i])
            if crossing < 0:
                assert actual is None, (i, j, "negative crossing")
            else:
                _close(actual, crossing)
                assert actual >= 0
                # Independently substitute every returned crossing into both lines.
                left = float(ls[i]) + actual / (1000 * float(bs[i]))
                right = float(ls[j]) + actual / (1000 * float(bs[j]))
                _close(left, right)
    return out


def check_example(solve):
    out = _assert_rational_model(
        solve, EXAMPLE_SIZES[:], deepcopy(EXAMPLE_LATENCIES), deepcopy(EXAMPLE_BANDWIDTHS)
    )
    assert out["times_us"] == [
        [1.0, 4.0, 7.0, 4.0, 7.0, 10.0],
        [9.0, 8.0, 9.0, 12.0, 11.0, 12.0],
        [17.0, 12.0, 11.0, 20.0, 15.0, 14.0],
    ]
    assert out["best"] == [0, 1, 2]


check_example.example = True


def check_actual_lower_envelope_switches(solve):
    out = _assert_rational_model(
        solve, [5999, 6000, 6001, 11999, 12000, 12001],
        deepcopy(EXAMPLE_LATENCIES), deepcopy(EXAMPLE_BANDWIDTHS)
    )
    assert out["best"] == [0, 0, 1, 1, 1, 2]
    _close(out["thresholds_bytes"][0][1], 6000)
    _close(out["thresholds_bytes"][1][2], 12000)


def check_dominated_pairwise_crossing(solve):
    out = _assert_rational_model(
        solve, [7999, 8000, 8001],
        deepcopy(EXAMPLE_LATENCIES), deepcopy(EXAMPLE_BANDWIDTHS)
    )
    _close(out["thresholds_bytes"][0][2], 8000)
    assert out["times_us"][1][0] == out["times_us"][1][2] == 9.0
    assert out["times_us"][1][1] == 8.0
    assert out["best"] == [1, 1, 1], "A pairwise crossing need not switch the winner"


def check_zero_negative_parallel_crossings(solve):
    out = _assert_rational_model(
        solve, [0, 1, 8000], deepcopy(EXAMPLE_LATENCIES), deepcopy(EXAMPLE_BANDWIDTHS)
    )
    matrix = out["thresholds_bytes"]
    assert matrix[1][3] == matrix[3][1] == 0.0
    assert matrix[2][4] == matrix[4][2] == 0.0
    assert matrix[2][3] == 4000.0
    assert matrix[0][3] is None  # Same bandwidth, different latency.
    assert all(matrix[i][i] is None for i in range(6))
    negative = _assert_rational_model(
        solve, [0, 8000], _table([1, 2, 3, 4, 5, 6]), _table([4, 2, 1, 4, 2, 1])
    )
    assert negative["thresholds_bytes"][0][1] is None  # Intersection at -4000 bytes.


def check_identical_lines_and_stable_ties(solve):
    out = _assert_rational_model(
        solve, [0, 1, 2000, 10**9],
        _table([1, 1, 2, 3, 4, 5]), _table([2] * 6)
    )
    assert out["thresholds_bytes"] == [[None] * 6 for _ in range(6)]
    assert out["best"] == [0] * 4
    tree_tie = _assert_rational_model(
        solve, [0, 1, 2000], _table([9, 9, 9, 1, 1, 1]), _table([2] * 6)
    )
    assert tree_tie["best"] == [3, 3, 3]


def check_near_ties_are_not_exact_ties(solve):
    out = _assert_rational_model(
        solve, [0, 1], _table([1 + 2**-40, 1, 2, 3, 4, 5]), _table([1] * 6)
    )
    assert out["best"] == [1, 1], "Do not apply an epsilon to winner selection"


def check_zero_size_uses_only_latency(solve):
    out = _assert_rational_model(
        solve, [0], _table([5, 4, 3, 1, 1, 2]), _table([100, 50, 25, 1, 8, 4])
    )
    assert out["best"] == [3]
    assert out["achieved_gbps"] == out["utilization"] == [0.0]
    assert out["times_us"] == [[5.0, 4.0, 3.0, 1.0, 1.0, 2.0]]


def check_all_six_flat_candidate_indices(solve):
    for target in range(6):
        latencies = [9] * 6
        bandwidths = [1] * 6
        latencies[target], bandwidths[target] = 1, 100
        out = _assert_rational_model(
            solve, [0, 1, 10**6], _table(latencies), _table(bandwidths)
        )
        assert out["best"] == [target] * 3


def check_decimal_gigabytes_and_microseconds(solve):
    out = _assert_rational_model(
        solve, [0, 1000, 2000], _table([0.5] * 6), _table([2] * 6)
    )
    assert out["times_us"] == [[0.5] * 6, [1.0] * 6, [1.5] * 6]
    _close(out["achieved_gbps"][2], Fraction(4, 3))
    _close(out["utilization"][2], Fraction(2, 3))


def check_fractional_calibrations(solve):
    _assert_rational_model(
        solve, [0, 1, 125, 1000, 10**6],
        _table([0.125, 0.25, 0.5, 0.75, 1.25, 2.5]),
        _table([0.125, 0.5, 1.25, 2.5, 8, 16])
    )


def check_nearly_parallel_is_not_parallel(solve):
    out = _assert_rational_model(
        solve, [0, 10**12], _table([1, 2, 3, 4, 5, 6]),
        _table([1, 1 + 2**-20, 2, 4, 8, 16])
    )
    assert out["thresholds_bytes"][0][1] > 10**9


def check_byte_and_bandwidth_scaling(solve):
    sizes = [0, 4000, 8000, 16000]
    latencies = deepcopy(EXAMPLE_LATENCIES)
    bandwidths = deepcopy(EXAMPLE_BANDWIDTHS)
    base = _assert_rational_model(solve, sizes, latencies, bandwidths)
    factor = 1000
    scaled = _assert_rational_model(
        solve, [size * factor for size in sizes], latencies,
        [[value * factor for value in row] for row in bandwidths]
    )
    assert scaled["times_us"] == base["times_us"]
    assert scaled["best"] == base["best"]
    for row in range(len(sizes)):
        _close(scaled["achieved_gbps"][row], factor * base["achieved_gbps"][row])
        _close(scaled["utilization"][row], base["utilization"][row])
    for i in range(6):
        for j in range(6):
            crossing = base["thresholds_bytes"][i][j]
            if crossing is None:
                assert scaled["thresholds_bytes"][i][j] is None
            else:
                _close(scaled["thresholds_bytes"][i][j], crossing * factor)


def check_empty_sizes_still_have_thresholds(solve):
    out = _assert_rational_model(
        solve, [], deepcopy(EXAMPLE_LATENCIES), deepcopy(EXAMPLE_BANDWIDTHS)
    )
    assert all(out[key] == [] for key in (
        "times_us", "best", "achieved_gbps", "utilization"
    ))
    assert out["thresholds_bytes"][0][1] == 6000.0


def check_unsorted_repeated_sizes(solve):
    out = _assert_rational_model(
        solve, [16000, 0, 8000, 16000, 1, 0],
        deepcopy(EXAMPLE_LATENCIES), deepcopy(EXAMPLE_BANDWIDTHS)
    )
    assert out["best"] == [2, 0, 1, 2, 0, 0]
    assert out["times_us"][0] == out["times_us"][3]
    assert out["times_us"][1] == out["times_us"][5]


def check_seeded_integer_fraction_oracle(solve):
    rng = random.Random(439)
    for _ in range(32):
        latencies = _table([rng.randint(1, 40) for _ in range(6)])
        bandwidths = _table([rng.randint(1, 32) for _ in range(6)])
        sizes = [0, 1, 999, 1000, 1024, 10**9]
        sizes += [rng.randrange(1, 10**7) for _ in range(12)]
        _assert_rational_model(solve, sizes, latencies, bandwidths)
