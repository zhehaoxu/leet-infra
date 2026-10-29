import heapq
import math


def _owner(c, p):
    sweep, offset = divmod(c, p)
    return offset if sweep % 2 == 0 else p - offset - 1


def _oracle(p, m, v):
    successors, degree = {}, {}
    for kind in ("F", "B"):
        for c in range(p * v):
            for b in range(m):
                op = (kind, c, b)
                deps = []
                if kind == "F" and c:
                    deps.append(("F", c - 1, b))
                if kind == "B":
                    deps.append(("F", c, b))
                    if c + 1 < p * v:
                        deps.append(("B", c + 1, b))
                degree[op] = len(deps)
                for dep in deps:
                    successors.setdefault(dep, []).append(op)
    ready = [[] for _ in range(p)]

    def push(op):
        kind, c, b = op
        heapq.heappush(ready[_owner(c, p)], (0 if kind == "B" else 1, b, -c if kind == "B" else c, op))

    for op, d in degree.items():
        if not d:
            push(op)
    rows = [[] for _ in range(p)]
    remaining = len(degree)
    while remaining:
        selected = [heapq.heappop(q)[-1] if q else None for q in ready]
        assert any(op is not None for op in selected)
        for r, op in enumerate(selected):
            rows[r].append(op)
        for op in selected:
            if op is not None:
                remaining -= 1
                for child in successors.get(op, []):
                    degree[child] -= 1
                    if degree[child] == 0:
                        push(child)
    return rows


def _verify(solve, p, m, v, layers, act):
    got = solve(p, m, v, layers, act)
    timeline = got["timeline"]
    assert timeline == _oracle(p, m, v), got
    t = got["makespan"]
    assert type(t) is int and t > 0
    assert len(timeline) == p and all(len(row) == t for row in timeline)
    done, live, peak = {}, [0] * p, [0] * p
    for tick in range(t):
        completed = []
        for r in range(p):
            op = timeline[r][tick]
            assert type(got["busy_table"][r][tick]) is bool
            assert got["busy_table"][r][tick] == (op is not None)
            if op is None:
                continue
            kind, c, b = op
            assert _owner(c, p) == r and op not in done
            if kind == "F" and c:
                assert ("F", c - 1, b) in done
            if kind == "B":
                assert ("F", c, b) in done
                if c + 1 < p * v:
                    assert ("B", c + 1, b) in done
            live[r] += 1 if kind == "F" else -1
            assert live[r] >= 0
            peak[r] = max(peak[r], live[r])
            completed.append(op)
        for op in completed:
            done[op] = tick
    assert len(done) == 2 * p * v * m and live == [0] * p
    unit = layers // (p * v) * act
    expected_peaks = [x * unit for x in peak]
    assert got["peak_activation_bytes"] == expected_peaks
    assert got["flush_peak_activation_bytes"] == [m * v * unit] * p
    assert got["memory_saved_bytes"] == p * m * v * unit - sum(expected_peaks)
    busy = sum(sum(row) for row in got["busy_table"])
    assert busy == len(done)
    assert math.isclose(got["util"], busy / (p * t), abs_tol=1e-12)
    assert math.isclose(got["bubble"] + got["util"], 1.0, abs_tol=1e-12)
    return got


def check_example(solve):
    out = _verify(solve, 2, 2, 2, 8, 1.0)
    assert out["timeline"][0][0] == ("F", 0, 0)
    assert out["timeline"][1][0] is None
    assert out["makespan"] == 10
    assert out["util"] == 0.8
    assert out["peak_activation_bytes"] == [6.0, 8.0]
    assert out["flush_peak_activation_bytes"] == [8.0, 8.0]
    assert out["memory_saved_bytes"] == 2.0


check_example.example = True


def check_grid_and_single_microbatch(solve):
    for p in (1, 2, 3, 4):
        for m in (1, 2, 5):
            for v in (1, 2, 3):
                _verify(solve, p, m, v, p * v * 2, 3.0)


def check_single_device_serial_work(solve):
    for v in (1, 2, 4):
        out = _verify(solve, 1, 5, v, v * 3, 2.0)
        assert out["makespan"] == 10 * v
        assert out["bubble"] == 0.0


def check_terminal_backward_has_priority(solve):
    out = _verify(solve, 2, 4, 2, 8, 1.0)
    terminal = 3
    row = out["timeline"][0]
    t = row.index(("F", terminal, 0))
    assert row[t + 1] == ("B", terminal, 0)
    assert out["memory_saved_bytes"] > 0


def check_memory_scaling_and_zero(solve):
    a = _verify(solve, 3, 4, 2, 12, 2.0)
    b = _verify(solve, 3, 4, 2, 24, 3.0)
    z = _verify(solve, 3, 4, 2, 12, 0.0)
    assert a["timeline"] == b["timeline"] == z["timeline"]
    assert b["peak_activation_bytes"] == [3 * x for x in a["peak_activation_bytes"]]
    assert z["peak_activation_bytes"] == [0.0] * 3
    assert z["memory_saved_bytes"] == 0.0


def check_bounds(solve):
    _verify(solve, 8, 32, 4, 8192, 0.5)


def check_invalid_layer_partition(solve):
    for layers in (7, 10):
        try:
            solve(2, 2, 2, layers, 1.0)
        except ValueError:
            continue
        raise AssertionError("nonintegral chunk sizes must be rejected")
