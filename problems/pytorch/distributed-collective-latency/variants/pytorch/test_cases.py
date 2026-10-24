import math


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)


def _ref(W, S, a, b):
    """独立重算：展开式（区别于因式的参考解）。"""
    ring_time = 2 * (W - 1) * a + 2 * (W - 1) / W * S * b
    rd_time = math.log2(W) * a + math.log2(W) * S * b
    faster = "ring" if ring_time <= rd_time else "recursive_doubling"
    speedup = max(ring_time, rd_time) / min(ring_time, rd_time)
    return {"ring_time": ring_time, "rd_time": rd_time, "faster": faster, "speedup": speedup}


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = solve(8, 1e6, 1e-5, 1e-9)
    assert _close(out["ring_time"], 0.00189), out
    assert _close(out["rd_time"], 0.00303), out
    assert out["faster"] == "ring", out
    assert _close(out["speedup"], 1.6031746031746033), out


def check_small_message_prefers_rd(solve):
    # 消息极小、启动开销主导 → 步数少的 recursive-doubling 更快
    out = solve(8, 100.0, 1e-5, 1e-9)
    assert out["faster"] == "recursive_doubling", out
    assert _close(out["speedup"], 4.626237623762376), out


def check_matches_reference_across_sizes(solve):
    for W in (2, 4, 8, 16, 32, 64):
        for S in (1e2, 1e4, 1e6, 1e8):
            for (a, b) in ((1e-5, 1e-9), (1e-4, 5e-10)):
                out = solve(W, S, a, b)
                ref = _ref(W, S, a, b)
                assert _close(out["ring_time"], ref["ring_time"]), (W, S, a, b, out)
                assert _close(out["rd_time"], ref["rd_time"]), (W, S, a, b, out)
                assert out["faster"] == ref["faster"], (W, S, a, b, out)
                assert _close(out["speedup"], ref["speedup"]), (W, S, a, b, out)


def check_speedup_ge_one_and_consistent(solve):
    out = solve(16, 4e6, 1e-4, 5e-10)
    assert out["speedup"] >= 1.0
    lo, hi = min(out["ring_time"], out["rd_time"]), max(out["ring_time"], out["rd_time"])
    assert _close(out["speedup"], hi / lo)
    # faster 必为耗时较小者
    assert (out["faster"] == "ring") == (out["ring_time"] <= out["rd_time"])


check_example.example = True
