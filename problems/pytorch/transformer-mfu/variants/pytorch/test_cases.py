import torch


def _ref(b, s, layers, hidden, ffn, peak_tflops, step_ms):
    """独立重算参考（不复用 solve 路径）。"""
    p_layer = 4 * hidden * hidden + 2 * hidden * ffn
    c = 2 * p_layer + 4 * s * hidden
    forward = b * s * layers * c
    step = 3 * forward
    mfu = (step / (step_ms / 1000.0) / 1e12) / peak_tflops
    return forward, step, mfu


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    fwd, step, mfu = solve(4, 1024, 6, 512, 2048, 150.0, 8.0)
    assert (fwd, step) == (206158430208, 618475290624), (fwd, step)
    assert abs(mfu - 0.5153961) < 1e-5, mfu


def check_matches_ref(solve):
    for args in [(2, 256, 4, 128, 512, 100.0, 10.0),
                 (8, 4096, 12, 1024, 4096, 300.0, 25.0),
                 (1, 128, 2, 64, 256, 60.0, 1.0)]:
        fwd, step, mfu = solve(*args)
        rf, rs, rm = _ref(*args)
        assert (fwd, step) == (rf, rs), args
        assert abs(mfu - rm) < 1e-9, args


def check_backward_is_three_forward(solve):
    fwd, step, _ = solve(4, 512, 6, 256, 1024, 120.0, 5.0)
    assert step == 3 * fwd


def check_mfu_monotonic_in_time(solve):
    # 同 FLOPs 下 step_ms 越大 → MFU 越低
    _, _, mfu_fast = solve(2, 512, 4, 256, 1024, 150.0, 4.0)
    _, _, mfu_slow = solve(2, 512, 4, 256, 1024, 150.0, 8.0)
    assert mfu_fast > mfu_slow
    assert abs(mfu_fast / mfu_slow - 2.0) < 1e-9, (mfu_fast, mfu_slow)


check_example.example = True
