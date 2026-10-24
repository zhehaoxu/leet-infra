import torch


def _ref(n_elems, dtype_bytes, num_ops):
    """独立重算参考（不复用 solve 路径）。"""
    non_fused = 2 * num_ops * n_elems * dtype_bytes
    fused = 2 * n_elems * dtype_bytes
    return non_fused, fused, non_fused - fused, non_fused / fused


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = solve(1 << 20, 2, 5)
    assert out == (20971520, 4194304, 16777216, 5.0), out


def check_matches_ref(solve):
    for args in [(1024, 4, 1), (8192, 2, 3), (1 << 22, 1, 8)]:
        n, dt, m = args
        got = tuple(solve(*args))
        exp = _ref(*args)
        assert got[0] == exp[0] and got[1] == exp[1] and got[2] == exp[2], args
        assert abs(got[3] - exp[3]) < 1e-9, args


def check_speedup_equals_ops(solve):
    # memory-bound 模型下加速比恰为算子数
    for m in (1, 2, 6, 10):
        _, _, _, sp = solve(4096, 2, m)
        assert abs(sp - m) < 1e-9, (m, sp)


def check_saved_zero_when_single_op(solve):
    # 单个算子无可融合对象，节省应为 0
    nf, fu, sv, sp = solve(1 << 16, 2, 1)
    assert sv == 0 and nf == fu and sp == 1.0


check_example.example = True
