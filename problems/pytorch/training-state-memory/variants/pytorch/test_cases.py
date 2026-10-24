def _ref(n_params, world):
    """独立重算（按分量拼装，不复用 solve 的化简式），交叉核对每卡字节。"""
    N = float(n_params)
    W = float(world)
    param_fp16 = 2.0 * N
    grad_fp16 = 2.0 * N
    master = 4.0 * N
    mom = 4.0 * N
    var = 4.0 * N
    opt = master + mom + var                      # 12N
    stage0 = param_fp16 + grad_fp16 + opt
    stage1 = param_fp16 + grad_fp16 + opt / W
    stage2 = param_fp16 + (grad_fp16 + opt) / W
    stage3 = (param_fp16 + grad_fp16 + opt) / W
    return (stage0, stage1, stage2, stage3)


def _close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    got = solve(1_000_000_000, 8)
    want = (16000000000.0, 5500000000.0, 3750000000.0, 2000000000.0)
    assert len(got) == 4
    for g, w in zip(got, want):
        assert _close(g, w), (got, want)


def check_matches_ref(solve):
    for args in [(7_000_000_000, 16), (1_500_000_000, 4), (375_000_000, 1), (1_000_000, 64)]:
        got = solve(*args)
        want = _ref(*args)
        for g, w in zip(got, want):
            assert _close(g, w), (args, got, want)


def check_monotonic_reduction(solve):
    # 分区档位越高，每卡显存越低；world=1 时四档应相等
    s0, s1, s2, s3 = solve(2_000_000_000, 8)
    assert s0 >= s1 >= s2 >= s3, (s0, s1, s2, s3)
    single = solve(2_000_000_000, 1)
    assert all(_close(v, single[0]) for v in single), single


def check_world_one_is_full(solve):
    N = 500_000_000
    s0, s1, s2, s3 = solve(N, 1)
    assert _close(s0, 16.0 * N), s0
    assert _close(s3, 16.0 * N), s3


check_example.example = True
