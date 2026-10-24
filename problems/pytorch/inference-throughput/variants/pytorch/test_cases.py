def _ref(n_params, batch, prompt_len, gen_len, effective_flops):
    """独立重算：先算每 token 平均、再分别乘 token 数，时间用逐阶段相加。"""
    per_tok = 2 * int(n_params)
    pre = batch * prompt_len * per_tok
    dec = batch * gen_len * per_tok
    ttft = float(pre) / float(effective_flops)
    total = float(pre) / float(effective_flops) + float(dec) / float(effective_flops)
    return pre, dec, ttft, total


def _close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    got = solve(7_000_000_000, 1, 512, 256, 1e14)
    assert got[0] == 7168000000000, got
    assert got[1] == 3584000000000, got
    assert _close(got[2], 0.07168) and _close(got[3], 0.10752), got


def check_matches_ref(solve):
    cases = [(1_000_000_000, 4, 128, 64, 3e13), (7_000_000_000, 1, 512, 256, 1e14),
             (4096, 1, 1, 1, 1e9), (13_000_000_000, 2, 2048, 512, 2e14)]
    for c in cases:
        got = solve(*c)
        want = _ref(*c)
        assert got[0] == want[0] and got[1] == want[1], (c, got, want)
        assert _close(got[2], want[2]) and _close(got[3], want[3]), (c, got, want)


def check_zero_generation(solve):
    # gen_len=0：无 decode，total 应等于 ttft
    pre, dec, ttft, total = solve(7e9, 1, 100, 0, 1e14)
    assert dec == 0, dec
    assert _close(total, ttft), (ttft, total)


def check_scales_linearly(solve):
    base = solve(1_000_000, 1, 10, 10, 1e12)
    b4 = solve(1_000_000, 4, 10, 10, 1e12)      # batch×4 → flops×4，时间×4
    assert b4[0] == base[0] * 4 and b4[1] == base[1] * 4
    assert _close(b4[3], base[3] * 4), (base, b4)
    # effective_flops×2 → 时间减半、FLOPs 不变
    fast = solve(1_000_000, 1, 10, 10, 2e12)
    assert fast[0] == base[0] and _close(fast[3], base[3] / 2), (base, fast)


def check_total_ge_ttft(solve):
    for c in [(7_000_000_000, 3, 256, 128, 1e14), (500, 1, 7, 9, 1e6)]:
        _, _, ttft, total = solve(*c)
        assert total >= ttft - 1e-12, (ttft, total)


check_example.example = True
