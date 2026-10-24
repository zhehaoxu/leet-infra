def _ref(message_bytes, world, bandwidth_bps, latency_per_hop=0.0):
    """独立重算：分别按 reduce-scatter 与 all-gather 两阶段累加，再求耗时。"""
    S = float(message_bytes)
    W = int(world)
    if W <= 1:
        return (0.0, 0.0)
    chunk = S / W
    steps = W - 1
    reduce_scatter_bytes = steps * chunk
    all_gather_bytes = steps * chunk
    bytes_per_rank = reduce_scatter_bytes + all_gather_bytes
    total_hops = 2 * steps
    time = bytes_per_rank / float(bandwidth_bps) + total_hops * float(latency_per_hop)
    return (bytes_per_rank, time)


def _close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    got = solve(1e9, 8, 1e11)
    assert _close(got[0], 1750000000.0), got
    assert _close(got[1], 0.0175), got


def check_single_rank_no_comm(solve):
    assert solve(1e9, 1, 1e11) == (0.0, 0.0)


def check_matches_ref(solve):
    cases = [(1e9, 8, 1e11, 0.0), (2e9, 16, 5e10, 1e-5), (1e6, 2, 1e9, 0.0),
             (7e8, 4, 2e11, 3e-6), (1e9, 64, 1e11, 0.0)]
    for c in cases:
        got = solve(*c)
        want = _ref(*c)
        assert _close(got[0], want[0]) and _close(got[1], want[1]), (c, got, want)


def check_volume_bounded_by_2S(solve):
    # 每卡传输量恒 <= 2S，且随 world 增大趋近 2S（不随卡数线性膨胀）
    S = 1e9
    v2 = solve(S, 2, 1e11)[0]
    v64 = solve(S, 64, 1e11)[0]
    assert v2 <= 2 * S + 1e-6 and v64 <= 2 * S + 1e-6
    assert v64 > v2, (v2, v64)          # 2S*(1/2)=1S vs 2S*(63/64)≈1.97S
    assert _close(v2, S), v2
    assert _close(v64, 2 * S * 63 / 64), v64


def check_latency_adds_up(solve):
    bw_vol = solve(1e9, 8, 1e11, 0.0)
    with_lat = solve(1e9, 8, 1e11, 1e-5)
    # 只增加延迟项：2*(8-1)*1e-5 = 1.4e-4
    assert _close(with_lat[0], bw_vol[0]), (with_lat, bw_vol)
    assert _close(with_lat[1] - bw_vol[1], 2 * 7 * 1e-5), (with_lat, bw_vol)


check_example.example = True
