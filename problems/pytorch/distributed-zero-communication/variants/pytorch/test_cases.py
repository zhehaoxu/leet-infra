def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算参考：分步累加通信件，不复用 solve 的闭合式
def _ref(n_params, world, grad_bytes, param_bytes):
    W = float(world)
    c = (W - 1.0) / W
    G = float(n_params) * grad_bytes
    P = float(n_params) * param_bytes
    ddp = c * G + c * G                       # all-reduce = rs + ag
    z12 = c * G + c * G                        # 梯度 rs + 参数 ag
    z3 = c * G + c * G + c * P + c * P         # 再叠前向/反向各一次参数 ag
    return ddp, z12, z3


# 样例用例
def check_example(solve):
    ddp, z12, z3 = solve(1e9, 8)
    assert _close(ddp, 3.5e9), ddp
    assert _close(z12, 3.5e9), z12
    assert _close(z3, 7.0e9), z3


def check_zero_equals_ddp(solve):
    # ZeRO-1/2 通信量与 DDP 相同（省显存不省通信）
    for N, W in [(1e9, 8), (7e8, 4), (1e6, 64)]:
        ddp, z12, _ = solve(N, W)
        assert _close(ddp, z12), (N, W, ddp, z12)


def check_zero3_heavier(solve):
    # ZeRO-3 通信严格多于 DDP（当 param_bytes>0, W>1）
    for N, W in [(1e9, 8), (5e8, 16)]:
        ddp, _, z3 = solve(N, W)
        assert z3 > ddp, (ddp, z3)
        assert _close(z3, ddp + 2 * ((W - 1) / W) * N * 2), z3


def check_world_one_no_comm(solve):
    ddp, z12, z3 = solve(1e9, 1)
    assert ddp == 0.0 and z12 == 0.0 and z3 == 0.0, (ddp, z12, z3)


def check_dtype_bytes(solve):
    # fp32 梯度/参数（4 bytes）应翻倍
    d2 = solve(1e8, 8, grad_bytes=2, param_bytes=2)
    d4 = solve(1e8, 8, grad_bytes=4, param_bytes=4)
    for a, b in zip(d4, d2):
        assert _close(a, 2.0 * b), (a, b)


def check_matches_ref(solve):
    for N, W, gb, pb in [(1e9, 8, 2, 2), (3e8, 16, 4, 2), (1e7, 2, 2, 4), (1e10, 128, 2, 2)]:
        got = solve(N, W, gb, pb)
        ref = _ref(N, W, gb, pb)
        assert len(got) == 3
        for g, r in zip(got, ref):
            assert _close(g, r), (N, W, gb, pb, got, ref)


check_example.example = True
