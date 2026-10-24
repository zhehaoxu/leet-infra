def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立参考：把 4 次 all-reduce（2 前向 + 2 反向）逐次累加
def _ref(batch, seq, hidden, world, n_layers, bpe):
    act = float(batch * seq * hidden * bpe)
    c = (world - 1) / world
    per_layer = 0.0
    for _ in range(4):                # 每层 4 次 all-reduce
        per_layer += 2.0 * c * act
    return act, per_layer, float(n_layers) * per_layer


# 样例用例
def check_example(solve):
    act, pl, tot = solve(8, 2048, 4096, 8, 32, 2)
    assert _close(act, 8 * 2048 * 4096 * 2), act
    assert _close(pl, 8 * (7 / 8) * act), pl
    assert _close(tot, 32 * pl), tot


def check_world_one_no_comm(solve):
    act, pl, tot = solve(4, 512, 1024, 1, 24, 2)
    assert pl == 0.0 and tot == 0.0, (pl, tot)
    assert _close(act, 4 * 512 * 1024 * 2), act   # 激活本身仍算得出


def check_total_is_n_layers_times_per_layer(solve):
    for n_layers in (1, 2, 12, 32):
        act, pl, tot = solve(2, 128, 256, 8, n_layers, 2)
        assert _close(tot, n_layers * pl), (n_layers, tot, pl)


def check_act_bytes_independent_of_world(solve):
    # act_bytes 只与 batch/seq/hidden/dtype 有关，与 world 无关
    a1 = solve(2, 128, 256, 4, 1, 2)[0]
    a2 = solve(2, 128, 256, 8, 1, 2)[0]
    assert _close(a1, a2), (a1, a2)


def check_dtype_bytes_linear(solve):
    _, pl2, _ = solve(2, 128, 256, 8, 1, 2)
    _, pl4, _ = solve(2, 128, 256, 8, 1, 4)
    assert _close(pl4, 2.0 * pl2), (pl2, pl4)


def check_per_layer_is_8c_act(solve):
    # 每层 = 8·c·act（4 次 all-reduce 各 2c·act）
    for W in (2, 4, 8, 16):
        act, pl, _ = solve(1, 64, 64, W, 1, 2)
        c = (W - 1) / W
        assert _close(pl, 8.0 * c * act), (W, pl, act)


def check_matches_ref(solve):
    for args in [(8, 2048, 4096, 8, 32, 2), (2, 256, 512, 4, 6, 4),
                 (1, 1, 1, 2, 1, 2), (16, 1024, 8192, 8, 48, 2)]:
        got = solve(*args)
        ref = _ref(*args)
        assert len(got) == 3
        for x, y in zip(got, ref):
            assert _close(x, y), (args, got, ref)


check_example.example = True
