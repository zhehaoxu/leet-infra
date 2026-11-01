from collections.abc import Callable

UNGEN = -1
APAD = 2048


def _oracle(max_steps, input_frames, dep_outputs):
    gen, inp, vocab, delay = 8, 8, 2049, 2
    apad = vocab - 1
    mat = [[UNGEN] * (gen + inp) for _ in range(max_steps + delay)]
    playable = []
    for step in range(max_steps):
        fin = input_frames[step] if step < len(input_frames) else []
        for c, t in enumerate(fin):
            mat[step][gen + c] = t
        out = dep_outputs[step] if step < len(dep_outputs) else None
        for c in range(gen):
            d = 0 if c == 0 else delay
            mat[max(step - d, 0)][c] = apad if out is None else out[c]
        if step <= delay:
            playable.append(None)
        else:
            f = mat[step - delay - 1]
            playable.append(None if any(v >= vocab - 1 for v in f) else list(f))
    return {
        "audio_tokens": mat,
        "playable": playable,
        "ungenerated_left": sum(r.count(UNGEN) for r in mat),
    }


def _compare(solve: Callable, *args):
    led = solve(*args)
    exp = _oracle(*args)
    assert led == exp, f"\nled={led}\nexp={exp}"
    return led


IN = [[100 + i for i in range(8)]] * 6


def _outs(n):
    return [[10 + s * 8 + c for c in range(8)] for s in range(n)]


def check_basic_overwrite_chain(solve):
    led = _compare(solve, 6, IN, _outs(6))
    assert led["audio_tokens"][0][:8] == [10] + list(range(27, 34))
    assert led["audio_tokens"][1][:8] == [18] + list(range(35, 42))
    assert led["audio_tokens"][2][:8] == [26] + list(range(43, 50))
    assert led["audio_tokens"][3][:8] == [34] + list(range(51, 58))
    # 输入半每步直写
    for s in range(4):
        assert led["audio_tokens"][s][8:] == [100 + i for i in range(8)]


def check_playable_frames(solve):
    led = _compare(solve, 6, IN, _outs(6))
    assert led["playable"][0] is None
    assert led["playable"][1] is None
    assert led["playable"][2] is None  # step <= delay=2
    assert led["playable"][3] == [10] + list(range(27, 34)) + [100 + i for i in range(8)]
    assert led["playable"][4][0] == 18
    assert led["playable"][5][0] == 26


def check_none_output_writes_pad(solve):
    # depformer 未生成 → 写 audio_pad=2048
    led = _compare(solve, 3, IN, [None, None, None])
    assert led["audio_tokens"][0][:8] == [APAD] * 8
    # slot1/2 只有 cb0 被 step1/2 写（cb1-7 delay 2 未及）→ 残留 UNGENERATED
    assert led["audio_tokens"][1][:8] == [APAD] + [UNGEN] * 7
    assert led["audio_tokens"][2][:8] == [APAD] + [UNGEN] * 7
    assert all(p is None for p in led["playable"])  # pad 帧不播


def check_pad_frame_not_playable(solve):
    # 中途一步输出 None（pad 混入）→ 对应播放帧整帧 None
    outs = _outs(6)
    outs[3] = None
    led = _compare(solve, 6, IN, outs)
    # step3 None 写 slot3(cb0) 与 slot1(cb1-7)；playable[4] 读 slot1 → 含 pad → None
    assert led["playable"][4] is None
    assert led["playable"][3] is not None
    # playable[5] 读 slot2：cb0=step2、cb1-7=step4 → 无 pad 可播
    assert led["playable"][5] is not None


def check_audio_pad_value_also_from_dep(solve):
    # dep 输出值恰为 2048 也触发整帧不播（>= vocab-1 判定）
    outs = _outs(6)
    outs[2][5] = APAD
    led = _compare(solve, 6, IN, outs)
    # outs[2] 写 slot2(cb0)/slot0(cb1-7)；slot0 在 playable[3] → None
    assert led["playable"][3] is None
    assert led["playable"][4] is not None


def check_short_input_frames(solve):
    # 输入流比输出短：后面步不写输入半（保留 UNGENERATED）
    led = _compare(solve, 4, IN[:2], _outs(4))
    for s in (2, 3):
        assert led["audio_tokens"][s][8:] == [UNGEN] * 8
    # playable[3] 读 slot0（输入半已写）→ 尾半是 100..107
    assert led["playable"][3][8:] == [100 + i for i in range(8)]


def check_ungenerated_count(solve):
    led = _compare(solve, 4, IN[:2], _outs(4))
    # 矩阵 6 行：行 0-3 生成半全写；行 4-5 生成半未写（8*2）；输入半行 2-3（16）
    assert led["ungenerated_left"] == 62  # 96 格 - 已写 34


def check_matrix_tail_rows(solve):
    # 矩阵 = max_steps + delay 行，尾行全 UNGENERATED（除非输入写满）
    led = _compare(solve, 3, IN, _outs(3))
    assert len(led["audio_tokens"]) == 5
    assert led["audio_tokens"][4] == [UNGEN] * 16


def check_single_step(solve):
    led = _compare(solve, 1, [], [None])
    assert led["audio_tokens"][0][:8] == [APAD] * 8
    assert led["playable"] == [None]
    assert led["ungenerated_left"] == 40  # 3 行 48 格 - 生成半 8


def check_invalid_inputs(solve):
    try:
        solve(0, [], [])
        raise SystemError("expected assertion")
    except AssertionError:
        pass
    assert True  # 其余合法性由入参保证（官方无额外防御）


def check_random_cross_check(solve):
    import random
    rng = random.Random(699699)
    for _ in range(300):
        n = rng.randint(1, 10)
        n_in = rng.randint(0, n)
        frames = [[rng.randint(0, 2000) for _ in range(8)] for _ in range(n_in)]
        outs = []
        for _ in range(rng.randint(0, n)):
            if rng.random() < 0.2:
                outs.append(None)
            else:
                outs.append([rng.randint(0, 2048) for _ in range(8)])
        _compare(solve, n, frames, outs)
