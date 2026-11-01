from collections.abc import Callable


def _oracle(prompt_hiddens, prompt_embeds, audio_mask,
            thinker_hiddens, thinker_embeds,
            text_bos_embed, eos_embed, pad_embed,
            codec_embeds, codec_tokens, n_decode):
    prefill = [h + (0.0 if a else e)
               for h, e, a in zip(prompt_hiddens, prompt_embeds, audio_mask)]
    reply = [h + e for h, e in zip(thinker_hiddens, thinker_embeds)]
    prefill += [text_bos_embed, reply[0]]
    queue = reply[1:] + [eos_embed, pad_embed]
    init = list(queue)
    steps = []
    for t in range(n_decode):
        steps.append([codec_embeds[codec_tokens[t]] + queue.pop(0), t])
    return {
        "prefill_embeds": prefill,
        "prefill_input_ids": [-1] * len(prompt_hiddens) + [-2, -3],
        "initial_reply_queue": init,
        "steps": steps,
        "final_reply_len": len(queue),
    }


def _compare(solve: Callable, *args):
    led = solve(*args)
    exp = _oracle(*args)
    assert led == exp, f"\nled={led}\nexp={exp}"
    return led


def check_prefill_add_not_concat(solve):
    # 隐态 + embedding 逐元素相加；audio 位置 embedding 置零但隐态保留
    led = _compare(solve, [1.0, 2.0], [10.0, 20.0], [False, True],
                   [100.0, 200.0], [1.0, 2.0], 0.5, 0.1, 0.2,
                   {5: 0.05, 6: 0.06}, [5, 6, 5], 3)
    assert led["prefill_embeds"] == [11.0, 2.0, 0.5, 101.0]
    assert led["prefill_input_ids"] == [-1, -1, -2, -3]


def check_prefill_tail_two_frames(solve):
    # 末尾追加 [text_bos, reply[0]]，reply[0] = hiddens[0]+embeds[0]
    led = _compare(solve, [0.0], [0.0], [False],
                   [7.0, 8.0], [0.3, 0.4], 0.5, 0.1, 0.2,
                   {9: 1.0}, [9, 9, 9], 1)
    assert led["prefill_embeds"] == [0.0, 0.5, 7.3]
    assert led["initial_reply_queue"] == [8.4, 0.1, 0.2]


def check_reply_queue_bos_pad_bailout(solve):
    # 队列 = reply[1:] + [eos, pad]；eos/pad 是兜底帧
    led = _compare(solve, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [False, False, False],
                   [100.0], [5.0], 0.5, 0.1, 0.2,
                   {1: 0.01}, [1, 1], 2)
    assert led["initial_reply_queue"] == [0.1, 0.2]
    # step0: codec 0.01 + eos 0.1；step1: codec + pad 0.2（队列恰耗尽）
    import math
    assert len(led["steps"]) == 2
    assert math.isclose(led["steps"][0][0], 0.11) and led["steps"][0][1] == 0
    assert math.isclose(led["steps"][1][0], 0.21) and led["steps"][1][1] == 1
    assert led["final_reply_len"] == 0


def check_decode_one_to_one_consumption(solve):
    # 1:1：每步恰消费 1 帧，队列砍头
    led = _compare(solve, [0.0], [0.0], [False],
                   [10.0, 20.0, 30.0], [1.0, 1.0, 1.0], 0.5, 0.1, 0.2,
                   {3: 0.3}, [3, 3, 3, 3], 4)
    # reply = [11,21,31]; queue = [21,31,0.1,0.2]
    assert led["initial_reply_queue"] == [21.0, 31.0, 0.1, 0.2]
    assert led["steps"] == [[21.3, 0], [31.3, 1], [0.4, 2], [0.5, 3]]
    assert led["final_reply_len"] == 0


def check_final_reply_len(solve):
    led = _compare(solve, [0.0], [0.0], [False],
                   [10.0, 20.0], [1.0, 1.0], 0.5, 0.1, 0.2,
                   {3: 0.3}, [3, 3, 3], 2)
    assert led["final_reply_len"] == 1  # T+1 - n_decode = 3-2


def check_audio_mask_zero_not_drop(solve):
    # audio 位置：embedding 置零参与相加（不是整帧丢掉，隐态仍在）
    led = _compare(solve, [5.0, 6.0, 7.0], [1.0, 1.0, 1.0], [False, True, True],
                   [9.0], [9.0], 0.0, 0.0, 0.0,
                   {1: 0.0}, [1], 1)
    assert led["prefill_embeds"][:3] == [6.0, 6.0, 7.0]


def check_t_equal_one_edge(solve):
    # T=1：队列只有 [eos, pad]
    led = _compare(solve, [2.0], [3.0], [False],
                   [4.0], [5.0], 0.5, 0.1, 0.2,
                   {2: 1.0}, [2, 2], 2)
    assert led["prefill_embeds"] == [5.0, 0.5, 9.0]
    assert led["initial_reply_queue"] == [0.1, 0.2]
    assert led["steps"] == [[1.1, 0], [1.2, 1]]
    assert led["final_reply_len"] == 0


def check_invalid_inputs(solve):
    # n_decode > T+1 超兜底 → 断言（官方空队列会崩，出题约束排除）
    try:
        solve([0.0], [0.0], [False], [1.0], [1.0], 0.5, 0.1, 0.2,
              {1: 0.0}, [1, 1, 1], 3)
        raise SystemError("expected assertion")
    except AssertionError:
        pass
    # 长度不等 → 断言
    try:
        solve([0.0, 1.0], [0.0], [False, False], [1.0], [1.0], 0.5, 0.1, 0.2,
              {1: 0.0}, [1], 1)
        raise SystemError("expected assertion")
    except AssertionError:
        pass


def check_random_cross_check(solve):
    import random
    rng = random.Random(697697)
    for _ in range(300):
        L = rng.randint(1, 6)
        T = rng.randint(1, 6)
        n = rng.randint(0, T + 1)
        ph = [rng.uniform(-5, 5) for _ in range(L)]
        pe = [rng.uniform(-5, 5) for _ in range(L)]
        am = [rng.random() < 0.3 for _ in range(L)]
        th = [rng.uniform(-5, 5) for _ in range(T)]
        te = [rng.uniform(-5, 5) for _ in range(T)]
        tb, eo, pd = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
        codes = {i: rng.uniform(-1, 1) for i in range(3)}
        ct = [rng.randrange(3) for _ in range(max(n, 1))]
        _compare(solve, ph, pe, am, th, te, tb, eo, pd, codes, ct, n)
