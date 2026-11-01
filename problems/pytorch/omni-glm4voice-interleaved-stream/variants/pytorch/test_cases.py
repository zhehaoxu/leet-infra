from collections.abc import Callable

BL = [25, 50, 100, 150, 200]


def _oracle(token_ids, audio_offset, end_token_id, block_size_list):
    text_tokens, audio_tokens, complete = [], [], []
    idx = 0
    block = block_size_list[0]
    fin = False
    chunks = []
    for tid in token_ids:
        fin = fin or tid == end_token_id
        if len(audio_tokens) >= block or (fin and audio_tokens):
            chunks.append([list(audio_tokens), block, fin])
            if idx < len(block_size_list) - 1:
                idx += 1
                block = block_size_list[idx]
            audio_tokens = []
        if not fin:
            complete.append(tid)
            if tid >= audio_offset:
                audio_tokens.append(tid - audio_offset)
            else:
                text_tokens.append(tid)
    return {
        "complete_tokens": complete,
        "text_tokens": text_tokens,
        "chunks": chunks,
        "dropped_tail": audio_tokens,
        "final_block_size": block,
        "is_finalize": fin,
    }


def _compare(solve: Callable, *args):
    led = solve(*args)
    exp = _oracle(*args)
    assert led == exp, f"\nled={led}\nexp={exp}"
    return led


def check_text_only_no_end(solve):
    led = _compare(solve, [1, 2, 3], 200, 999, BL)
    assert led["complete_tokens"] == [1, 2, 3]
    assert led["text_tokens"] == [1, 2, 3]
    assert led["chunks"] == [] and led["dropped_tail"] == []
    assert led["final_block_size"] == 25 and not led["is_finalize"]


def check_audio_end_flushes_finalize_block(solve):
    # end 且 audio 非空 → 立刻发 finalize 块；end 不分类
    led = _compare(solve, [100, 200, 201, 202, 300], 200, 300, BL)
    assert led["complete_tokens"] == [100, 200, 201, 202]
    assert led["text_tokens"] == [100]
    assert led["chunks"] == [[[0, 1, 2], 25, True]]
    assert led["dropped_tail"] == [] and led["is_finalize"]


def check_end_empty_audio_no_flush(solve):
    # quirk：end 且 audio 空 → 不 flush；end 后 token 全部停止分类
    led = _compare(solve, [100, 300, 101, 200], 200, 300, BL)
    assert led["complete_tokens"] == [100]
    assert led["text_tokens"] == [100]
    assert led["chunks"] == []
    assert led["is_finalize"]


def check_post_end_tokens_ignored(solve):
    # end 后即使有音频 token 也不进 complete/分类
    led = _compare(solve, [100, 200, 300, 201, 202], 200, 300, BL)
    assert led["complete_tokens"] == [100, 200]
    assert led["chunks"] == [[[0], 25, True]]
    assert led["dropped_tail"] == []


def check_first_block_flush_at_25(solve):
    # flush 判定先于分类：第 26 个 audio 到来前 len=25>=25 触发
    ids = list(range(200, 230))
    led = _compare(solve, ids, 200, 999, BL)
    assert led["chunks"] == [[list(range(25)), 25, False]]
    assert led["dropped_tail"] == [25, 26, 27, 28, 29]
    assert led["final_block_size"] == 50  # flush 后升级
    assert not led["is_finalize"]


def check_progressive_block_upgrade(solve):
    # 30 + 55 audio：块 25 → 升 50 → 块 50 → 尾 10 丢（无 end）
    ids = list(range(200, 230)) + list(range(200, 255))
    led = _compare(solve, ids, 200, 999, BL)
    assert [c[1] for c in led["chunks"]] == [25, 50]
    assert [c[2] for c in led["chunks"]] == [False, False]
    assert led["dropped_tail"] == list(range(45, 55))
    assert led["final_block_size"] == 100


def check_finalize_after_upgrade_caps_at_last(solve):
    # 阶梯到顶不再升级：多块 flush 后 block 恒 200
    ids = list(range(200, 200 + 530))
    led = _compare(solve, ids, 200, 999, [25, 50, 100, 150, 200])
    assert [c[1] for c in led["chunks"]] == [25, 50, 100, 150, 200]
    assert led["chunks"][-1][0] == list(range(325, 525))
    assert led["dropped_tail"] == list(range(525, 530))
    assert led["final_block_size"] == 200  # 到顶不再升级


def check_audio_offset_math(solve):
    # audio 值 = id - offset，无上界检查（offset+0 之外的大 id 也是音频）
    led = _compare(solve, [205, 299, 1000], 200, 999, BL)
    assert led["dropped_tail"] == [5, 99, 800]
    assert led["text_tokens"] == []


def check_empty_stream(solve):
    led = _compare(solve, [], 200, 999, BL)
    assert led["complete_tokens"] == [] and led["chunks"] == []
    assert led["final_block_size"] == 25 and not led["is_finalize"]


def check_interleaved_text_audio(solve):
    # 交错：audio 块 flush 不打断 text 收集
    ids = [10, 11] + list(range(200, 226)) + [12]  # 26 audio → flush 25，尾 1
    led = _compare(solve, ids, 200, 999, BL)
    assert led["text_tokens"] == [10, 11, 12]
    assert len(led["chunks"]) == 1
    assert led["chunks"][0][0] == list(range(25))
    assert led["dropped_tail"] == [25]


def check_invalid_inputs(solve):
    try:
        solve([1], 200, 999, [])
        raise SystemError("expected assertion")
    except (AssertionError, IndexError):
        pass


def check_random_cross_check(solve):
    import random
    rng = random.Random(698698)
    for _ in range(400):
        n = rng.randint(0, 120)
        offset = 200
        end = 999 if rng.random() < 0.5 else rng.randint(0, 199)
        ids = []
        for _ in range(n):
            if rng.random() < 0.6:
                ids.append(offset + rng.randint(0, 60))
            else:
                ids.append(rng.randint(0, 199))
        bl = [25, 50, 100, 150, 200]
        if rng.random() < 0.5:
            bl = bl[: rng.randint(1, 5)]
        _compare(solve, ids, offset, end, bl)
