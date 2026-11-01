def _py(chunks, n_batch, logits_last=False, n_past=0):
    """oracle：事件账独立重算——先把每个 chunk 的批切分边界全部枚举，再统一回填 pos。

    与 solve 的「单指针顺序推进」路径不同：oracle 按 chunk 先产出
    (kind, 批边界) 事件表，pos 账最后从事件表独立累计。
    """
    if not isinstance(chunks, (list, tuple)):
        raise ValueError
    if not isinstance(n_batch, int) or isinstance(n_batch, bool) or n_batch <= 0:
        raise ValueError
    for ch in chunks:
        if not isinstance(ch, dict) or ch.get("type") not in ("text", "image"):
            raise ValueError
        nt = ch.get("n_tokens")
        if not isinstance(nt, int) or isinstance(nt, bool) or nt < 0:
            raise ValueError

    events = []  # (chunk_index, kind, [(size, offset_in_chunk), ...])
    for ci, ch in enumerate(chunks):
        n_tokens = ch["n_tokens"]
        cuts = []
        if ch["type"] == "text":
            start = 0
            while start < n_tokens:
                size = n_batch
                if n_tokens - start < size:
                    size = n_tokens - start
                cuts.append((size, start))
                start = start + size
        else:
            if n_tokens == 0:
                cuts = []
            else:
                full = n_tokens // n_batch
                rem = n_tokens - full * n_batch
                for b in range(full):
                    cuts.append((n_batch, b * n_batch))
                if rem > 0:
                    cuts.append((rem, full * n_batch))
        events.append((ci, ch["type"], cuts))

    batches = []
    logits_pos = None
    pos = n_past
    n_chunks = len(chunks)
    for ci, kind, cuts in events:
        if kind == "text":
            for bi, (size, off) in enumerate(cuts):
                is_last_batch = bi == len(cuts) - 1
                take = (
                    logits_last
                    and ci == n_chunks - 1
                    and is_last_batch
                )
                if take:
                    logits_pos = pos + off + size - 1
                batches.append(
                    {
                        "chunk_index": ci,
                        "kind": "text",
                        "n_tokens": size,
                        "pos_start": pos + off,
                        "logits": take,
                    }
                )
            pos = pos + sum(size for size, _ in cuts)
        else:
            for size, off in cuts:
                batches.append(
                    {
                        "chunk_index": ci,
                        "kind": "image",
                        "n_tokens": size,
                        "pos_start": pos + off,
                        "logits": False,
                    }
                )
            total = sum(size for size, _ in cuts)
            pos = pos + total
    return {"batches": batches, "new_n_past": pos, "logits_pos": logits_pos}


def _compare(solve, chunks, n_batch, logits_last=False, n_past=0):
    led = solve(chunks, n_batch, logits_last, n_past)
    ref = _py(chunks, n_batch, logits_last, n_past)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_mixed(solve):
    # [text 5, image 576, text 3]、n_batch=256、logits_last=True
    led = _compare(
        solve,
        [
            {"type": "text", "n_tokens": 5},
            {"type": "image", "n_tokens": 576},
            {"type": "text", "n_tokens": 3},
        ],
        256,
        True,
    )
    bs = led["batches"]
    assert len(bs) == 5
    # text 5：1 批 pos 0..4
    assert bs[0] == {"chunk_index": 0, "kind": "text", "n_tokens": 5, "pos_start": 0, "logits": False}
    # image 576：3 批 256/256/64，pos_start 5/261/517
    assert bs[1]["n_tokens"] == 256 and bs[1]["pos_start"] == 5
    assert bs[2]["n_tokens"] == 256 and bs[2]["pos_start"] == 261
    assert bs[3]["n_tokens"] == 64 and bs[3]["pos_start"] == 517
    assert all(b["logits"] is False for b in bs[1:4])
    # text 3：pos 581..583，logits 在最后一个 token
    assert bs[4] == {"chunk_index": 2, "kind": "text", "n_tokens": 3, "pos_start": 581, "logits": True}
    assert led["new_n_past"] == 584
    assert led["logits_pos"] == 583


def check_image_no_logits_quirk(solve):
    # 官方 quirk：image chunk 分支不处理 logits_last → 无 logits
    led = _compare(solve, [{"type": "image", "n_tokens": 100}], 64, True)
    assert led["logits_pos"] is None
    assert all(b["logits"] is False for b in led["batches"])
    assert led["new_n_past"] == 100
    # 前置 text chunk 也救不回来（image 是最后 chunk）
    led2 = _compare(
        solve,
        [{"type": "text", "n_tokens": 2}, {"type": "image", "n_tokens": 10}],
        4,
        True,
    )
    assert led2["logits_pos"] is None
    # logits 标记条件是「最后一个 chunk」：中间 text chunk 不设
    assert led2["batches"][0]["logits"] is False


def check_exact_division_and_remainder(solve):
    # 恰整除：末批满批；余 1：末批单 token
    led = _compare(solve, [{"type": "image", "n_tokens": 512}], 256)
    assert [b["n_tokens"] for b in led["batches"]] == [256, 256]
    assert [b["pos_start"] for b in led["batches"]] == [0, 256]
    led2 = _compare(solve, [{"type": "image", "n_tokens": 513}], 256)
    assert [b["n_tokens"] for b in led2["batches"]] == [256, 256, 1]
    assert led2["batches"][2]["pos_start"] == 512
    assert led2["new_n_past"] == 513


def check_text_multi_batch(solve):
    # text 逐 token 推进：n_batch=3、7 tokens → 批 3/3/1
    led = _compare(solve, [{"type": "text", "n_tokens": 7}], 3, True)
    bs = led["batches"]
    assert [(b["n_tokens"], b["pos_start"]) for b in bs] == [(3, 0), (3, 3), (1, 6)]
    assert bs[-1]["logits"] is True and led["logits_pos"] == 6
    # 中间批 logits=False
    assert bs[0]["logits"] is False and bs[1]["logits"] is False


def check_n_past_offset(solve):
    # 起始位置非 0：全部 pos 平移
    led = _compare(
        solve,
        [{"type": "text", "n_tokens": 2}, {"type": "image", "n_tokens": 5}],
        4,
        False,
        100,
    )
    assert led["batches"][0]["pos_start"] == 100
    assert led["batches"][1]["pos_start"] == 102
    assert led["batches"][1]["pos_start"] == 102 and led["batches"][1]["n_tokens"] == 4
    assert led["batches"][2]["pos_start"] == 106
    assert led["new_n_past"] == 107


def check_empty_cases(solve):
    # 空序列：无批、位置不动、无 logits（官方 "no chunks to eval"）
    led = _compare(solve, [], 256, True, 42)
    assert led == {"batches": [], "new_n_past": 42, "logits_pos": None}
    # 空文本 chunk：无批；空 image chunk：无批且不推进
    led2 = _compare(solve, [{"type": "text", "n_tokens": 0}], 8, True)
    assert led2 == {"batches": [], "new_n_past": 0, "logits_pos": None}
    led3 = _compare(solve, [{"type": "image", "n_tokens": 0}], 8)
    assert led3 == {"batches": [], "new_n_past": 0, "logits_pos": None}
    # logits_last 但 text chunk 为空且是最后 chunk：无 token 可标
    led4 = _compare(
        solve,
        [{"type": "text", "n_tokens": 4}, {"type": "text", "n_tokens": 0}],
        2,
        True,
    )
    assert led4["logits_pos"] is None


def check_single_token_batch(solve):
    led = _compare(solve, [{"type": "text", "n_tokens": 3}], 1, True)
    bs = led["batches"]
    assert [(b["n_tokens"], b["pos_start"], b["logits"]) for b in bs] == [
        (1, 0, False), (1, 1, False), (1, 2, True),
    ]


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    ve("nope", 8)                                  # chunks 非列表
    ve([{"type": "audio", "n_tokens": 3}], 8)      # type 不在题面域
    ve([{"type": "text", "n_tokens": -1}], 8)      # 负 token 数
    ve([{"type": "text", "n_tokens": 1.5}], 8)     # 非 int
    ve([{"type": "text", "n_tokens": 3}], 0)       # n_batch = 0（官方 assert）
    ve([{"type": "text", "n_tokens": 3}], -4)
    ve([{"type": "text", "n_tokens": 3}], True)    # bool 冒充 int
    ve([{"type": "text", "n_tokens": 3}], 8, "y")  # logits_last 非 bool
    ve([{"type": "text", "n_tokens": 3}], 8, False, -1)
    ve([{"n_tokens": 3}], 8)                       # 缺 type
    ve([{"type": "text"}], 8)                      # 缺 n_tokens


def check_oracle_random(solve):
    import random

    rng = random.Random(20260690)
    for _ in range(300):
        nc = rng.randint(0, 6)
        chunks = []
        for _i in range(nc):
            chunks.append(
                {
                    "type": rng.choice(["text", "image"]),
                    "n_tokens": rng.randint(0, 40),
                }
            )
        n_batch = rng.randint(1, 16)
        ll = rng.random() < 0.6
        np0 = rng.randint(0, 50)
        led = solve(chunks, n_batch, ll, np0)
        ref = _py(chunks, n_batch, ll, np0)
        assert led == ref, "random mismatch"
