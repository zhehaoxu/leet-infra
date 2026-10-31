"""TMRoPE：文本+音频段 3D 位置 id 账（test cases）。"""

import random


def _py(input_ids, audio_seqlens):
    """独立 oracle：直接对官方逻辑的逐步重放（不写循环复用 solution 结构）。"""
    TEXT, AS, AP, AE = 0, 1, 2, 3
    assert all(t in (TEXT, AS, AP, AE) for t in input_ids)

    out = []          # 全部位置值，按输入序（展开后）
    kinds = []        # 每个位置的来源标注，用于断言
    ai = 0
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i] == AS:
            assert input_ids[i + 1] == AP and input_ids[i + 2] == AE
            L = audio_seqlens[ai]
            a_len = ((L - 1) // 2 + 1 - 2) // 2 + 1
            # bos / body / eos 的位置值 = 当前 out 的 max + 1 连续排
            base = (max(out) + 1) if out else 0
            out.append(base); kinds.append("bos")
            for _ in range(a_len):
                base = (max(out) + 1) if out else 0
                out.append(base); kinds.append("audio")
            base = (max(out) + 1) if out else 0
            out.append(base); kinds.append("eos")
            ai += 1
            i += 3
        else:
            base = (max(out) + 1) if out else 0
            out.append(base); kinds.append("text")
            i += 1
    delta = (max(out) + 1 - len(out)) if out else 0  # 官方按展开后长度
    return {"pos": out, "delta": delta, "kinds": kinds,
            "audio_len": [((L - 1) // 2 + 1 - 2) // 2 + 1 for L in audio_seqlens]}


def _close(actual, expect):
    if isinstance(actual, dict):
        for k in expect:
            assert _close(actual[k], expect[k]), f"mismatch at {k}"
        return True
    if isinstance(actual, list):
        assert isinstance(expect, list) and len(actual) == len(expect), \
            f"len {len(actual)} != {len(expect)}"
        return all(_close(a, e) for a, e in zip(actual, expect))
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, ids, seqlens):
    led = solve(list(ids), list(seqlens))
    ref = _py(list(ids), list(seqlens))
    assert len(led["position_ids"]) == 3
    # 三行相同
    assert led["position_ids"][0] == led["position_ids"][1] \
        == led["position_ids"][2]
    _close(led["position_ids"][0], ref["pos"])
    assert led["mrope_delta"] == ref["delta"]
    return led, ref


def check_pure_text(solve):
    # 纯文本 = 标准 1D 位置，delta = 0
    led, _ = _compare(solve, [0, 0, 0, 0, 0], [])
    assert led["position_ids"][0] == [0, 1, 2, 3, 4]
    assert led["mrope_delta"] == 0


def check_single_audio_anchor(solve):
    # T T [A] T T T；L=200 → audio_len = ((199)//2+1-2)//2+1 = 50
    ids = [0, 0, 1, 2, 3, 0, 0, 0]
    led, ref = _compare(solve, ids, [200])
    assert ref["audio_len"] == [50]
    # 手算：text 0,1；bos 2；audio 3..52；eos 53；tail 54,55,56
    assert led["position_ids"][0] == [0, 1] + [2] + list(range(3, 53)) + [53] \
        + [54, 55, 56]
    # 展开后长度 57 == max+1 → 音频不改变 delta（非零 delta 只来自视频）
    assert led["mrope_delta"] == 0


def check_leading_audio(solve):
    # 序列以音频开头：st_idx 从 0 起
    ids = [1, 2, 3, 0, 0]
    led, ref = _compare(solve, ids, [100])
    # L=100 → (49+1-2)//2+1 = 25? (99//2=49; 49+1=50; 50-2=48; 48//2=24; +1=25)
    assert ref["audio_len"] == [25]
    assert led["position_ids"][0] == [0] + list(range(1, 26)) + [26, 27, 28]
    assert led["mrope_delta"] == 0


def check_two_audio_blocks(solve):
    ids = [0, 1, 2, 3, 0, 0, 1, 2, 3, 0]
    led, ref = _compare(solve, ids, [40, 8])
    # 块1：L=40 → (19+1-2)//2+1 = 10；块2：L=8 → (3+1-2)//2+1 = 2
    assert ref["audio_len"] == [10, 2]
    pos = led["position_ids"][0]
    # 展开后长度：10 输入 - 2 块*3 + (1+10+1) + (1+2+1) = 20
    assert len(pos) == 20
    # 手算：T(0) bos1 aud2..11 eos12 T,T(13,14) bos15 aud16,17 eos18 T(19)
    assert pos == [0, 1] + list(range(2, 13)) + [13, 14, 15, 16, 17, 18, 19]
    assert led["mrope_delta"] == 0


def check_audio_len_formula(solve):
    # 公式边界：L=1 → 0；L=2 → 0；L=3 → 1；L=200 → 50
    # 经 solve 间接测：单块序列展开长度 = 2 + audio_len
    for L, want in [(1, 0), (2, 0), (3, 1), (100, 25), (200, 50)]:
        led = solve([1, 2, 3], [L])
        got = len(led["position_ids"][0]) - 2
        assert got == want, f"L={L}: {got} != {want}"


def check_oracle_random(solve):
    rng = random.Random(674)
    for _ in range(150):
        n_blocks = rng.randint(0, 3)
        seqlens = [rng.choice([1, 2, 3, 8, 25, 40, 100, 200])
                   for _ in range(n_blocks)]
        ids = []
        for j in range(n_blocks + 1):
            ids += [0] * rng.randint(0, 4)
            if j < n_blocks:
                ids += [1, 2, 3]
        led, ref = _compare(solve, ids, seqlens)
        # 展开后长度 = 输入 - 3*块数 + sum(2+audio_len)
        exp_len = len(ids) - 3 * len(seqlens) + sum(2 + a for a in ref["audio_len"])
        assert len(led["position_ids"][0]) == exp_len


def check_invalid_inputs(solve):
    for bad_ids, bad_seq in [
        ([1, 2, 3], []),           # 块数与 seqlens 不符
        ([1, 2], [10]),            # 块结构残缺
        ([2, 2], []),              # 游离 pad
        ([0, 5], []),              # 非法 token id
        ([1, 2, 3], [0]),          # seqlen < 1
        ([1, 2, 3], [1.5]),        # 非整型
    ]:
        try:
            solve(list(bad_ids), list(bad_seq))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {bad_ids} {bad_seq}")
