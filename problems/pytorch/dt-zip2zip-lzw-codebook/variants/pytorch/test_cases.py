import numpy as np

KEYS = {"compressed_ids", "attention_mask", "codebook"}


def _oracle_decode(compressed, initial_vocab_size, max_codebook_size,
                   max_subtokens, disabled_ids):
    """独立实现上游 decode_fn（2610.13605 zip2zip-compression codec.rs），
    用于对 solve 的编码结果做往返校验。"""
    disabled = set(disabled_ids or [])
    reverse = {}
    buffer = []
    next_id = initial_vocab_size
    out = []
    for cid in compressed:
        if cid in disabled:
            buffer.clear()
            out.append(cid)
            continue
        if cid < initial_vocab_size:
            decoded = [cid]
        elif cid in reverse:
            decoded = list(reverse[cid])
        elif len(buffer) == max_subtokens:
            decoded = list(buffer)          # force merge
        else:
            buffer.append(buffer[0])        # KwKwK 自指合并
            reverse[cid] = list(buffer)
            next_id += 1
            out.extend(buffer)
            continue
        out.extend(decoded)
        if next_id == initial_vocab_size + max_codebook_size:
            buffer.clear()
            continue
        if not buffer:
            buffer = list(decoded)
            continue
        if len(buffer) == max_subtokens:
            buffer = list(decoded)
            continue
        dec = list(decoded)
        while dec:
            buffer.append(dec[0])
            phrase = tuple(buffer)
            if phrase not in {tuple(v) for v in reverse.values()}:
                reverse[next_id] = list(buffer)
                next_id += 1
                buffer = list(dec)
                break
            elif len(buffer) == max_subtokens:
                buffer = dec[1:]
                break
            dec.pop(0)
    return out


def _verify(solve, ids, initial_vocab_size, max_codebook_size, max_subtokens,
            pad_token_id, **kwargs):
    snap = list(ids)
    got = solve(ids, initial_vocab_size, max_codebook_size, max_subtokens,
                pad_token_id, **kwargs)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    comp = got["compressed_ids"]
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in comp)
    assert len(got["attention_mask"]) == len(comp)
    for c, m in zip(comp, got["attention_mask"]):
        assert m == (0 if c == pad_token_id else 1)
    hyper_ids = sorted(got["codebook"])
    assert hyper_ids == list(range(initial_vocab_size,
                                   initial_vocab_size + len(hyper_ids))), hyper_ids
    for hyper, phrase in got["codebook"].items():
        assert 2 <= len(phrase) <= max_subtokens
        assert all(0 <= x < initial_vocab_size for x in phrase)
    assert ids == snap, "不应修改输入 ids"
    # 往返：无 truncation 且无 padding 时，独立解码器必须还原原序列
    if not kwargs.get("truncation") and kwargs.get("padding", "do_not_pad") == "do_not_pad":
        roundtrip = _oracle_decode(
            comp, initial_vocab_size, max_codebook_size, max_subtokens,
            kwargs.get("disabled_ids"))
        assert roundtrip == list(ids), f"往返失败：{roundtrip} vs {list(ids)}"
    return got


def check_example(solve):
    """题面示例：vocab 4 / 码本 6 / max_subtokens 3 的 LZW 生长期。"""
    got = _verify(solve, [0, 1, 0, 1, 0, 1, 2], 4, 6, 3, 100)
    assert got["compressed_ids"] == [0, 1, 4, 4, 2], got["compressed_ids"]
    assert got["attention_mask"] == [1, 1, 1, 1, 1]
    assert got["codebook"] == {4: [0, 1], 5: [1, 0], 6: [0, 1, 0], 7: [0, 1, 2]}


check_example.example = True


def check_matches_oracle(solve):
    """多组固定数据对拍 + 往返：重复模式、禁用 id、满码本、长随机序列。"""
    cases = [
        # 满码本：limit=6，后续短语不再入本、逐 token 发射
        dict(ids=[0, 1, 0, 1, 2, 2, 3, 3], ivs=4, mcs=2, mst=3, pad=100),
        # disabled id 打断合并
        dict(ids=[0, 1, 3, 0, 1, 0, 1], ivs=4, mcs=8, mst=3, pad=100,
             disabled_ids=[3]),
        # max_subtokens=2 的紧凑合并
        dict(ids=[1, 1, 1, 1, 1, 2, 2], ivs=4, mcs=6, mst=2, pad=100),
        # 长序列：重复块让码本快速生长再撞上限
        dict(ids=list(range(4)) * 6, ivs=4, mcs=5, mst=4, pad=100),
        # 全 disabled
        dict(ids=[3, 3, 3], ivs=4, mcs=4, mst=2, pad=100, disabled_ids=[3]),
    ]
    for c in cases:
        _verify(solve, c["ids"], c["ivs"], c["mcs"], c["mst"], c["pad"],
                disabled_ids=c.get("disabled_ids"))


def check_truncation(solve):
    """truncation：发射封顶、输入提前终止，码本只按已消费的输入生长。"""
    got = _verify(solve, [0, 1, 0, 1, 0, 1, 2], 4, 6, 3, 100,
                  truncation=True, max_length=3)
    assert got["compressed_ids"] == [0, 1, 4], got["compressed_ids"]
    assert got["attention_mask"] == [1, 1, 1]
    # 达到上限后 while 顶部早退：码本只到 [0,1,0]，[0,1,2] 不会入本
    assert got["codebook"] == {4: [0, 1], 5: [1, 0], 6: [0, 1, 0]}

    # max_length=1：发射第一条后立即停
    got = _verify(solve, [0, 1, 0, 1], 4, 6, 3, 100, truncation=True, max_length=1)
    assert got["compressed_ids"] == [0]
    assert got["codebook"] == {4: [0, 1]}


def check_padding_modes(solve):
    """max_length 左填充：pad 在左、mask 对应置 0；超长未截断要报错。"""
    got = _verify(solve, [0, 1, 0, 1, 0, 1, 2], 4, 6, 3, 100,
                  padding="max_length", max_length=8)
    assert got["compressed_ids"] == [100, 100, 100, 0, 1, 4, 4, 2]
    assert got["attention_mask"] == [0, 0, 0, 1, 1, 1, 1, 1]

    # truncation + max_length 组合：同一 max_length 既封顶发射又作填充目标；
    # 输入耗尽时压缩结果短于 max_length，触发左填充
    got = _verify(solve, [0, 1, 0, 1], 4, 6, 3, 100,
                  truncation=True, max_length=5, padding="max_length")
    assert got["compressed_ids"] == [100, 100, 0, 1, 4], got["compressed_ids"]
    assert got["attention_mask"] == [0, 0, 1, 1, 1]
    assert got["codebook"] == {4: [0, 1], 5: [1, 0]}

    # 截断到恰好 max_length 时填充为空操作
    got = _verify(solve, [0, 1, 0, 1, 0, 1, 2], 4, 6, 3, 100,
                  truncation=True, max_length=3, padding="max_length")
    assert got["compressed_ids"] == [0, 1, 4]
    assert got["attention_mask"] == [1, 1, 1]

    # 恰好等长：不填充
    got = _verify(solve, [0, 1, 0, 1, 0, 1, 2], 4, 6, 3, 100,
                  padding="max_length", max_length=5)
    assert got["compressed_ids"] == [0, 1, 4, 4, 2]
    assert got["attention_mask"] == [1] * 5


def check_codebook_freeze(solve):
    """码本满后不再生长：后续短语逐 token 发射，next_id 冻结在上限。"""
    got = _verify(solve, [0, 1, 0, 1, 2, 2, 3, 3], 4, 2, 3, 100)
    assert got["codebook"] == {4: [0, 1], 5: [1, 0]}
    assert got["compressed_ids"] == [0, 1, 4, 2, 2, 3, 3], got["compressed_ids"]


def check_max_subtokens_cap(solve):
    """短语达到 max_subtokens 即发射清空，不会出现更长短语。"""
    got = _verify(solve, [0, 0, 0, 0, 0, 0], 4, 8, 2, 100)
    # [0] -> [0,0] 未知入本(4) 发 0；buffer=[0] -> [0,0] 已知且 len==2 发 4 清空，循环往复
    # 末尾 buffer=[0] 走 flush 发 0
    assert got["compressed_ids"] == [0, 4, 4, 0], got["compressed_ids"]
    assert got["codebook"] == {4: [0, 0]}


def check_validation(solve):
    """类型、范围、组合约束等非法输入抛 ValueError。"""

    def bad(fn, tag=""):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError: " + tag)

    base = dict(initial_vocab_size=4, max_codebook_size=6, max_subtokens=3,
                pad_token_id=100)
    bad(lambda: solve([], **base))
    bad(lambda: solve([0, "a"], **base))
    bad(lambda: solve([0, True], **base))
    bad(lambda: solve([0, -1], **base))
    bad(lambda: solve([0, 4], **base))                      # 基词 id 越界
    bad(lambda: solve([0, 1], initial_vocab_size=0, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=0,
                      max_subtokens=3, pad_token_id=100))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=1, pad_token_id=100))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=5))     # pad 与发射 id 重叠
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100, disabled_ids=[4]))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100, disabled_ids=[1, 1]))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100, truncation=True))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100, truncation=1))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100, padding="longest"))
    bad(lambda: solve([0, 1], initial_vocab_size=4, max_codebook_size=6,
                      max_subtokens=3, pad_token_id=100, padding="max_length"))
    bad(lambda: solve([0, 1, 0, 1, 0, 1, 2], initial_vocab_size=4,
                      max_codebook_size=6, max_subtokens=3, pad_token_id=100,
                      padding="max_length", max_length=3))  # 未截断且超长
