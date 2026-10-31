import numpy as np
import torch

KEYS = {"segment_ids", "bias"}


def _oracle(chunk_lengths, prefix_len=0, reencode_num=5, pad_to=None):
    seg = [0] * prefix_len
    for index, length in enumerate(chunk_lengths):
        seg += [index + 1] * length
        seg += [0] * reencode_num
    if pad_to is not None:
        seg += [-1] * (pad_to - len(seg))
    n = len(seg)
    allowed = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if seg[i] == -1:
            continue
        for j in range(i + 1):
            if seg[j] == -1:
                continue
            if seg[i] == 0 or seg[i] == seg[j]:
                allowed[i, j] = True
    bias = np.where(allowed, np.float32(0.0), np.float32(-np.inf)).astype(np.float32)
    return torch.tensor(seg, dtype=torch.int64), torch.tensor(bias)


def _verify(solve, *args, **kwargs):
    got = solve(*args, **kwargs)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    want_seg, want_bias = _oracle(*args, **kwargs)
    assert got["segment_ids"].dtype == torch.int64, got["segment_ids"].dtype
    assert got["bias"].dtype == torch.float32, got["bias"].dtype
    assert torch.equal(got["segment_ids"], want_seg), (
        f"segment_ids：{got['segment_ids'].tolist()} vs {want_seg.tolist()}")
    assert torch.equal(got["bias"], want_bias), "bias 与独立实现不一致"
    return got


def _allowed(bias):
    return [torch.nonzero(bias[i] == 0.0).flatten().tolist() for i in range(bias.shape[0])]


def check_example(solve):
    """题面示例：两个 chunk（2、1 个 token）、1 个 prefix、每段 1 个 link token。"""
    got = _verify(solve, [2, 1], prefix_len=1, reencode_num=1)
    assert got["segment_ids"].tolist() == [0, 1, 1, 0, 2, 0]
    assert got["bias"].shape == (6, 6)
    assert _allowed(got["bias"]) == [[0], [1], [1, 2], [0, 1, 2, 3], [4], [0, 1, 2, 3, 4, 5]]


check_example.example = True


def check_matches_oracle(solve):
    """多组 chunk/prefix/link/pad 组合对拍 numpy 实现。"""
    cases = [
        dict(chunk_lengths=[2, 1], prefix_len=1, reencode_num=1),
        dict(chunk_lengths=[3, 2, 1], prefix_len=0, reencode_num=5),
        dict(chunk_lengths=[1], prefix_len=4, reencode_num=0),
        dict(chunk_lengths=[2, 2], prefix_len=2, reencode_num=3, pad_to=13),
        dict(chunk_lengths=[4, 1, 3], prefix_len=1, reencode_num=0, pad_to=10),
    ]
    for case in cases:
        _verify(solve, **case)


def check_link_tokens_are_global_readers(solve):
    """link token（segment 0）能看到它之前的全部 token；非 0 段落看不到它。"""
    got = _verify(solve, [2, 2], prefix_len=0, reencode_num=1)
    assert got["segment_ids"].tolist() == [1, 1, 0, 2, 2, 0]
    rows = _allowed(got["bias"])
    assert rows[2] == [0, 1, 2], rows[2]
    assert 2 not in rows[3] and 2 not in rows[4], "文档 token 不应看见前一段的 link token"
    assert rows[5] == [0, 1, 2, 3, 4, 5], rows[5]


def check_nonzero_segments_are_isolated(solve):
    """非 0 段落之间互不可见，即使已经越过 causal 下三角。"""
    got = _verify(solve, [2, 2], prefix_len=1, reencode_num=0)
    assert got["segment_ids"].tolist() == [0, 1, 1, 2, 2]
    rows = _allowed(got["bias"])
    assert rows[3] == [3], rows[3]
    assert rows[4] == [3, 4], rows[4]
    assert rows[0] == [0]


def check_causal_and_diagonal(solve):
    """严格上三角恒为 -inf；合法目标的对角线必须可看。"""
    got = _verify(solve, [3, 2], prefix_len=2, reencode_num=2)
    bias = got["bias"]
    n = bias.shape[0]
    upper = torch.ones((n, n), dtype=torch.bool).triu(1)
    assert torch.isinf(bias[upper]).all() and (bias[upper] < 0).all()
    for i in range(n):
        assert bias[i, i] == 0.0, f"有效 token 的对角线不可见：row {i}"


def check_padding_is_fully_masked(solve):
    """pad 的 segment 为 -1：整行整列（含对角）都是 -inf。"""
    got = _verify(solve, [2], prefix_len=1, reencode_num=1, pad_to=6)
    assert got["segment_ids"].tolist() == [0, 1, 1, 0, -1, -1]
    bias = got["bias"]
    for i in (4, 5):
        assert torch.isinf(bias[i]).all() and (bias[i] < 0).all(), f"pad 行 {i} 未全遮"
        assert torch.isinf(bias[:, i]).all() and (bias[:, i] < 0).all(), f"pad 列 {i} 未全遮"


def check_zero_link_tokens(solve):
    """reencode_num=0 时没有 link token：每段只剩自己的内容，段间完全隔离。"""
    got = _verify(solve, [2, 2], prefix_len=0, reencode_num=0)
    assert got["segment_ids"].tolist() == [1, 1, 2, 2]
    rows = _allowed(got["bias"])
    assert rows == [[0], [0, 1], [2], [2, 3]], rows


def check_prefix_only(solve):
    """只有 prefix（全是 segment 0）时退化为纯 causal 掩码。"""
    got = _verify(solve, [], prefix_len=4, reencode_num=5)
    assert got["segment_ids"].tolist() == [0, 0, 0, 0]
    assert _allowed(got["bias"]) == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]


def check_validation(solve):
    """非法 chunk 长度、负 prefix/link、pad_to 过小、类型错误都要抛 ValueError。"""
    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    bad(lambda: solve([2, 0], prefix_len=0, reencode_num=1))
    bad(lambda: solve([2, -1], prefix_len=0, reencode_num=1))
    bad(lambda: solve([2], prefix_len=-1, reencode_num=1))
    bad(lambda: solve([2], prefix_len=0, reencode_num=-1))
    bad(lambda: solve([2], prefix_len=0, reencode_num=1, pad_to=2))
    bad(lambda: solve([2], prefix_len=0, reencode_num=1, pad_to=0))
    bad(lambda: solve("22", prefix_len=0, reencode_num=1))
    bad(lambda: solve([2], prefix_len=0, reencode_num=1.5))
    bad(lambda: solve([True], prefix_len=0, reencode_num=1))
    bad(lambda: solve([2], prefix_len=True, reencode_num=1))
    bad(lambda: solve([2], prefix_len=0, reencode_num=1, pad_to=True))
    bad(lambda: solve([], prefix_len=0, reencode_num=0))
