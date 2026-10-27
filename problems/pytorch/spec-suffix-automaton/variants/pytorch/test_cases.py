import torch


def _oracle(text, pattern):
    """按 endpos 等价类数状态（与 SAM 的分裂逻辑完全无关的刻画），并暴力枚举子串。"""
    seq = [int(x) for x in text.tolist()]
    n = len(seq)
    endpos = {}
    for i in range(n):
        for j in range(i + 1, n + 1):
            endpos.setdefault(tuple(seq[i:j]), set()).add(j)
    distinct = len(endpos)
    states = len({frozenset(s) for s in endpos.values()}) + 1   # +1 是初始状态
    subs = endpos.keys()

    pat = [int(x) for x in pattern.tolist()]
    matched = []
    for i in range(len(pat)):
        best = 0
        for ln in range(i + 1, 0, -1):
            if tuple(pat[i - ln + 1:i + 1]) in subs:
                best = ln
                break
        matched.append(best)
    return states, distinct, matched


def check_example(solve):
    torch.manual_seed(0)
    text = torch.tensor([1, 2])
    pattern = torch.tensor([1, 2, 1, 3])
    states, distinct, match_len = solve(text, pattern)
    assert int(states) == 3, f"got {int(states)}"
    assert int(distinct) == 3, f"子串 {{1, 2, 12}} 共 3 个，got {int(distinct)}"
    assert match_len.tolist() == [1, 2, 1, 0], f"got {match_len.tolist()}"


check_example.example = True


def check_clone_split_matches_endpos_classes(solve):
    """状态分裂：重复前缀会让状态数超过 n+1，且必须与 endpos 等价类数一致。"""
    torch.manual_seed(1)
    for seq, pat in (([1, 2, 2], [2, 2, 1, 2]), ([1, 2, 1, 2, 3], [2, 1, 2, 1, 1, 2, 9])):
        text = torch.tensor(seq)
        pattern = torch.tensor(pat)
        got = solve(text, pattern)
        exp = _oracle(text, pattern)
        assert int(got[0]) == exp[0], f"{seq} 状态数 {int(got[0])} vs {exp[0]}"
        assert int(got[1]) == exp[1], f"{seq} 不同子串数 {int(got[1])} vs {exp[1]}"
        assert got[2].tolist() == exp[2], f"{seq} match_len {got[2].tolist()} vs {exp[2]}"
        n = len(seq)
        assert n + 1 <= int(got[0]) <= 2 * n - 1, "最小 DFA 的状态数应在 [n+1, 2n-1]"

    # [1,2,2] 的 "2" 有两个后缀实例，必然分裂：5 = 4 + 1
    assert int(solve(torch.tensor([1, 2, 2]), torch.tensor([]))[0]) == 5
    # [1,2,1,2,3] 的 endpos 恰好两两不同，不分裂：6 = n + 1
    assert int(solve(torch.tensor([1, 2, 1, 2, 3]), torch.tensor([]))[0]) == 6


def check_uniform_text_and_reset(solve):
    """全同 token：子串数 = n；失配后 best 要能正确回落到 0 再重新长起来。"""
    torch.manual_seed(2)
    text = torch.tensor([7, 7, 7])
    pattern = torch.tensor([7, 9, 7, 7])
    states, distinct, match_len = solve(text, pattern)
    assert int(distinct) == 3, f"got {int(distinct)}"
    assert int(states) == 4, f"got {int(states)}"
    assert match_len.tolist() == [1, 0, 1, 2], f"got {match_len.tolist()}"


def check_boundaries_and_invariants(solve):
    """空查询、全未见 token、与语料相同的前缀，以及 match_len 的增长约束。"""
    torch.manual_seed(3)
    text = torch.tensor([3, 1, 4, 1, 5])
    states, distinct, match_len = solve(text, torch.tensor([], dtype=torch.int64))
    assert match_len.tolist() == [] and match_len.dtype == torch.int64, "空查询应返回空 int64"
    assert int(distinct) == _oracle(text, torch.tensor([], dtype=torch.int64))[1]
    assert int(states) == _oracle(text, torch.tensor([], dtype=torch.int64))[0]

    _, _, none_hit = solve(text, torch.tensor([9, 9, 9]))
    assert none_hit.tolist() == [0, 0, 0], f"got {none_hit.tolist()}"

    _, _, same = solve(text, text.clone())
    assert same.tolist() == [1, 2, 3, 4, 5], f"查询与语料相同时应逐位累加，got {same.tolist()}"

    _, _, got = solve(text, torch.tensor([1, 4, 1, 4, 1, 4]))
    for i, value in enumerate(got.tolist()):
        assert value <= min(i + 1, int(text.numel())), f"第 {i} 位超过长度上界"
        assert i == 0 or value <= got[i - 1].item() + 1, "每多一个 token 最多长 1"


def check_matches_oracle_and_errors(solve):
    """随机小语料对拍（覆盖分裂、无分裂、单 token 词表）+ 非法输入。"""
    torch.manual_seed(21)
    for _ in range(6):
        n = int(torch.randint(1, 9, (1,)))
        vocab = int(torch.randint(1, 4, (1,)))
        text = torch.randint(0, vocab, (n,))
        pattern = torch.randint(0, vocab + 1, (int(torch.randint(0, 7, (1,))),))
        got = solve(text, pattern)
        exp = _oracle(text, pattern)
        assert int(got[0]) == exp[0], f"{text.tolist()} 状态数不符"
        assert int(got[1]) == exp[1], f"{text.tolist()} 子串数不符"
        assert got[2].tolist() == exp[2], f"{text.tolist()} {pattern.tolist()} match_len 不符"

    try:
        solve(torch.tensor([], dtype=torch.int64), torch.tensor([1]))
        raise AssertionError("空语料应抛 ValueError")
    except ValueError:
        pass
    try:
        solve(torch.zeros(2, 2, dtype=torch.int64), torch.tensor([1]))
        raise AssertionError("二维输入应抛 ValueError")
    except ValueError:
        pass
