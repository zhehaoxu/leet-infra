import torch


def _oracle(datastore, context, gamma, min_match):
    """独立纯 Python：枚举全部 (n, entry, pos) 匹配候选，按 (n, -entry, -pos) 取最大。"""
    cands = []
    for e_idx, entry in enumerate(datastore):
        entry = list(entry)
        for n in range(min_match, len(context) + 1):
            if n > len(entry):
                continue
            suffix = list(context[-n:])
            for i in range(len(entry) - n + 1):
                if list(entry[i:i + n]) == suffix:
                    cands.append((n, e_idx, i))
    if not cands:
        return []
    n, e_idx, i = max(cands, key=lambda t: (t[0], -t[1], -t[2]))
    return list(datastore[e_idx][i + n:i + n + gamma])


def check_example(solve):
    ds = [[1, 2, 3, 4, 5], [9, 2, 3, 7, 8], [2, 3, 6]]
    out = solve(ds, [0, 1, 2, 3], 2, 2)
    assert torch.equal(out, torch.tensor([4, 5])), out


def check_no_match(solve):
    # 仅长度 1 的尾巴能对上，低于 min_match=2 -> 空
    out = solve([[8, 5]], [1, 2, 5], 2, 2)
    assert out.numel() == 0, out


def check_tie_break_entry(solve):
    # 同长 [1,2,3] 在 entry0/entry1 都命中 -> 取更早 entry0 的后继 8
    ds = [[7, 1, 2, 3, 8], [9, 1, 2, 3, 7]]
    out = solve(ds, [5, 1, 2, 3], 1, 2)
    assert torch.equal(out, torch.tensor([8])), out


def check_earliest_position(solve):
    # 同 entry 内 [1,2,3] 出现在 i=0 与 i=3 -> 取最早 i=0，草稿 [1,2]
    out = solve([[1, 2, 3, 1, 2, 3, 9]], [0, 1, 2, 3], 2, 3)
    assert torch.equal(out, torch.tensor([1, 2])), out


def check_gamma_truncate(solve):
    # 匹配点后面只剩 1 个 token，gamma=5 也被截断
    out = solve([[1, 2, 3, 4]], [0, 1, 2, 3], 5, 2)
    assert torch.equal(out, torch.tensor([4])), out


def check_matches_oracle(solve):
    torch.manual_seed(31)
    for _ in range(60):
        V = int(torch.randint(2, 7, (1,)))
        ne = int(torch.randint(1, 4, (1,)))
        ds = []
        for _ in range(ne):
            ln = int(torch.randint(2, 7, (1,)))
            ds.append([int(x) for x in torch.randint(0, V, (ln,))])
        clen = int(torch.randint(1, 5, (1,)))
        context = [int(x) for x in torch.randint(0, V, (clen,))]
        gamma = int(torch.randint(1, 4, (1,)))
        min_match = int(torch.randint(1, clen + 1, (1,)))
        out = solve(ds, context, gamma, min_match)
        exp = torch.tensor(_oracle(ds, context, gamma, min_match), dtype=torch.long)
        assert torch.equal(out, exp), (ds, context, gamma, min_match, out, exp)


check_example.example = True
