import torch


def _oracle(scores, n_sink, window, k_spec, k_retr, rhi, chi):
    """逐 head 用 python 重算统计量、分类与选取（结构异于参考解的向量化判定）。"""
    heads, length = scores.shape
    types, keep = [], []
    for h in range(heads):
        row = [float(x) for x in scores[h]]
        total = sum(row)
        rec = sum(row[length - window:]) / total if total > 0 else 0.0
        conc = (max(row) / total) if total > 0 else 0.0
        if rec >= rhi:
            t = 0
        elif conc >= chi:
            t = 1
        else:
            t = 2
        mask = [False] * length
        for j in range(min(n_sink, length)):
            mask[j] = True
        if t == 0:
            for j in range(max(length - window, 0), length):
                mask[j] = True
        else:
            k = k_spec if t == 1 else k_retr
            cand = [j for j in range(n_sink, length)]
            cand.sort(key=lambda j: (-row[j], j))
            for j in cand[:k]:
                mask[j] = True
        types.append(t)
        keep.append(mask)
    keep_t = torch.tensor(keep, dtype=torch.bool)
    return (torch.tensor(types, dtype=torch.long), keep_t,
            keep_t.sum(dim=1))


def check_example(solve):
    torch.manual_seed(0)
    scores = torch.tensor([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 3.0],   # 尾部主导 → streaming
        [3.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],   # 尖峰 → specialized
        [1.0] * 8,                                   # 均匀 → retrieval
    ])
    t, keep, kept = solve(scores, 1, 2, 2, 4, 0.5, 0.25)
    assert t.tolist() == [0, 1, 2], f"got {t.tolist()}"
    assert keep[0].tolist() == [True, False, False, False, False, False, True, True], (
        f"got {keep[0].tolist()}")
    assert keep[1].tolist() == [True, True, True, False, False, False, False, False], (
        f"got {keep[1].tolist()}")
    assert keep[2].tolist() == [True, True, True, True, True, False, False, False], (
        f"got {keep[2].tolist()}")
    assert kept.tolist() == [3, 3, 5], f"got {kept.tolist()}"


check_example.example = True


def check_classification_boundaries(solve):
    """边界取等：R==rhi 判 streaming，C==chi 判 specialized（判定顺序 streaming 优先）。"""
    torch.manual_seed(1)
    # 4 列、window=2：后两列占一半 → R == 0.5
    scores = torch.tensor([[0.5, 0.5, 0.5, 1.0],
                           [1.0, 0.0, 0.0, 3.0]]) / 4.0
    t, _, _ = solve(scores, 0, 2, 1, 2, 0.5, 0.25)
    assert int(t[0]) == 0, f"R==0.5 应判 streaming，got {int(t[0])}"
    assert int(t[1]) == 0, "R 与 C 同时达标时应优先 streaming"
    scores2 = torch.tensor([[2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0]])  # C=2/8=0.25, R=1/8
    t2, _, _ = solve(scores2, 1, 1, 2, 4, 0.9, 0.25)
    assert int(t2[0]) == 1, f"C==0.25 应判 specialized，got {int(t2[0])}"


def check_streaming_ignores_scores(solve):
    """streaming 头只看位置：sink + 末窗，即使末窗分数为 0、中间分最高也不保留中间。"""
    torch.manual_seed(2)
    scores = torch.zeros(2, 10)
    scores[0, 4] = 100.0
    scores[0, 9] = 200.0
    _, keep, _ = solve(scores, 2, 3, 1, 5, 0.5, 0.99)
    assert keep[0, :2].all(), "sink 必须保留"
    assert keep[0, 7:].all(), "末窗必须保留"
    assert not keep[0, 2:7].any(), "中间高分段在 streaming 策略下不保留"


def check_budgets_differ_per_type(solve):
    """specialized 与 retrieval 预算不同：同一行分布下 retrieval 保留更多且含低分段。"""
    torch.manual_seed(3)
    # 尖峰在 index0（sink），其余均匀 → C 高、R 低
    scores = torch.tensor([[9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    _, keep_s, _ = solve(scores, 1, 1, 1, 3, 0.9, 0.25)
    _, keep_r, _ = solve(scores, 1, 1, 1, 3, 0.9, 0.95)
    assert int(keep_s.sum()) == 2, f"specialized 应保留 sink+1，got {int(keep_s.sum())}"
    assert int(keep_r.sum()) == 4, f"retrieval 应保留 sink+3，got {int(keep_r.sum())}"
    assert not torch.equal(keep_s, keep_r)


def check_matches_oracle_and_errors(solve):
    """随机用例与逐 head 重算对拍，并校验非法参数。"""
    torch.manual_seed(21)
    scores = torch.rand(5, 9) ** 2
    for args in ((1, 3, 2, 4, 0.4, 0.2), (2, 1, 0, 9, 0.6, 0.3)):
        exp = _oracle(scores, *args)
        got = solve(scores, *args)
        assert torch.equal(got[0], exp[0]), f"{args} 分类不符"
        assert torch.equal(got[1], exp[1]), f"{args} 掩码不符"
        assert torch.equal(got[2], exp[2]), f"{args} 保留数不符"
    zero = torch.zeros(1, 4)
    t, keep, _ = solve(zero, 1, 2, 1, 2, 0.5, 0.25)
    assert int(t[0]) == 2, "全零头应退化为 retrieval"
    for bad in ((scores, 1, 0, 1, 1, 0.5, 0.25), (scores, 1, 2, 1, 1, 1.5, 0.25)):
        try:
            solve(*bad)
            raise AssertionError(f"{bad[1:]} 应抛 ValueError")
        except ValueError:
            pass
