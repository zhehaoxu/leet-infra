import torch


def _oracle(head_scores, seq_len, sink_size, recent_size, head_compression_ratio):
    """独立纯 Python：选流式头，逐 (h,t) 判定 keep，不做向量化切片。"""
    H = len(head_scores)
    L = int(seq_len)
    S = int(sink_size)
    R = int(recent_size)
    n_pruned = int(H * head_compression_ratio + 0.5)
    order = sorted(range(H), key=lambda h: (float(head_scores[h]), h))
    streaming = set(order[:n_pruned])
    mask = []
    for h in range(H):
        row = []
        for t in range(L):
            if h not in streaming:
                row.append(True)
            else:
                row.append(bool(t < S or t >= L - R))
        mask.append(row)
    return torch.tensor(mask, dtype=torch.bool)


def check_example(solve):
    head_scores = torch.tensor([0.9, 0.2, 0.7, 0.1, 0.5])
    out = solve(head_scores, 8, 2, 2, 0.4)
    exp = torch.tensor([
        [True, True, True, True, True, True, True, True],
        [True, True, False, False, False, False, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, False, False, False, False, True, True],
        [True, True, True, True, True, True, True, True],
    ])
    assert torch.equal(out, exp), out


def check_matches_oracle(solve):
    torch.manual_seed(4)
    for _ in range(40):
        H = int(torch.randint(1, 9, (1,)))
        L = int(torch.randint(1, 14, (1,)))
        S = int(torch.randint(0, L + 1, (1,)))
        R = int(torch.randint(0, L + 1, (1,)))
        ratio = float(torch.rand(1))
        scores = torch.rand(H)
        got = solve(scores, L, S, R, ratio)
        exp = _oracle(scores, L, S, R, ratio)
        assert torch.equal(got, exp), (H, L, S, R, ratio, got, exp)


def check_retrieval_heads_full(solve):
    # ratio=0 → 无流式头，全部整行 True
    scores = torch.tensor([0.3, 0.9, 0.1])
    out = solve(scores, 6, 1, 1, 0.0)
    assert torch.equal(out, torch.ones(3, 6, dtype=torch.bool)), out


def check_streaming_prunes_middle(solve):
    # 全成流式头 → 每行都只剩首 S + 尾 R
    scores = torch.tensor([0.1, 0.2])
    out = solve(scores, 6, 2, 2, 1.0)
    exp = torch.tensor([
        [True, True, False, False, True, True],
        [True, True, False, False, True, True],
    ])
    assert torch.equal(out, exp), out


def check_windows_cover_all(solve):
    # S + R >= L：流式头也等价整行 True
    scores = torch.tensor([0.1, 0.9])
    out = solve(scores, 5, 3, 3, 1.0)  # 全流式，但窗口重叠覆盖全长
    assert torch.equal(out, torch.ones(2, 5, dtype=torch.bool)), out


check_example.example = True
