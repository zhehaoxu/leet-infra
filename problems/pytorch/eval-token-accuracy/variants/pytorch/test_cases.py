import torch

PRED = torch.tensor([[7, 7, 1, 2, 9], [4, 6, 1, 1, 1]])
LABELS = torch.tensor([[-100, -100, 1, 2, 3], [4, 5, -100, -100, -100]])
EXAMPLE_MICRO = 0.6            # 参考解跑出的值
EXAMPLE_MACRO = 0.5833333333   # (2/3 + 1/2) / 2


def _close(a, b):
    return torch.allclose(
        torch.tensor(float(a), dtype=torch.float64),
        torch.tensor(float(b), dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def _oracle(pred, labels, ignore_index=-100):
    """独立重算：纯 Python 逐元素双循环计数，不用任何张量归约。"""
    bsz, length = pred.shape
    hits = 0
    valids = 0
    per_sample = []
    for b in range(bsz):
        sh = 0
        sv = 0
        for i in range(length):
            if int(labels[b, i]) == ignore_index:
                continue
            sv += 1
            if int(pred[b, i]) == int(labels[b, i]):
                sh += 1
        hits += sh
        valids += sv
        if sv > 0:
            per_sample.append(sh / sv)
    micro = hits / valids if valids > 0 else 0.0
    macro = sum(per_sample) / len(per_sample) if per_sample else 0.0
    return micro, macro


def check_example(solve):
    micro, macro = solve(PRED, LABELS)
    assert _close(micro, EXAMPLE_MICRO), f"micro got {micro}"
    assert _close(macro, EXAMPLE_MACRO), f"macro got {macro}"
    assert abs(micro - macro) > 1e-3, "样例应当体现 micro / macro 的差异"


check_example.example = True


def check_matches_oracle(solve):
    for pred, labels in [(PRED, LABELS),
                         (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[1, -100, 3], [-100, -100, 6]])),
                         (torch.tensor([[1], [2], [3]]), torch.tensor([[-100], [2], [7]]))]:
        m1, a1 = solve(pred, labels)
        m2, a2 = _oracle(pred, labels)
        assert _close(m1, m2) and _close(a1, a2), (pred.tolist(), labels.tolist(), m1, a1, m2, a2)


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(11)
    for _ in range(6):
        bsz = int(torch.randint(1, 5, (1,), generator=g))
        length = int(torch.randint(1, 9, (1,), generator=g))
        vocab = int(torch.randint(2, 7, (1,), generator=g))
        labels = torch.randint(0, vocab, (bsz, length), generator=g)
        drop = torch.rand(bsz, length, generator=g) < 0.4
        labels = torch.where(drop, torch.full_like(labels, -100), labels)
        pred = torch.randint(0, vocab, (bsz, length), generator=g)
        m1, a1 = solve(pred, labels)
        m2, a2 = _oracle(pred, labels)
        assert _close(m1, m2) and _close(a1, a2), (pred.tolist(), labels.tolist())


def check_no_valid_token_returns_zero(solve):
    # 全 prompt / 全 pad：空评测集 -> (0.0, 0.0)
    labels = torch.full((3, 4), -100, dtype=torch.long)
    pred = torch.randint(0, 5, (3, 4))
    micro, macro = solve(pred, labels)
    assert micro == 0.0 and macro == 0.0, (micro, macro)


def check_perfect_prediction_gives_one(solve):
    # 预测完全等于标签：两个口径都是 1.0
    labels = torch.tensor([[1, 2, -100, 4], [5, -100, -100, -100]])
    micro, macro = solve(labels, labels)
    assert micro == 1.0 and macro == 1.0, (micro, macro)


def check_equal_valid_lengths_agree(solve):
    # 每个样本有效位数相同 -> micro == macro（不对称消失）
    pred = torch.tensor([[1, 2, 0, 9], [1, 0, 3, 9]])
    labels = torch.tensor([[1, 2, -100, -100], [1, 5, -100, -100]])
    micro, macro = solve(pred, labels)
    assert _close(micro, 0.75), micro
    assert _close(macro, 0.75), macro


def check_unequal_valid_lengths_diverge(solve):
    # 行0 只有 1 个有效位且答对，行1 有 4 个有效位且只答对 1 个
    pred = torch.tensor([[7, 7, 3, 7], [1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, -100], [1, 9, 9, 9]])
    micro, macro = solve(pred, labels)
    assert _close(micro, 2.0 / 5.0), micro          # (1+1)/(1+4)
    assert _close(macro, 0.625), macro              # (1.0 + 0.25)/2
    assert abs(micro - macro) > 1e-6


def check_ignored_positions_never_count(solve):
    # 把 ignore_index 位置上随便改成命中，两个口径都不许变
    pred = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[1, -100, 3, -100]])
    before = solve(pred, labels)
    pred2 = torch.tensor([[1, 9, 3, 9]])
    pred2[0, 1] = -100  # 在忽略位上「猜中」ignore_index 本身
    pred2[0, 3] = -100
    after = solve(pred2, labels)
    assert _close(before[0], after[0]) and _close(before[1], after[1]), (before, after)


def check_all_wrong_gives_zero(solve):
    pred = torch.tensor([[0, 0, 5, 5]])
    labels = torch.tensor([[1, 2, 3, 4]])
    micro, macro = solve(pred, labels)
    assert micro == 0.0 and macro == 0.0, (micro, macro)
