import itertools
import math

import torch


def _collapse(seq, blank):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return [x for x in out if x != blank]


def _oracle(log_probs, labels):
    """暴力枚举全部 V^T 条路径，坍缩后等于目标标签者累乘概率求和，取 -log。"""
    T, V = log_probs.shape
    blank = V - 1
    probs = log_probs.exp()
    tgt = [int(v) for v in labels.tolist()]
    tot = 0.0
    for path in itertools.product(range(V), repeat=T):
        if _collapse(list(path), blank) == tgt:
            pr = 1.0
            for tt, a in enumerate(path):
                pr *= float(probs[tt, a])
            tot += pr
    return -math.log(tot)


LP = torch.tensor([[0.3, 0.6, 0.1], [0.5, 0.2, 0.3], [0.1, 0.4, 0.5]]).log()


def check_example(solve):
    out = solve(LP, torch.tensor([0]))
    assert abs(out.item() - 1.78379130) < 1e-5, out.item()


check_example.example = True


def check_labels_01(solve):
    out = solve(LP, torch.tensor([0, 1]))
    assert abs(out.item() - 1.77195680) < 1e-5, out.item()


def check_matches_oracle(solve):
    torch.manual_seed(13)
    lp = torch.log_softmax(torch.randn(4, 3), dim=-1)
    for lab in ([0], [1], [0, 1], [1, 0], [0, 1, 0]):
        y = solve(lp, torch.tensor(lab))
        assert abs(y.item() - _oracle(lp, torch.tensor(lab))) < 1e-5, lab


def check_repeat_labels(solve):
    # 相邻重复标签需 blank 分隔，验证与暴力枚举一致
    torch.manual_seed(21)
    lp = torch.log_softmax(torch.randn(5, 3), dim=-1)
    for lab in ([0, 0], [1, 1], [0, 0, 1], [1, 0, 0]):
        y = solve(lp, torch.tensor(lab))
        assert abs(y.item() - _oracle(lp, torch.tensor(lab))) < 1e-5, lab


def check_empty_labels(solve):
    # 空标签：唯一接受路径是全 blank，损失 = -Σ log p[t, blank]
    blank = LP.shape[1] - 1
    expect = -sum(float(LP[t, blank]) for t in range(LP.shape[0]))
    out = solve(LP, torch.tensor([], dtype=torch.int64))
    assert abs(out.item() - expect) < 1e-5, (out.item(), expect)
