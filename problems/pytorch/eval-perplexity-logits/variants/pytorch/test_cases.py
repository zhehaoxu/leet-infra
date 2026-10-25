import math

import torch

LOGITS = torch.tensor(
    [
        [[2.0, 0.5, -1.0], [0.0, 1.0, 3.0], [0.2, 0.1, 0.3]],
        [[1.0, 1.0, 1.0], [-2.0, 0.5, 0.5], [0.0, 0.0, 4.0]],
    ]
)
LABELS = torch.tensor([[-100, 0, 2], [1, -100, 1]])
EXAMPLE_PPL = 1.4644929  # 参考解在 (2,3,3) 样例上跑出的 float32 值


def _oracle(logits, labels, ignore_index=-100):
    """独立重算：纯 Python 三层循环 + 手写 logsumexp，完全不走 softmax/gather。"""
    bsz, length, vocab = logits.shape
    total = 0.0
    count = 0
    for b in range(bsz):
        for i in range(length - 1):
            lab = int(labels[b, i + 1])
            if lab == ignore_index:
                continue
            row = [float(logits[b, i, v]) for v in range(vocab)]
            m = max(row)
            lse = m + math.log(sum(math.exp(x - m) for x in row))
            total += lse - row[lab]
            count += 1
    if count == 0:
        return -1.0
    return math.exp(total / count)


def check_example(solve):
    y = solve(LOGITS, LABELS)
    exp = torch.tensor(EXAMPLE_PPL, dtype=torch.float32)
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    assert torch.allclose(
        solve(LOGITS, LABELS),
        torch.tensor(_oracle(LOGITS, LABELS), dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    g = torch.Generator().manual_seed(7)
    for shape in [(3, 5, 6), (2, 4, 3), (4, 7, 2)]:
        lg = torch.randn(*shape, generator=g)
        lb = torch.randint(0, shape[2], (shape[0], shape[1]), generator=g)
        mask = torch.rand(shape[0], shape[1], generator=g) < 0.35
        lb = torch.where(mask, torch.full_like(lb, -100), lb)
        assert torch.allclose(
            solve(lg, lb),
            torch.tensor(_oracle(lg, lb), dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )


def check_all_labels_invalid_returns_sentinel(solve):
    # prompt-only / 全 pad：有效位置数 0 -> 哨兵 -1.0
    lb = torch.full((2, 3), -100, dtype=torch.long)
    y = solve(LOGITS, lb)
    assert torch.allclose(y, torch.tensor(-1.0), atol=0, rtol=0), y.tolist()


def check_single_length_returns_sentinel(solve):
    # L=1 时右移后没有任何配对，等价空集 -> 哨兵
    lg = torch.randn(2, 1, 5, generator=torch.Generator().manual_seed(1))
    lb = torch.tensor([[7], [3]])
    y = solve(lg, lb)
    assert torch.allclose(y, torch.tensor(-1.0), atol=0, rtol=0), y.tolist()


def check_uniform_logits_gives_vocab_size(solve):
    # logits 全等 -> softmax 均匀 1/V -> ppl == V
    vocab = 7
    lg = torch.zeros(2, 4, vocab)
    lb = torch.randint(0, vocab, (2, 4))
    y = solve(lg, lb)
    assert torch.allclose(y, torch.tensor(float(vocab)), atol=1e-4, rtol=1e-4), y.tolist()


def check_perfect_prediction_gives_one(solve):
    # 每个「预测位」都把概率压到它要预测的下一个标签上 -> ppl ≈ 1
    bsz, length, vocab = 2, 4, 5
    lb = torch.randint(0, vocab, (bsz, length), generator=torch.Generator().manual_seed(3))
    lg = torch.full((bsz, length, vocab), -30.0)
    lg.scatter_(2, lb[:, 1:].unsqueeze(-1), 30.0)  # 位置 i 强预测 labels[i+1]
    y = solve(lg, lb)
    assert torch.allclose(y, torch.tensor(1.0), atol=1e-4, rtol=1e-4), y.tolist()


def check_single_valid_position(solve):
    # 只有一个有效位时 ppl = exp(NLL) 逐位复算
    lg = torch.zeros(1, 3, 2)
    lg[0, 1, 1] = 2.0
    lb = torch.tensor([[-100, -100, 1]])
    p = 1.0 / (1.0 + math.exp(-2.0))
    y = solve(lg, lb)
    assert torch.allclose(y, torch.tensor(1.0 / p), atol=1e-5, rtol=1e-5), y.tolist()


def check_shift_direction_matters(solve):
    # 位置 0 的 logits 配 label[1]：若错误地不移位则会用到 label[0]=-100 / 自身 logits
    lg = torch.zeros(1, 2, 2)
    lg[0, 0, 0] = 5.0  # 位置0 强烈预测 id0
    lg[0, 1, 1] = 5.0  # 位置1 强烈预测 id1
    lb = torch.tensor([[-100, 0]])  # 只有 (位置0 -> id0) 这一对有效
    y = solve(lg, lb)
    assert float(y) < 1.05, f"右移对齐应命中强预测，ppl≈1，got {float(y)}"
    lb_bad = torch.tensor([[0, -100]])  # 移位后没有任何有效位
    assert torch.allclose(solve(lg, lb_bad), torch.tensor(-1.0), atol=0, rtol=0)
