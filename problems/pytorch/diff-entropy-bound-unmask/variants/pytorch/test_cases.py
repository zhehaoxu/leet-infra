import math

import torch


def _entropy(scaled):
    m = max(scaled)
    lse = m + math.log(sum(math.exp(v - m) for v in scaled))
    total = 0.0
    for v in scaled:
        log_p = v - lse
        total -= math.exp(log_p) * log_p
    return total


def _oracle(logits, sampled, random_tokens, timestep, num_inference_steps, bound, t_max, t_min):
    fraction = (num_inference_steps - timestep) / num_inference_steps
    temperature = t_min + (t_max - t_min) * fraction
    prev, accepted, entropy = [], [], []
    for row_logits, row_sampled, row_random in zip(logits, sampled, random_tokens):
        scaled = [[v / temperature for v in position] for position in row_logits]
        row_entropy = [_entropy(position) for position in scaled]
        order = sorted(range(len(row_entropy)), key=lambda i: (row_entropy[i], i))
        cum = 0.0
        acc = [False] * len(row_entropy)
        for i in order:
            if cum <= bound:
                acc[i] = True
            cum += row_entropy[i]
        prev.append([s if a else r for a, s, r in zip(acc, row_sampled, row_random)])
        accepted.append(acc)
        entropy.append(row_entropy)
    return prev, accepted, entropy


def check_example(solve):
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    sampled = torch.tensor([[1, 1, 0]])
    random_tokens = torch.tensor([[2, 2, 1]])

    prev, accepted, entropy = solve(logits, sampled, random_tokens, 0, 4, entropy_bound=1.0)
    assert prev.dtype == torch.int64 and accepted.dtype == torch.bool and entropy.dtype == torch.float32
    assert tuple(prev.shape) == (1, 3) and tuple(accepted.shape) == (1, 3) and tuple(entropy.shape) == (1, 3)
    assert [round(float(v), 6) for v in entropy[0]] == [0.365770, 1.098612, 0.983653], entropy
    assert prev[0].tolist() == [1, 2, 0], prev
    assert accepted[0].tolist() == [True, False, True], accepted

    prev2, accepted2, _ = solve(logits, sampled, random_tokens, 0, 4, entropy_bound=0.2)
    assert prev2[0].tolist() == [1, 2, 1], prev2
    assert accepted2[0].tolist() == [True, False, False], accepted2


check_example.example = True


def check_default_bound(solve):
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    sampled = torch.tensor([[1, 1, 0]])
    random_tokens = torch.tensor([[2, 2, 1]])
    prev, accepted, _ = solve(logits, sampled, random_tokens, 0, 4)
    assert accepted[0].tolist() == [True, False, False], "默认熵界 0.1 只接收最低熵的位置"
    assert prev[0].tolist() == [1, 2, 1]

    prev_all, accepted_all, _ = solve(logits, sampled, random_tokens, 0, 4, entropy_bound=10.0)
    assert accepted_all[0].tolist() == [True, True, True]
    assert prev_all[0].tolist() == [1, 1, 0], "全部接收时必须用候选 token"


def check_temperature_anneal(solve):
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    sampled = torch.tensor([[1, 1, 0]])
    random_tokens = torch.tensor([[2, 2, 1]])

    _, _, last = solve(logits, sampled, random_tokens, 3, 4, entropy_bound=1.0)
    got = [round(float(v), 6) for v in last[0]]
    assert got == [0.106921, 1.098612, 0.885382], got
    assert abs(got[1] - 1.098612) < 1e-5, "均匀分布最大熵 log(3) 与温度无关"

    _, _, unit = solve(logits, sampled, random_tokens, 0, 4, t_max=1.0, t_min=1.0)
    p = torch.softmax(torch.tensor([[2.0, 0.0, -1.0]]), dim=-1)[0]
    manual = float(-(p * torch.log(p)).sum())
    assert abs(float(unit[0, 0]) - manual) < 1e-5, (unit[0, 0], manual)

    _, _, colder = solve(logits, sampled, random_tokens, 3, 4, t_max=0.5, t_min=0.2)
    assert float(colder[0, 0]) < float(last[0, 0]), "温度更低时尖锐分布的熵应更小"


def check_acceptance_semantics(solve):
    logits = torch.tensor([[
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]])
    sampled = torch.tensor([[1, 2, 3, 4, 5]])
    random_tokens = torch.tensor([[6, 7, 0, 1, 2]])

    prevs = []
    sets = []
    for bound in (0.0, 0.3, 0.8, 1.5, 10.0):
        prev, accepted, entropy = solve(logits, sampled, random_tokens, 0, 4, entropy_bound=bound)
        acc = accepted[0].tolist()
        sets.append({i for i, a in enumerate(acc) if a})
        prevs.append(prev)
        for i, a in enumerate(acc):
            want = sampled[0, i].item() if a else random_tokens[0, i].item()
            assert prev[0, i].item() == want, (bound, i, prev[0].tolist())
        # 被接收的位置严格是「按熵升序的前缀」
        order = sorted(range(5), key=lambda i: (float(entropy[0, i]), i))
        k = len(sets[-1])
        assert sets[-1] == set(order[:k]), (bound, sets[-1], order)

    assert sets[0] == {0}, "熵界为 0 时只接收熵最小的位置"
    assert sets[1] <= sets[2] <= sets[3] <= sets[4], "熵界增大时接收集合必须单调扩张"
    assert len(sets) == 5 and len(sets[4]) == 5
    assert torch.equal(prevs[4], sampled), "熵界足够大时全部用候选 token"


def check_ties_prefer_lower_index(solve):
    logits = torch.zeros(1, 4, 4)
    sampled = torch.tensor([[1, 2, 3, 0]])
    random_tokens = torch.tensor([[0, 1, 2, 3]])
    _, accepted, entropy = solve(logits, sampled, random_tokens, 0, 4, entropy_bound=0.0)
    assert accepted[0].tolist() == [True, False, False, False], accepted
    assert float(entropy.max() - entropy.min()) < 1e-6, "同分布的熵必须相等"


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(20261029)
    for _ in range(20):
        batch = int(torch.randint(1, 4, (1,), generator=g).item())
        block = int(torch.randint(1, 6, (1,), generator=g).item())
        vocab = int(torch.randint(2, 7, (1,), generator=g).item())
        logits = torch.randn(batch, block, vocab, generator=g) * 2.5
        sampled = torch.randint(0, vocab, (batch, block), generator=g)
        random_tokens = torch.randint(0, vocab, (batch, block), generator=g)
        num_inference_steps = int(torch.randint(1, 9, (1,), generator=g).item())
        timestep = int(torch.randint(0, num_inference_steps, (1,), generator=g).item())
        bound = float(torch.rand(1, generator=g).item()) * 3.0
        t_max = float(torch.rand(1, generator=g).item()) * 1.3 + 0.2
        t_min = float(torch.rand(1, generator=g).item()) * 1.3 + 0.2

        prev, accepted, entropy = solve(
            logits, sampled, random_tokens, timestep, num_inference_steps,
            entropy_bound=bound, t_max=t_max, t_min=t_min,
        )
        want_prev, want_acc, want_ent = _oracle(
            logits.tolist(), sampled.tolist(), random_tokens.tolist(),
            timestep, num_inference_steps, bound, t_max, t_min,
        )
        assert prev.tolist() == want_prev, (prev.tolist(), want_prev)
        assert accepted.tolist() == want_acc, (accepted.tolist(), want_acc)
        assert torch.allclose(entropy, torch.tensor(want_ent, dtype=torch.float32), rtol=1e-5, atol=1e-6), (
            entropy, want_ent,
        )


def check_bad_input(solve):
    logits = torch.randn(2, 3, 5)
    sampled = torch.randint(0, 5, (2, 3))
    random_tokens = torch.randint(0, 5, (2, 3))
    bads = [
        lambda: solve(logits[0], sampled, random_tokens, 0, 4),
        lambda: solve(logits.reshape(2, 15), sampled, random_tokens, 0, 4),
        lambda: solve(logits.to(torch.int64), sampled, random_tokens, 0, 4),
        lambda: solve(torch.full((2, 3, 5), float("nan")), sampled, random_tokens, 0, 4),
        lambda: solve(torch.randn(0, 3, 5), sampled[:0], random_tokens[:0], 0, 4),
        lambda: solve(torch.randn(2, 3, 0), sampled, random_tokens, 0, 4),
        lambda: solve(logits, sampled.to(torch.float32), random_tokens, 0, 4),
        lambda: solve(logits, sampled[:, :2], random_tokens, 0, 4),
        lambda: solve(logits, sampled, random_tokens.unsqueeze(0), 0, 4),
        lambda: solve(logits, sampled + 5, random_tokens, 0, 4),
        lambda: solve(logits, sampled, random_tokens - 6, 0, 4),
        lambda: solve(logits, sampled, random_tokens, 4, 4),
        lambda: solve(logits, sampled, random_tokens, -1, 4),
        lambda: solve(logits, sampled, random_tokens, 0, 0),
        lambda: solve(logits, sampled, random_tokens, 0.0, 4),
        lambda: solve(logits, sampled, random_tokens, True, 4),
        lambda: solve(logits, sampled, random_tokens, 0, 4, entropy_bound=-0.1),
        lambda: solve(logits, sampled, random_tokens, 0, 4, entropy_bound=float("inf")),
        lambda: solve(logits, sampled, random_tokens, 0, 4, entropy_bound=True),
        lambda: solve(logits, sampled, random_tokens, 0, 4, t_max=0.0),
        lambda: solve(logits, sampled, random_tokens, 0, 4, t_min=-0.5),
        lambda: solve(logits, sampled, random_tokens, 0, 4, t_max=float("nan")),
    ]
    for fn in bads:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("非法输入应抛 ValueError")
