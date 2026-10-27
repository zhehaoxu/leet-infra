import torch


def _oracle(attn, budget, n_sink):
    """逐标量重算：python 双层循环累加 + 每步重排（结构异于参考解的 cumsum 路径）。"""
    s_steps, length = attn.shape
    prompt = length - s_steps
    acc = [0.0] * length
    keep = [[False] * length for _ in range(s_steps)]
    run = [0.0] * length
    for s in range(s_steps):
        for j in range(length):
            run[j] += float(attn[s, j])
        for j in range(length):
            acc[j] += float(attn[s, j])
        valid = prompt + s
        for j in range(min(n_sink, valid)):
            keep[s][j] = True
        cand = [j for j in range(n_sink, valid)]
        cand.sort(key=lambda j: (-run[j], j))
        for j in cand[:budget]:
            keep[s][j] = True
    return (torch.tensor(acc, dtype=torch.float32),
            torch.tensor(keep, dtype=torch.bool))


def check_example(solve):
    torch.manual_seed(0)
    attn = torch.zeros(2, 6)
    attn[0, :4] = torch.tensor([0.1, 0.5, 0.2, 0.2])
    attn[1, :5] = torch.tensor([0.05, 0.1, 0.25, 0.3, 0.3])
    acc, keep = solve(attn, 2, 1)
    exp = torch.tensor([0.15, 0.6, 0.45, 0.5, 0.3, 0.0])
    assert torch.allclose(acc, exp, atol=1e-6, rtol=1e-6), f"got {acc.tolist()}"
    assert keep[0].tolist() == [True, True, True, False, False, False], f"got {keep[0].tolist()}"
    assert keep[1].tolist() == [True, True, False, True, False, False], f"got {keep[1].tolist()}"


check_example.example = True


def check_accumulates_without_dividing(solve):
    """跨步累加求和不除 query 数：4 步全 1 → 累计 4（若取平均则为 1）。"""
    torch.manual_seed(1)
    s_steps, length = 4, 6
    attn = torch.zeros(s_steps, length)
    prompt = length - s_steps
    for s in range(s_steps):
        attn[s, :prompt + s] = 1.0
    acc, _ = solve(attn, 2, 0)
    assert torch.allclose(acc[:prompt], torch.full((prompt,), 4.0)), f"got {acc.tolist()}"
    assert float(acc[-1]) == 0.0, "末位 token 从未被注意过，累计必须为 0"


def check_sink_always_kept(solve):
    """sink 位置即使累计分为 0 也必须每步保留。"""
    torch.manual_seed(2)
    attn = torch.zeros(3, 7)
    attn[:, 4:] = 5.0
    _, keep = solve(attn, 1, 2)
    assert keep[:, 0].all() and keep[:, 1].all(), "sink 必须全程保留"
    assert keep[1, 4] and keep[2, 4], "高分位应压过 sink 之后的低分位"


def check_budget_and_ties(solve):
    """并列取小下标、每步保留数 = n_sink + min(budget, 有效列-n_sink)。"""
    torch.manual_seed(3)
    s_steps, length = 3, 8
    attn = torch.zeros(s_steps, length)
    prompt = length - s_steps
    for s in range(s_steps):
        attn[s, :prompt + s] = 1.0
    _, keep = solve(attn, 2, 1)
    for s in range(s_steps):
        valid = prompt + s
        assert int(keep[s].sum()) == 1 + min(2, valid - 1), f"step{s} got {int(keep[s].sum())}"
        assert keep[s, 1] and keep[s, 2], "全并列时应取最小的两个非 sink 下标"


def check_matches_oracle_and_errors(solve):
    """随机用例与逐标量重算对拍，并校验非法参数。"""
    torch.manual_seed(21)
    attn = torch.randn(5, 11).clamp_min(0.0)
    for s in range(5):
        attn[s, 6 + s:] = 0.0
    exp_acc, exp_keep = _oracle(attn, 3, 2)
    acc, keep = solve(attn, 3, 2)
    assert torch.allclose(acc, exp_acc, atol=1e-6, rtol=1e-6), "累计分与重算不符"
    assert torch.equal(keep, exp_keep), "保留掩码与重算不符"
    for bad in ((attn, -1, 0), (attn, 1, 12)):
        try:
            solve(*bad)
            raise AssertionError(f"{bad[1:]} 应抛 ValueError")
        except ValueError:
            pass
