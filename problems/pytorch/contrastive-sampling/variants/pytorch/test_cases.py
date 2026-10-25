import math

import torch


def _oracle(logits, sim, k, lam):
    """独立重算：纯 Python 手工 softmax/log + sorted + 逐条比较，不用任何张量排序。"""
    lg = [float(x) for x in logits.tolist()]
    v = len(lg)
    mx = max(lg)
    ex = [math.exp(x - mx) for x in lg]
    tot = math.fsum(ex)
    p = [x / tot for x in ex]
    lp = [x - mx - math.log(tot) for x in lg]      # 全词表 log_softmax（不做核内归一）

    k = min(int(k), v)
    cand = sorted(range(v), key=lambda i: (-p[i], i))[:k]
    t = sim.shape[1] if sim.dim() > 1 else 0
    lam = float(lam)
    deg = []
    for pos in range(k):                     # sim 的第 pos 行 = 第 pos 个候选（降序）
        if t > 0:
            deg.append(max(min(1.0, max(0.0, (float(sim[pos][i]) + 1.0) / 2.0))
                           for i in range(t)))
        else:
            deg.append(0.0)
    scores = [(1.0 - lam) * lp[cand[pos]] - lam * deg[pos] for pos in range(k)]
    pos, best = 0, scores[0]
    for j in range(1, k):
        if scores[j] > best:
            pos, best = j, scores[j]
    return (torch.tensor(cand, dtype=torch.long), torch.tensor(deg),
            torch.tensor(scores), torch.tensor(cand[pos], dtype=torch.long))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([1.5, 1.2, 0.5, -1.0])
    sim = torch.tensor([[0.95, 0.90], [-0.80, -0.60], [0.00, 0.20]])
    cand, deg, scores, chosen = solve(logits, sim, 3, 0.6)
    assert torch.equal(cand, torch.tensor([0, 1, 2])), cand.tolist()
    assert torch.allclose(deg, torch.tensor([0.975, 0.2, 0.6]), atol=1e-6), deg.tolist()
    assert torch.allclose(scores, torch.tensor([-0.898704, -0.553703, -1.073704]),
                          atol=1e-5, rtol=1e-5), scores.tolist()
    assert chosen.item() == 1, chosen.item()          # 退化惩罚把选择从 0 翻到 1


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(60):
        v = torch.randint(3, 13, ()).item()
        logits = torch.randn(v) * 2
        k = torch.randint(1, v + 2, ()).item()        # 允许 k > V（应被夹到 V）
        t = torch.randint(0, 5, ()).item()
        sim = torch.empty(k, t).uniform_(-1, 1) if t else torch.zeros(k, 0)
        for lam in (0.0, 0.35, 0.6, 1.0):
            got = solve(logits, sim, k, lam)
            exp = _oracle(logits, sim, k, lam)
            assert torch.equal(got[0], exp[0]), (v, k, t, lam)
            assert torch.allclose(got[1], exp[1], atol=1e-6), (v, k, t, lam)
            assert torch.allclose(got[2], exp[2], atol=1e-5, rtol=1e-5), (v, k, t, lam)
            assert torch.equal(got[3], exp[3]), (v, k, t, lam)


def check_lam_zero_is_greedy(solve):
    # λ=0 → 只看 logprob，等价于在 top-k 里取最 probable 的（即全局 argmax）
    logits = torch.tensor([1.5, 1.2, 0.5, -1.0])
    sim = torch.tensor([[0.95, 0.90], [-0.80, -0.60], [0.00, 0.20]])
    cand, _, scores, chosen = solve(logits, sim, 3, 0.0)
    assert chosen.item() == int(torch.argmax(logits)) == cand[0].item()
    logp = torch.log_softmax(logits, dim=-1)[cand]
    assert torch.allclose(scores, logp, atol=1e-6), scores.tolist()


def check_no_history_deg_is_zero(solve):
    # 无历史（T=0）→ 退化惩罚恒 0，任意 λ 都不改变贪心结果
    logits = torch.tensor([0.2, 1.9, -0.4, 0.7])
    sim = torch.zeros(4, 0)
    for lam in (0.0, 0.5, 1.0):
        _, deg, scores, chosen = solve(logits, sim, 4, lam)
        assert deg.abs().max().item() == 0.0
        assert chosen.item() == 1, (lam, chosen.item())
        assert torch.allclose(scores, (1 - lam) * torch.log_softmax(logits, -1)[[1, 3, 0, 2]],
                              atol=1e-6), scores.tolist()


def check_lam_one_picks_least_redundant(solve):
    # λ=1 → 分数 = -deg，选「最不退化」的候选；候选集合本身不变
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    sim = torch.tensor([[1.0, 0.8], [-1.0, -0.5], [0.0, 0.4], [-0.2, -0.9]])
    cand, deg, scores, chosen = solve(logits, sim, 4, 1.0)
    assert deg.tolist()[0] == 1.0 and abs(deg[1].item() - 0.25) < 1e-6, deg.tolist()
    assert torch.allclose(deg, torch.tensor([1.0, 0.25, 0.7, 0.4]), atol=1e-6), deg.tolist()
    assert torch.allclose(scores, -deg, atol=1e-6)
    assert chosen.item() == int(torch.argmin(deg)) == cand[1].item()


def check_clamps_and_ties(solve):
    # k > V 夹到 V；打分并列时取候选顺序里更靠前者（= logprob 更大者）
    logits = torch.tensor([1.0, 1.0, 0.0])
    sim = torch.zeros(3, 2)                  # 行数 = 实际候选数 min(k, V) = 3
    cand, _, _, chosen = solve(logits, sim, 99, 0.5)
    assert torch.equal(cand, torch.tensor([0, 1, 2])), cand.tolist()
    assert chosen.item() == 0, chosen.item()
    _, _, _, c2 = solve(logits, sim, 99, 0.5)
    assert c2.item() == chosen.item()

check_example.example = True
