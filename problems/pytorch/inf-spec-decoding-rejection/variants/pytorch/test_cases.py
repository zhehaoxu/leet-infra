import torch


def _oracle(draft_probs, target_probs, draft_tokens, u_accept, u_resample):
    """独立纯 python：CDF 反查用累加循环，接受/拒绝逐元素手算。"""
    V = int(target_probs.shape[1])
    gamma = int(draft_tokens.shape[0])
    dp = draft_probs.tolist()
    tp = target_probs.tolist()
    ua = [float(x) for x in u_accept.tolist()]
    ur = float(u_resample)
    dt = [int(x) for x in draft_tokens.tolist()]

    def smp(probs, u):
        acc = 0.0
        for idx in range(V):
            acc += probs[idx]
            if acc >= u:
                return idx
        return V - 1

    res = []
    all_acc = True
    for i in range(gamma):
        xt = dt[i]
        p = tp[i][xt]
        q = dp[i][xt]
        ratio = p / q if q > 0 else float('inf')
        if ua[i] < min(1.0, ratio):
            res.append(xt)
        else:
            modified = [max(0.0, tp[i][v] - dp[i][v]) for v in range(V)]
            s = sum(modified)
            modified = [m / s for m in modified]
            res.append(smp(modified, ur))
            all_acc = False
            break
    if all_acc:
        res.append(smp(tp[gamma], ur))
    return res


_DP = torch.tensor([[0.5, 0.3, 0.1, 0.1],
                    [0.2, 0.5, 0.2, 0.1],
                    [0.1, 0.1, 0.6, 0.2]])
_TP = torch.tensor([[0.3, 0.4, 0.2, 0.1],
                    [0.4, 0.2, 0.3, 0.1],
                    [0.1, 0.1, 0.6, 0.2],
                    [0.25, 0.25, 0.25, 0.25]])


def check_example(solve):
    out = solve(_DP, _TP, torch.tensor([0, 1, 2]),
                torch.tensor([0.5, 0.7, 0.9]), 0.5)
    assert out.tolist() == [0, 0], out.tolist()


def check_all_accepted_bonus(solve):
    dp = torch.tensor([[0.5, 0.3, 0.1, 0.1],
                       [0.2, 0.5, 0.2, 0.1],
                       [0.1, 0.1, 0.6, 0.2]])
    tp = torch.tensor([[0.6, 0.3, 0.05, 0.05],
                       [0.2, 0.6, 0.1, 0.1],
                       [0.1, 0.1, 0.7, 0.1],
                       [0.1, 0.2, 0.3, 0.4]])
    out = solve(dp, tp, torch.tensor([0, 1, 2]),
                torch.tensor([0.5, 0.1, 0.5]), 0.65)
    assert out.tolist() == [0, 1, 2, 3], out.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(3)
    for _ in range(30):
        gamma = int(torch.randint(1, 5, (1,)))
        V = int(torch.randint(3, 7, (1,)))
        dp = torch.rand(gamma, V)
        dp = dp / dp.sum(-1, keepdim=True)
        tp = torch.rand(gamma + 1, V)
        tp = tp / tp.sum(-1, keepdim=True)
        dt = torch.randint(0, V, (gamma,))
        ua = torch.rand(gamma)
        ur = float(torch.rand(1))
        got = solve(dp, tp, dt, ua, ur)
        exp = _oracle(dp, tp, dt, ua, ur)
        assert got.tolist() == exp, (dt.tolist(), ua.tolist(), ur, got.tolist())


check_example.example = True
