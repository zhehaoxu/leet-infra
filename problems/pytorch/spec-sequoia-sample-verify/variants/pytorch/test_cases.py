import torch


def _oracle(P, Q, k, v_list, r_list, u_final):
    """独立纯 Python：Sequoia Algorithm 2，自带 norm/cdf。"""
    def norm(x):
        s = sum(x)
        if s > 0:
            return [v / s for v in x]
        n = len(x)
        return [1.0 / n] * n

    def cdf(d, u):
        cum = 0.0
        for idx, val in enumerate(d):
            cum += val
            if u < cum:
                return idx
        return len(d) - 1

    R = norm(list(P)); D = norm(list(Q)); S = set(); V = len(R)
    for i in range(k):
        si = cdf(D, v_list[i])
        if D[si] > 0 and r_list[i] < R[si] / D[si]:
            return [si, 0]
        R = norm([max(0.0, R[v] - D[v]) for v in range(V)])
        D[si] = 0.0
        S.add(si)
        if sum(D) == 0:
            D = [0.0 if t in S else 1.0 for t in range(V)]
        D = norm(D)
    return [cdf(R, u_final), 1]


def _mk(x):
    return torch.tensor(x, dtype=torch.float)


def check_example(solve):
    out = solve(_mk([0.1, 0.2, 0.3, 0.4]), _mk([0.4, 0.3, 0.2, 0.1]),
                2, _mk([0.5, 0.9]), _mk([0.9, 0.5]), 0.3)
    assert torch.equal(out, torch.tensor([3, 0])), out


def check_accept_first_branch(solve):
    out = solve(_mk([0.2, 0.4, 0.3, 0.1]), _mk([0.3, 0.3, 0.2, 0.2]),
                2, _mk([0.5, 0.6]), _mk([0.9, 0.1]), 0.3)
    assert torch.equal(out, torch.tensor([1, 0])), out


def check_residual_fallback(solve):
    # k=1 且被拒 -> 从更新后的 R=[0,0,0.25,0.75] 采样回退
    out = solve(_mk([0.1, 0.2, 0.3, 0.4]), _mk([0.4, 0.3, 0.2, 0.1]),
                1, _mk([0.5]), _mk([0.9]), 0.3)
    assert torch.equal(out, torch.tensor([3, 1])), out


def check_matches_oracle(solve):
    torch.manual_seed(3)
    for _ in range(60):
        V = int(torch.randint(2, 7, (1,)))
        raw_p = torch.rand(V) + 0.05
        P = (raw_p / raw_p.sum()).tolist()
        raw_q = torch.rand(V) + 0.05
        Q = (raw_q / raw_q.sum()).tolist()
        k = int(torch.randint(1, V + 1, (1,)))
        v_list = torch.rand(k).tolist()
        r_list = torch.rand(k).tolist()
        u_final = float(torch.rand(1))
        Pt, Qt = _mk(P), _mk(Q)
        out = solve(Pt, Qt, k, _mk(v_list), _mk(r_list), u_final)
        exp = torch.tensor(_oracle(P, Q, k, v_list, r_list, u_final), dtype=torch.long)
        assert torch.equal(out, exp), (P, Q, k, v_list, r_list, u_final, out, exp)


check_example.example = True
