import math

import torch

# 主示例：gamma=2, C=3, V=4（全部取值都是 1/16 的整数倍，逐位精确）
EX_Q = torch.tensor([[0.5, 0.25, 0.125, 0.125],
                     [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
EX_P = torch.tensor([[0.5, 0.125, 0.25, 0.125],
                     [0.125, 0.625, 0.125, 0.125],
                     [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
EX_CAND = torch.tensor([[1, 2, 0], [0, 0, 2]], dtype=torch.long)
EX_U = torch.tensor([[0.75, 0.25, 0.0], [0.75, 0.5, 0.25]], dtype=torch.float32)
EX_U_END = 0.9

# 对照示例：gamma=1, C=3, 同一份数据下 mode 0 / mode 1 的分歧
EX2_Q = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=torch.float32)
EX2_P = torch.tensor([[0.0, 0.3125, 0.4375, 0.25],
                      [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
EX2_CAND = torch.tensor([[0, 1, 2]], dtype=torch.long)
EX2_U = torch.tensor([[0.5, 0.375, 0.9]], dtype=torch.float32)
EX2_U_END = 0.5


def _cdf(row, u):
    """收尾采样：最小 j 使前缀和严格 > u，兜底取最后一个下标。"""
    s = 0.0
    for j, v in enumerate(row):
        s += v
        if s > u:
            return j
    return len(row) - 1


def _oracle(P, Q, cand, u, u_end, no_replace):
    """独立重算：while/else 控制流 + 列表推导 + 标志位，逐元素 zip 求残差。"""
    Pl, Ql = P.tolist(), Q.tolist()
    Cl, Ul = cand.tolist(), u.tolist()
    g = len(Ql)
    C = len(Cl[0]) if g else 0
    out, picks, trials = [], [], 0
    j = 0
    aborted = False
    while j < g and not aborted:
        p = Pl[j][:]
        q = Ql[j][:]
        i = 0
        chosen = None
        while i < C:
            x = Cl[j][i]
            trials += 1
            px, qx = p[x], q[x]
            if qx == 0.0:
                prob = 1.0 if px > 0.0 else 0.0      # q=0：有目标质量则必收；0/0 视作 0
            else:
                prob = min(1.0, px / qx)
            if Ul[j][i] <= prob:
                chosen = x
                break
            raw = [max(a - b, 0.0) for a, b in zip(p, q)]
            s = sum(raw)
            if s > 0.0:
                p = [v / s for v in raw]
            if no_replace:
                q = [0.0 if k == x else v for k, v in enumerate(q)]
                s2 = sum(q)
                if s2 > 0.0:
                    q = [v / s2 for v in q]
            i += 1
        if chosen is None:
            picks.append(-1)
            out.append(_cdf(p, u_end))
            aborted = True
        else:
            picks.append(i)
            out.append(chosen)
            j += 1
    if not aborted:
        out.append(_cdf(Pl[g], u_end))
    return (torch.tensor(out, dtype=torch.long),
            torch.tensor(picks, dtype=torch.long),
            int(trials))


def _cmp(solve, P, Q, cand, u, u_end, no_replace):
    got = solve(P, Q, cand, u, u_end, no_replace)
    exp = _oracle(P, Q, cand, u, u_end, no_replace)
    assert torch.equal(got[0], exp[0]), f"out {got[0].tolist()} != {exp[0].tolist()}"
    assert torch.equal(got[1], exp[1]), f"picks {got[1].tolist()} != {exp[1].tolist()}"
    assert got[2] == exp[2], f"trials {got[2]} != {exp[2]}"
    n_ok = int((got[1] >= 0).sum())
    assert got[0].numel() == n_ok + 1, f"len(out)={got[0].numel()} n_ok={n_ok}"
    if got[1].numel() and int(got[1][-1]) == -1:
        assert got[0].numel() == got[1].numel()
    else:
        assert got[0].numel() == got[1].numel() + 1
    return got


def _comp(total, parts, gen):
    """把 total 拆成 parts 个非负整数之和（允许 0，保证行和恰为 1）。"""
    if parts == 1:
        return [total]
    cuts = sorted(gen.randint(1, total, (parts - 1,)).tolist())
    bounds = [0] + cuts + [total]
    return [bounds[k + 1] - bounds[k] for k in range(parts)]


def _rand_case(seed, g, C, V):
    torch.manual_seed(seed)
    Qr, Pr, Cr, Ur = [], [], [], []
    for _ in range(g):
        Qr.append([c / 16.0 for c in _comp(16, V, torch)])
        Pr.append([c / 16.0 for c in _comp(16, V, torch)])
        Cr.append(torch.randint(0, V, (C,)).tolist())
        Ur.append([k / 16.0 for k in torch.randint(0, 16, (C,)).tolist()])
    bonus = [c / 16.0 for c in _comp(16, V, torch)]
    P = torch.tensor(Pr + [bonus], dtype=torch.float32)
    Q = torch.tensor(Qr, dtype=torch.float32) if g else torch.zeros(0, V, dtype=torch.float32)
    cand = torch.tensor(Cr, dtype=torch.long) if g else torch.zeros(0, C, dtype=torch.long)
    u = torch.tensor(Ur, dtype=torch.float32) if g else torch.zeros(0, C, dtype=torch.float32)
    return P, Q, cand, u, (torch.randint(0, 16, (1,)).item() / 16.0)


def check_example(solve):
    out, picks, trials = _cmp(solve, EX_P, EX_Q, EX_CAND, EX_U, EX_U_END, 0)
    assert out.tolist() == [2, 1], out.tolist()
    assert picks.tolist() == [1, -1], picks.tolist()
    assert trials == 5
    # gamma=1 的对照：mode 0 收第 1 个候选，mode 1 因为 draft 被重归一化而改收第 2 个
    o0 = _cmp(solve, EX2_P, EX2_Q, EX2_CAND, EX2_U, EX2_U_END, 0)
    assert o0[0].tolist() == [1, 2] and o0[1].tolist() == [1] and o0[2] == 2
    o1 = _cmp(solve, EX2_P, EX2_Q, EX2_CAND, EX2_U, EX2_U_END, 1)
    assert o1[0].tolist() == [2, 2] and o1[1].tolist() == [2] and o1[2] == 3
    # 把 u[0][1] 抬到 0.5（正好踩在阈值上仍接受）与 0.5625（越过阈值 -> 走到候选 2）
    u_edge = EX2_U.clone()
    u_edge[0, 1] = 0.5
    assert _cmp(solve, EX2_P, EX2_Q, EX2_CAND, u_edge, EX2_U_END, 0)[0].tolist() == [1, 2]
    u_edge[0, 1] = 0.5625
    assert _cmp(solve, EX2_P, EX2_Q, EX2_CAND, u_edge, EX2_U_END, 0)[0].tolist() == [2, 2]


check_example.example = True


def check_matches_oracle(solve):
    for seed in range(260):
        g = seed % 5
        C = 1 + (seed // 5) % 4
        V = 2 + (seed // 17) % 5
        P, Q, cand, u, u_end = _rand_case(1000 + seed, g, C, V)
        _cmp(solve, P, Q, cand, u, u_end, seed % 2)


def check_c1_matches_textbook_rejection_sampling(solve):
    # C=1 时应当塌回教科书式推测采样：u <= min(1,p/q) 则收，
    # 否则从 norm(max(p-q,0)) 收尾——这里用 torch 张量算子写成第三条独立路径。
    for seed in range(60):
        g = 4
        P, Q, cand, u, u_end = _rand_case(5000 + seed, g, 1, 5)
        got = _cmp(solve, P, Q, cand, u, u_end, seed % 2)
        out_ref, picks_ref, done = [], [], False
        for j in range(g):
            x = int(cand[j, 0])
            p = P[j].clone()
            q = Q[j]
            px, qx = float(p[x]), float(q[x])
            ratio = float("inf") if qx == 0.0 else px / qx
            if float(u[j, 0]) <= min(1.0, ratio):
                out_ref.append(x)
                picks_ref.append(0)
            else:
                res = torch.clamp(p - q, min=0.0)
                s = float(res.sum())
                if s > 0.0:
                    p = res / s
                hit = torch.nonzero(torch.cumsum(p, dim=0) > u_end)
                out_ref.append(int(hit[0].item()) if hit.numel() else 5 - 1)
                picks_ref.append(-1)
                done = True
                break
        if not done:
            hit = torch.nonzero(torch.cumsum(P[g], dim=0) > u_end)
            out_ref.append(int(hit[0].item()) if hit.numel() else 5 - 1)
        assert got[0].tolist() == out_ref, (got[0].tolist(), out_ref)
        assert got[1].tolist() == picks_ref, (got[1].tolist(), picks_ref)
        assert got[2] == len(picks_ref)


def check_duplicate_candidate_is_dead(solve):
    # 论文关键观察：某候选一旦被拒，它在残差 p 里的质量就归零，重复采样救不回来
    cand = torch.tensor([[0, 0, 2]], dtype=torch.long)
    out, picks, trials = _cmp(solve, EX2_P, EX2_Q, cand,
                              torch.tensor([[0.5, 0.0625, 0.9]], dtype=torch.float32), 0.5, 0)
    assert picks.tolist() == [2] and out.tolist() == [2, 2] and trials == 3
    # 但 u == 0 恰好踩在 <= 0 的边界上，重复候选仍会被收下
    out2, picks2, t2 = _cmp(solve, EX2_P, EX2_Q, cand,
                            torch.tensor([[0.5, 0.0, 0.9]], dtype=torch.float32), 0.5, 0)
    assert picks2.tolist() == [1] and out2.tolist() == [0, 2] and t2 == 2


def check_first_come_first_served_and_zero_draft(solve):
    Q = torch.tensor([[0.25, 0.25, 0.25, 0.25]] * 3, dtype=torch.float32)
    P = torch.tensor([[0.1, 0.7, 0.1, 0.1],
                      [0.1, 0.1, 0.7, 0.1],
                      [0.1, 0.1, 0.1, 0.7],
                      [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    cand = torch.tensor([[3, 0, 1], [0, 1, 2], [1, 2, 0]], dtype=torch.long)
    # 每个位置第一个候选的 ratio 都是 0.1/0.25=0.4，硬币必须严格踩在 0.4 之下才「先到先得」；
    # 0.375 = 6/16 既在 1/16 格点上又留出比较余量，避免浮点边界歧义。
    u = torch.tensor([[0.0625, 0.0, 0.0], [0.0, 0.5, 0.5], [0.375, 0.5, 0.5]], dtype=torch.float32)
    out, picks, trials = _cmp(solve, P, Q, cand, u, 0.8125, 0)
    assert picks.tolist() == [0, 0, 0] and trials == 3
    assert out.tolist() == [3, 0, 1, 3]          # 末位是 bonus = cdf(P[3], 0.8125) = 3
    # gamma=0：一次普通的全模型解码，没有任何验证尝试
    P0 = torch.tensor([[0.125, 0.375, 0.5]], dtype=torch.float32)
    Q0 = torch.zeros(0, 3, dtype=torch.float32)
    o0, p0, t0 = _cmp(solve, P0, Q0, torch.zeros(0, 2, dtype=torch.long),
                      torch.zeros(0, 2, dtype=torch.float32), 0.4375, 0)
    assert o0.tolist() == [1] and p0.numel() == 0 and t0 == 0


def check_zero_q_and_zero_p_candidates(solve):
    # q=0 且 p>0：ratio=+inf，直接接受；q=0 且 p=0：ratio 记 0，只有 u==0 才接受
    Q = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
    P = torch.tensor([[0.25, 0.25, 0.5, 0.0],
                      [0.5, 0.25, 0.25, 0.0]], dtype=torch.float32)
    out_a, picks_a, tr_a = _cmp(solve, P, Q, torch.tensor([[2, 0]], dtype=torch.long),
                                torch.tensor([[0.5, 0.5]], dtype=torch.float32), 0.5, 0)
    assert picks_a.tolist() == [0] and out_a.tolist() == [2, 1] and tr_a == 1
    out_b, picks_b, tr_b = _cmp(solve, P, Q, torch.tensor([[3, 1]], dtype=torch.long),
                                torch.tensor([[0.5, 0.0]], dtype=torch.float32), 0.5, 1)
    assert picks_b.tolist() == [1] and out_b.tolist() == [1, 1] and tr_b == 2


def check_no_replace_equals_replace_when_single_candidate(solve):
    # C=1 时两种算法只用到最初那份 q（重归一化发生在拒绝之后、再没有候选可测），结果必然一致
    for seed in range(40):
        P, Q, cand, u, u_end = _rand_case(9000 + seed, 3, 1, 4)
        a = _cmp(solve, P, Q, cand, u, u_end, 0)
        b = _cmp(solve, P, Q, cand, u, u_end, 1)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


def check_length_and_trial_bounds(solve):
    for seed in range(80):
        g, C = 1 + seed % 4, 1 + (seed // 4) % 3
        P, Q, cand, u, u_end = _rand_case(3000 + seed, g, C, 4)
        for mode in (0, 1):
            out, picks, trials = _cmp(solve, P, Q, cand, u, u_end, mode)
            assert 1 <= out.numel() <= g + 1
            assert 0 <= trials <= g * C
            assert picks.numel() in (out.numel(), out.numel() - 1)
            assert torch.all(out >= 0) and torch.all(out < 4)
            assert torch.all((picks >= -1) & (picks < C))


def check_all_zero_coins_accept_every_first_candidate(solve):
    # u 全为 0：0 <= 任何 ratio 恒成立，于是每步第 0 个候选都被接受，残差从未被更新
    g, C, V = 4, 3, 4
    P, Q, cand, u, u_end = _rand_case(2024, g, C, V)
    out, picks, trials = _cmp(solve, P, Q, cand, torch.zeros(g, C), u_end, 1)
    assert picks.tolist() == [0] * g and trials == g
    assert out.numel() == g + 1
    assert out[:-1].tolist() == [int(cand[j, 0]) for j in range(g)]


def check_all_coins_reject_and_empty_candidate_rows(solve):
    # 接受率为 0 的极端：候选 token 在 p、q 上同时为 0（0/0 记 ratio=0），硬币又都大于 0，
    # 于是位置 0 当场候选耗尽；残差全零 -> p 保持原样，末位 token 直接从原 p 反查。
    Q = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
    P = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    cand = torch.tensor([[2, 3, 2]], dtype=torch.long)      # 重复候选 + 0/0 候选
    u = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    for mode in (0, 1):
        # u_end 恰等于首个前缀和 0.5：严格 > 约定必须跳到下标 1（踩线不算命中）
        for u_end, exp in ((0.4375, 0), (0.5, 1)):
            out, picks, trials = _cmp(solve, P, Q, cand, u, u_end, mode)
            assert trials == 3 and picks.tolist() == [-1], (mode, u_end, out, picks, trials)
            assert out.tolist() == [exp], (mode, u_end, out.tolist())
    # C=0（候选行是空的）：位置 0 无须任何 trial 即耗尽，picks 只有一个 -1
    Pc = torch.tensor([[0.125, 0.375, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0],
                       [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    Qc = torch.tensor([[0.25, 0.25, 0.25, 0.25]] * 2, dtype=torch.float32)
    out0, picks0, trials0 = _cmp(solve, Pc, Qc, torch.zeros(2, 0, dtype=torch.long),
                                 torch.zeros(2, 0, dtype=torch.float32), 0.4375, 1)
    assert out0.tolist() == [1] and picks0.tolist() == [-1] and trials0 == 0


def check_terminal_cdf_is_measure_preserving(solve):
    # g=0 时 solve 退化成纯粹的 CDF 反查采样器：把 u_end 铺满 512 个等分格点（格点取区间中点，
    # 避开 <= / >= 的边界歧义），经验频率必须逐点等于目标分布。浮点比较一律走 allclose。
    V, K = 4, 512
    row = [0.25, 0.5, 0.125, 0.125]
    P = torch.tensor([row] + [[0.25] * V], dtype=torch.float32)
    Q = torch.zeros(0, V, dtype=torch.float32)
    cand = torch.zeros(0, 3, dtype=torch.long)
    u = torch.zeros(0, 3, dtype=torch.float32)
    counts = torch.zeros(V, dtype=torch.float64)
    for i in range(K):
        out = solve(P, Q, cand, u, (i + 0.5) / K, 0)[0]
        assert out.dtype == torch.long and out.numel() == 1, out
        counts[int(out[0])] += 1.0
    emp = (counts / K).to(torch.float32)
    assert torch.allclose(emp, torch.tensor(row, dtype=torch.float32), atol=1e-5, rtol=1e-5), \
        emp.tolist()


def _single_draft_155(P, Q, cand, u, u_end):
    """邻题 inf-spec-decoding-rejection(order 155) 的原始口径，逐字照搬：
    接受用**严格** `u < min(1, p/q)`，收尾采样用 `>=`（最小 k 使 cumsum >= u），
    残差无条件归一化。C=1 且硬币不踩线时两题必须给出同一个 token 序列。"""
    g = int(Q.shape[0])
    out = []
    for j in range(g):
        x = int(cand[j, 0])
        px, qx = float(P[j][x]), float(Q[j][x])
        ratio = px / qx if qx > 0 else float("inf")
        if float(u[j, 0]) < min(1.0, ratio):
            out.append(x)
        else:
            mod = torch.clamp(P[j] - Q[j], min=0.0)
            mod = mod / float(mod.sum())
            out.append(int((torch.cumsum(mod, dim=0) < u_end).sum().item()))
            return torch.tensor(out, dtype=torch.long)
    out.append(int((torch.cumsum(P[g], dim=0) < u_end).sum().item()))
    return torch.tensor(out, dtype=torch.long)


def _tie_free(P, Q, cand, u, u_end):
    """本轮数据是否避开了两题的三处口径差异：<= vs <、> vs >=、0/0 的 ratio、零残差归一化。"""
    g = int(Q.shape[0])
    for j in range(g):
        x = int(cand[j, 0])
        px, qx = float(P[j][x]), float(Q[j][x])
        if qx <= 0.0 or px <= 0.0:                     # 0/0 与 p=0 时两题的 ratio 约定不同
            return False
        if abs(float(u[j, 0]) - min(1.0, px / qx)) < 1e-6:
            return False
        res = torch.clamp(P[j] - Q[j], min=0.0)
        if float(res.sum()) < 1e-6:                    # 155 在这里会除以 0
            return False
        for row in (res / res.sum(), P[g]):
            if any(abs(float(v) - float(u_end)) < 1e-6 for v in torch.cumsum(row, dim=0).tolist()):
                return False
    return True


def check_c1_agrees_with_single_draft_rejection_155(solve):
    # 与单草稿拒绝采样对拍：C=1 时本题就是 155，连续均匀数几乎不可能踩线，
    # 逐位跳过仍可能踩线的轮次，其余轮次的输出序列必须与 155 的口径逐元素相等。
    used = 0
    for seed in range(240):
        torch.manual_seed(7000 + seed)
        g, V = 1 + seed % 4, 4 + seed % 3
        P = torch.rand(g + 1, V, dtype=torch.float32)
        P = P / P.sum(-1, keepdim=True)
        Q = torch.rand(g, V, dtype=torch.float32)
        Q = Q / Q.sum(-1, keepdim=True)
        cand = torch.randint(0, V, (g, 1), dtype=torch.long)
        u = torch.rand(g, 1, dtype=torch.float32)
        u_end = float(torch.rand(1).item())
        if not _tie_free(P, Q, cand, u, u_end):
            continue
        used += 1
        for mode in (0, 1):
            out, picks, trials = solve(P, Q, cand, u, u_end, mode)
            exp = _single_draft_155(P, Q, cand, u, u_end)
            assert torch.equal(out, exp), (seed, mode, out.tolist(), exp.tolist())
            assert trials == len(picks) and torch.all(picks == 0) or picks[-1] == -1
    assert used >= 20, f"踩线过滤后只剩 {used} 轮，对拍不成立"

