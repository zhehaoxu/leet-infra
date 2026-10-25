import math

import torch

# 题面示例：R=2, g=3, D=3, V=4, tau=0.625（全部取值都是 1/16 的整数倍，float32 精确）
EX_Q = torch.tensor([
    [[[4, 5, 4, 3], [2, 12, 1, 1], [8, 4, 2, 2]],
     [[8, 4, 2, 2], [2, 4, 8, 2], [4, 6, 4, 2]],
     [[1, 2, 13, 0], [5, 5, 4, 2], [4, 4, 4, 4]]],
    [[[5, 4, 4, 3], [4, 5, 4, 3], [2, 4, 9, 1]],
     [[1, 12, 1, 2], [6, 3, 4, 3], [4, 4, 4, 4]],
     [[6, 3, 4, 3], [2, 1, 10, 3], [3, 3, 4, 6]]],
], dtype=torch.float32) / 16.0
EX_P = torch.tensor([
    [[2, 9, 3, 2], [3, 8, 3, 2], [5, 6, 3, 2]],
    [[1, 2, 9, 4], [1, 2, 4, 9], [4, 4, 4, 4]],
], dtype=torch.float32) / 16.0
EX_TAU = 0.625


def _oracle(Q, P, tau, L):
    """独立重算：反向扫层定早退深度、用 argmax(range) 取 token、
    用「执行过的 block 编号清单」长度核算层数、自底向上按 fsum 汇总。"""
    R, g, D, V = Q.shape
    Ql = Q.tolist()
    Pl = P.tolist()
    toks, lays, depths = [], [], []
    for r in range(R):
        if g == 0:
            toks.append(1)
            lays.append(L)
            depths.append([])
            continue
        es = []
        for j in range(g):
            e = D
            for l in reversed(range(D)):          # 反向扫描，反复覆盖 -> 留下最浅的达标层
                if max(Ql[r][j][l]) >= tau:
                    e = l + 1
            es.append(e)
        blocks = []
        for e in es:                              # 草稿阶段：逐位把跑过的 block 编号记下来
            blocks += list(range(1, e + 1))
        shared = min(es)                          # 缓存前缀：所有草稿位共同跑到的最深层
        tail = list(range(shared + 1, L + 1))
        blocks += tail if tail else [L]           # 已在末层早退时，收尾仍要一次 LM head
        drafted = [max(range(V), key=lambda v: Ql[r][j][es[j] - 1][v]) for j in range(g)]
        judged = [max(range(V), key=lambda v: Pl[r][j][v]) for j in range(g)]
        bad = [j for j in range(g) if drafted[j] != judged[j]]
        a = bad[0] if bad else g
        toks.append(a + 1)
        lays.append(len(blocks))
        depths.append(es)
    speedup = math.fsum(toks) * L / math.fsum(lays)
    return (torch.tensor(toks, dtype=torch.long),
            torch.tensor(lays, dtype=torch.long),
            torch.tensor([d if d else [0] * 0 for d in depths], dtype=torch.long).reshape(R, g),
            float(speedup))


def _cmp(solve, Q, P, tau, L):
    got = solve(Q, P, tau, L)
    exp = _oracle(Q, P, tau, L)
    assert torch.equal(got[0], exp[0]), f"toks {got[0].tolist()} != {exp[0].tolist()}"
    assert torch.equal(got[1], exp[1]), f"lays {got[1].tolist()} != {exp[1].tolist()}"
    assert got[2].shape == exp[2].shape, f"depth shape {tuple(got[2].shape)} != {tuple(exp[2].shape)}"
    assert torch.equal(got[2], exp[2]), f"depth {got[2].tolist()} != {exp[2].tolist()}"
    assert math.isclose(got[3], exp[3], rel_tol=1e-9, abs_tol=1e-9), f"{got[3]} != {exp[3]}"
    return got


def check_example(solve):
    toks, lays, depth, sp = _cmp(solve, EX_Q, EX_P, EX_TAU, 5)
    assert toks.tolist() == [3, 2]
    assert lays.tolist() == [10, 10]
    assert depth.tolist() == [[2, 3, 1], [3, 1, 2]]
    assert abs(sp - 1.25) < 1e-9
    # 同一个草稿调度换成 L=3 的浅模型：验证阶段只跑 2 个 block，加速比反而掉到 1 以下
    _, lays3, _, sp3 = _cmp(solve, EX_Q, EX_P, EX_TAU, 3)
    assert lays3.tolist() == [8, 8]
    assert abs(sp3 - 0.9375) < 1e-9
    # g=0：不投机，一次全模型前向出一个 token，speedup 恰为 1
    toks0, lays0, depth0, sp0 = solve(torch.zeros(2, 0, 3, 4), torch.zeros(2, 0, 4), EX_TAU, 6)
    assert toks0.tolist() == [1, 1] and lays0.tolist() == [6, 6]
    assert depth0.shape == (2, 0) and abs(sp0 - 1.0) < 1e-12


check_example.example = True


def _rand_case(seed, R, g, D, V, L):
    torch.manual_seed(seed)
    Q = torch.rand(R, g, D, V) + 0.05
    Q = Q / Q.sum(dim=-1, keepdim=True)
    P = torch.rand(R, g, V) + 0.05
    P = P / P.sum(dim=-1, keepdim=True)
    return Q.float(), P.float()


def check_matches_oracle(solve):
    for seed in range(120):
        R = 1 + seed % 3
        g = 1 + (seed // 3) % 4
        D = 1 + seed % 4
        V = 2 + (seed // 7) % 5
        L = D + (seed // 11) % 4
        Q, P = _rand_case(seed, R, g, D, V, L)
        _cmp(solve, Q, P, [0.0, 0.3, 0.625, 0.9][seed % 4], L)


def check_bounds_and_speedup_identity(solve):
    for seed in range(30):
        Q, P = _rand_case(seed + 500, 3, 4, 3, 6, 5)
        toks, lays, depth, sp = _cmp(solve, Q, P, 0.625, 5)
        assert torch.all((toks >= 1) & (toks <= 5)), toks.tolist()
        assert torch.all((lays >= 4 + 1) & (lays <= 4 * 3 + 5)), lays.tolist()
        assert torch.all(depth >= 1) and torch.all(depth <= 3)
        assert math.isclose(sp, toks.sum().item() * 5 / lays.sum().item(), rel_tol=1e-12,
                            abs_tol=1e-12)


def check_speedup_is_token_weighted_not_mean_of_requests(solve):
    # 请求0 又准又浅（3 token / 5 block），请求1 又不准又深（1 token / 6 block）：
    # batch 口径 16/11 明显低于两个 per-request speedup 的算术平均 1.53333...
    Q = torch.tensor([
        [[[0.875, 0.0625, 0.0625], [0.4, 0.4, 0.2]],
         [[0.875, 0.0625, 0.0625], [0.4, 0.4, 0.2]]],
        [[[0.5, 0.25, 0.25], [0.125, 0.5, 0.375]],
         [[0.5, 0.25, 0.25], [0.125, 0.5, 0.375]]],
    ], dtype=torch.float32)
    P = torch.tensor([
        [[0.875, 0.0625, 0.0625], [0.875, 0.0625, 0.0625]],
        [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
    ], dtype=torch.float32)
    toks, lays, depth, sp = _cmp(solve, Q, P, 0.75, 4)
    assert toks.tolist() == [3, 1] and lays.tolist() == [5, 6]
    assert depth.tolist() == [[1, 1], [2, 2]]
    assert sp == 16 / 11
    assert abs(sp - 1.5333334) > 0.01


def check_tau_only_moves_exit_depth(solve):
    # 抬高阈值只会让早退更深（草稿位与 P 都不变时层数变化只来自调度）
    Q, P = _rand_case(77, 2, 4, 4, 5, 6)
    lo = _cmp(solve, Q, P, 0.05, 6)
    hi = _cmp(solve, Q, P, 0.95, 6)
    assert torch.all(hi[2] >= lo[2]), (lo[2].tolist(), hi[2].tolist())
    assert torch.all(hi[2] <= 4) and torch.all(lo[2] >= 1)


def check_p_does_not_change_schedule(solve):
    # 目标分布只影响产出 token 数，不影响早退深度与执行层数
    Q, P = _rand_case(91, 2, 3, 3, 4, 5)
    a = _cmp(solve, Q, P, 0.625, 5)
    b = _cmp(solve, Q, torch.full_like(P, 0.25), 0.625, 5)
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    assert torch.all((b[0] >= 1) & (b[0] <= 4))


def check_first_block_boundary_and_argmax_ties(solve):
    # max == tau 的边界（>= 才算达标）+ 全相同时 argmax 取最小下标 -> 全部一致命中 bonus
    R, g, D, V, L = 2, 3, 2, 4, 4
    Q = torch.full((R, g, D, V), 0.25)
    P = torch.full((R, g, V), 0.25)
    toks, lays, depth, sp = _cmp(solve, Q, P, 0.25, L)
    assert torch.equal(depth, torch.ones(R, g, dtype=torch.long))
    assert toks.tolist() == [g + 1, g + 1]
    assert lays.tolist() == [g + max(L - 1, 1)] * R
    assert abs(sp - (2 * 4 * 4) / (2 * 6)) < 1e-12


def check_all_rejected_still_pays_the_whole_round(solve):
    # 第一位就不一致：本轮只产出 1 个 token，但 g 次草稿 + 验证的层数一分不少
    R, g, D, V, L = 1, 3, 3, 4, 5
    Q = torch.zeros(R, g, D, V)
    for j in range(g):
        for l in range(D):
            Q[0, j, l, 3] = 0.875
            Q[0, j, l, 0] = 0.125
    P = torch.zeros(R, g, V)
    P[..., 0] = 0.875
    P[..., 1] = 0.125
    toks, lays, depth, sp = _cmp(solve, Q, P, 0.75, L)
    assert toks.tolist() == [1]
    assert depth.tolist() == [[1, 1, 1]]
    assert lays.tolist() == [3 + (L - 1)]
    assert abs(sp - 1 * L / lays[0].item()) < 1e-12


def check_full_depth_exit_leaves_only_lm_head(solve):
    # tau 高于一切 -> 每个草稿位都跑满 D 层；D == L 时验证阶段只剩 1 个 block（LM head）
    Q, P = _rand_case(31, 2, 3, 4, 4, 4)
    toks, lays, depth, sp = _cmp(solve, Q, P, 1.0000001, 4)
    assert torch.equal(depth, torch.full((2, 3), 4, dtype=torch.long))
    assert lays.tolist() == [3 * 4 + 1, 3 * 4 + 1]
    assert sp == 4 * sum(toks.tolist()) / (2 * 13)


def check_single_draft_position(solve):
    # g=1 的最小投机轮：产出 2 或 1 个 token，层数 = e_0 + max(L-e_0,1)
    Q, P = _rand_case(45, 3, 1, 2, 3, 5)
    toks, lays, depth, sp = _cmp(solve, Q, P, 0.5, 5)
    assert toks.numel() == 3
    for r in range(3):
        e = depth[r, 0].item()
        assert lays[r].item() == e + max(5 - e, 1)
        assert toks[r].item() in (1, 2)
