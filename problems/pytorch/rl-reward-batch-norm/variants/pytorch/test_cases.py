import math

import torch


def _oracle(reward_batches, group_ids, stat_source, momentum, whiten_std, eps):
    """独立重算，结构与参考解完全不同：
    - 纯 Python list + math.fsum 求和（不向量化、不用 index_add_）；
    - 组统计用 dict 按组号收集成员再逐组两遍求均值/总体方差；
    - 账本用普通 float 逐步递推；
    - rms 用 sqrt(fsum(v*v)/n) 现算。"""
    x = [[float(v) for v in row] for row in reward_batches.tolist()]
    g = [int(v) for v in group_ids.reshape(-1).tolist()]
    t_n = len(x)
    b_n = len(x[0]) if t_n else 0
    src = int(stat_source)
    m = float(momentum)
    whiten = int(whiten_std) == 1
    e = float(eps)

    def mean(vals):
        return math.fsum(vals) / len(vals)

    def pvar(vals):
        mu = mean(vals)
        return math.fsum((v - mu) ** 2 for v in vals) / len(vals)

    norm = []
    rms = []
    run_mu = 0.0
    run_var = 1.0
    led_mu = [0.0]
    led_var = [1.0]
    for t in range(t_n):
        row = x[t]
        b_mu = mean(row)
        b_var = pvar(row)
        if src == 1:
            mu = [run_mu] * b_n
            var = [run_var] * b_n
        elif src == 2:
            buckets = {}
            for j in range(b_n):
                buckets.setdefault(g[j], []).append(row[j])
            gm = {k: mean(v) for k, v in buckets.items()}
            gv = {k: pvar(v) for k, v in buckets.items()}
            mu = [gm[g[j]] for j in range(b_n)]
            var = [gv[g[j]] for j in range(b_n)]
        else:
            mu = [b_mu] * b_n
            var = [b_var] * b_n

        out_row = []
        for j in range(b_n):
            z = row[j] - mu[j]
            out_row.append(z / math.sqrt(var[j] + e) if whiten else z)
        norm.append(out_row)
        rms.append(math.sqrt(math.fsum(v * v for v in out_row) / b_n) if b_n else 0.0)

        run_mu = (1.0 - m) * run_mu + m * b_mu
        run_var = (1.0 - m) * run_var + m * b_var
        led_mu.append(run_mu)
        led_var.append(run_var)

    def tf(vals, shape):
        return torch.tensor(vals, dtype=torch.float32).reshape(shape)

    return (tf([v for r in norm for v in r], (t_n, b_n)),
            tf(rms, (t_n,)), tf(led_mu, (t_n + 1,)), tf(led_var, (t_n + 1,)))


def _close(a, b, name):
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (name, a.tolist(), b.tolist())


# ---- 样例用例 --------------------------------------------------------------------
def check_example(solve):
    rb = torch.tensor([[1.0, 2, 3, 4], [2.0, 3, 4, 5], [10.0, 20, 30, 40]])
    gid = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    n0, rms0, rm0, rv0 = solve(rb, gid, 0, 0.5, 1, 1e-5)
    _close(n0[0], torch.tensor([-1.341635, -0.447212, 0.447212, 1.341635]), "src0.norm[0]")
    _close(n0[2], torch.tensor([-1.341641, -0.447214, 0.447214, 1.341641]), "src0.norm[2]")
    _close(rms0, torch.tensor([0.999996, 0.999996, 1.0]), "src0.rms")
    _close(rm0, torch.tensor([0.0, 1.25, 2.375, 13.6875]), "src0.running_mean")
    _close(rv0, torch.tensor([1.0, 1.125, 1.1875, 63.09375]), "src0.running_var")
    assert n0.shape == (3, 4) and rm0.shape == (4,) and n0.dtype == torch.float32

    n1, rms1, rm1, rv1 = solve(rb, gid, 1, 0.5, 1, 1e-5)
    _close(n1[0], torch.tensor([0.999995, 1.99999, 2.999985, 3.99998]), "src1.norm[0]")
    _close(n1[2], torch.tensor([6.99715, 16.17374, 25.350332, 34.526924]), "src1.norm[2]")
    _close(rms1, torch.tensor([2.738599, 2.368768, 23.15868]), "src1.rms")
    _close(rm1, rm0, "src1.running_mean")          # 账本与 stat_source 无关
    _close(rv1, rv0, "src1.running_var")

    n2, rms2, rm2, rv2 = solve(rb, gid, 2, 0.5, 1, 1e-5)
    _close(n2[0], torch.tensor([-0.99998, 0.99998, -0.99998, 0.99998]), "src2.norm[0]")
    _close(rms2[:2], torch.tensor([0.99998, 0.99998]), "src2.rms")
    _close(rm2, rm0, "src2.running_mean")


check_example.example = True


# ---- 退化探针 --------------------------------------------------------------------
def check_momentum_zero_freezes_ledger(solve):
    """m=0：账本恒为初值 (0,1)；src=1 时归一化几乎不改尺度。"""
    rb = torch.tensor([[3.0, 5.0, 9.0, 33.0], [100.0, 200.0, 300.0, 400.0]])
    gid = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    n, rms, rm, rv = solve(rb, gid, 1, 0.0, 1, 1e-5)
    assert torch.equal(rm, torch.zeros(3)), rm
    assert torch.equal(rv, torch.ones(3)), rv
    scale = 1.0 / math.sqrt(1.0 + 1e-5)
    _close(n, rb * scale, "frozen")
    _close(rms, torch.tensor([17.349264, 273.859924]), "frozen.rms")   # 探针爆表


def check_momentum_one_ledger_is_previous_batch_stats(solve):
    """m=1：账本第 t+1 项恰是第 t 批的批均值/批总体方差（无历史拖尾）。"""
    rb = torch.tensor([[1.0, 2, 3, 4], [10.0, 20, 30, 40], [-5.0, 0.0, 5.0, 100.0]])
    gid = torch.zeros(4, dtype=torch.long)
    _n, _r, rm, rv = solve(rb, gid, 0, 1.0, 1, 1e-5)
    assert torch.equal(rm, torch.tensor([0.0, 2.5, 25.0, 25.0])), rm
    assert torch.allclose(rv, torch.tensor([1.0, 1.25, 125.0, 1887.5]), rtol=1e-5), rv


def check_no_whiten_is_centering_only(solve):
    """whiten_std=0 + src=0：每批均值恰为 0，且不做任何缩放。"""
    rb = torch.tensor([[10.0, 20.0, 30.0, 1000.0], [2.0, 4.0, 6.0, 8.0]])
    gid = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    n, rms, _rm, _rv = solve(rb, gid, 0, 0.3, 0, 1e-5)
    assert torch.allclose(n.mean(dim=1), torch.zeros(2), atol=1e-4), n
    _close(n[0], torch.tensor([-255.0, -245.0, -235.0, 735.0]), "centered")
    _close(n[1], torch.tensor([-3.0, -1.0, 1.0, 3.0]), "centered-2")
    _close(rms, torch.tensor([424.411346, 2.236068]), "centered.rms")   # 未白化 ⇒ 原始尺度


def check_flat_rewards_give_zeros(solve):
    """批内全相等：方差为 0、分子为 0 ⇒ 输出恰为 0（eps>0 保证不除零）。"""
    rb = torch.tensor([[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]])
    gid = torch.tensor([0, 0, 0], dtype=torch.long)
    n, rms, rm, rv = solve(rb, gid, 0, 0.5, 1, 1e-5)
    assert torch.equal(n, torch.zeros(2, 3)), n
    assert torch.equal(rms, torch.zeros(2)), rms
    _close(rv, torch.tensor([1.0, 0.5, 0.25]), "flat.var")     # 方差被吸向 0
    _close(rm, torch.tensor([0.0, 3.5, 5.25]), "flat.mean")    # 均值仍正常追踪


def check_group_of_one_and_equal_group(solve):
    """组大小为 1 或组内全相等 ⇒ 分子与方差同时为 0 ⇒ 该项恰为 0（不是 ±1）。"""
    rb = torch.tensor([[4.0, 9.0, 11.0, 42.0]])
    gid = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    n, rms, _rm, _rv = solve(rb, gid, 2, 0.5, 1, 1e-5)
    _close(n[0], torch.tensor([0.0, -0.999995, 0.999995, 0.0]), "mixed-groups")
    _close(rms, torch.tensor([0.707103]), "mixed-groups.rms")


def check_single_group_equals_batch_source(solve):
    """所有 group_id 相同 ⇒ 组统计退化为批全局统计，src=2 与 src=0 完全一致。"""
    rb = torch.tensor([
        [1.0, 2.5, -3.0, 7.0, 0.5, 0.5, 12.0, -1.25],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        [-1.0, 3.0, -2.0, 8.0, 5.0, 5.0, 5.0, 0.0],
        [2.0, 2.0, 3.0, 3.0, 5.0, 8.0, 13.0, 21.0],
    ], dtype=torch.float32)
    gid = torch.zeros(8, dtype=torch.long)
    a = solve(rb, gid, 0, 0.2, 1, 1e-5)
    b = solve(rb, gid, 2, 0.2, 1, 1e-5)
    for i, name in enumerate(["norm", "rms", "mean", "var"]):
        _close(a[i], b[i], "single-group." + name)


def check_first_batch_with_ledger_is_noop(solve):
    """T=1、src=1：账本还没被任何批更新过 ⇒ 只做 (x-0)/sqrt(1+eps) 的线性缩放。"""
    rb = torch.tensor([[2.0, 6.0, 10.0]])
    gid = torch.tensor([0, 1, 0], dtype=torch.long)
    n, rms, rm, rv = solve(rb, gid, 1, 0.5, 1, 1e-4)
    _close(n, rb / math.sqrt(1.0 + 1e-4), "first-batch")
    _close(rms, torch.tensor([6.830959]), "first-batch.rms")
    _close(rm, torch.tensor([0.0, 3.0]), "ledger")
    _close(rv, torch.tensor([1.0, 5.833333]), "ledger-var")


# ---- 规模用例：与独立 oracle 对拍 -------------------------------------------------
def check_matches_oracle_battery(solve):
    base = torch.tensor([
        [1.0, 2, 3, 4, 5, 6],
        [0.5, -1.5, 3.0, 7.25, 2.0, 2.0],
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        [-7.0, -7.0, -7.0, -7.0, -7.0, -7.0],
        [1.0, 1.5, 0.25, 9.0, 9.0, 3.5],
        [12.0, 0.0, 0.0, 4.0, 8.0, 16.0],
    ], dtype=torch.float32)
    batteries = [
        (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long), 0, 0.5, 1, 1e-5),
        (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long), 1, 0.5, 1, 1e-5),
        (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long), 2, 0.5, 1, 1e-5),
        (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long), 2, 0.9, 0, 1e-3),
        (torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long), 2, 0.1, 1, 1e-5),
        (torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long), 2, 0.0, 1, 1e-5),
        (torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long), 2, 1.0, 1, 1e-6),
        (torch.tensor([3, 1, 3, 1, 3, 1], dtype=torch.long), 0, 0.35, 1, 1e-4),
        (torch.tensor([3, 1, 3, 1, 3, 1], dtype=torch.long), 1, 0.35, 0, 1e-4),
    ]
    for t in (1, 2, 3, 4, 5, 6):
        rb = base[:t]
        for gid, src, mom, wh, eps in batteries:
            args = (rb, gid, src, mom, wh, eps)
            got = solve(*args)
            ref = _oracle(*args)
            for i, name in enumerate(["normalized", "out_rms", "running_mean", "running_var"]):
                _close(got[i], ref[i], "battery.%s.%s" % (name, (t, src, mom, wh)))


def check_scale_mismatch_probe_fires(solve):
    """奖励尺度中途跳变：src=0 的 rms 恒 ≈1，src=1 的 rms 明显偏离 1（事故可观测）。"""
    rb = torch.tensor([[1.0, 2, 3, 4], [1.5, 2.5, 3.5, 4.5],
                       [500.0, 600.0, 700.0, 800.0], [550.0, 650.0, 750.0, 850.0]])
    gid = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    _n0, rms_good, rm_a, rv_a = solve(rb, gid, 0, 0.1, 1, 1e-5)
    _n1, rms_bad, rm_b, rv_b = solve(rb, gid, 1, 0.1, 1, 1e-5)
    _close(rms_good, torch.tensor([0.999996, 0.999996, 1.0, 1.0]), "src0.rms")
    _close(rms_bad, torch.tensor([2.738599, 2.932146, 643.909302, 18.216745]), "src1.rms")
    _close(rm_a, rm_b, "ledger-is-source-independent")   # 账本不受 stat_source 影响
    _close(rv_a, rv_b, "ledger-is-source-independent")
    assert float(rms_bad[3]) < float(rms_bad[2]), rms_bad    # 账本追上新尺度后失配回落


def check_ledger_follows_batch_means(solve):
    """账本收敛性自检：常数批序列下 EMA 从初值单调逼近批统计（批均值 5、批方差 5）。"""
    const = torch.tensor([[2.0, 4.0, 6.0, 8.0]] * 12)
    gid = torch.arange(4, dtype=torch.long) % 2
    _n, _rms, rm, rv = solve(const, gid, 0, 0.25, 1, 1e-5)
    assert rm.numel() == 13 and rv.numel() == 13
    assert bool((rm[1:] >= rm[:-1] - 1e-6).all()), rm        # 单调上升逼近 5
    assert abs(float(rm[-1]) - 4.841618) < 1e-4, rm[-1]
    assert abs(float(rv[-1]) - 4.873295) < 1e-4, rv[-1]
    _g = solve(const, gid, 0, 0.25, 1, 1e-5)
    _r = _oracle(const, gid, 0, 0.25, 1, 1e-5)
    for i in range(4):
        _close(_g[i], _r[i], "ledger-drift")


def check_group_source_makes_each_group_unit_scale(solve):
    """src=2 + 白化：每个 ≥2 成员的组自己变成零均值、单位 rms（组间尺度互不干扰）。"""
    rb = torch.tensor([
        [1.0, 2.0, 4.0, 8.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        [3.0, 4.0, 5.0, 6.0, -10.0, -20.0, -30.0, -40.0, -50.0],
    ])
    gid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    n, rms, _rm, _rv = solve(rb, gid, 2, 0.5, 1, 1e-5)
    _close(rms, torch.tensor([1.0, 0.999998]), "per-group-unit-rms")
    for row in range(2):
        for grp in (0, 1):
            sel = n[row][gid == grp]
            assert abs(float(sel.mean())) < 1e-5, (row, grp, sel)
            assert abs(float((sel ** 2).mean()) - 1.0) < 1e-4, (row, grp, sel)
    # 组内极差小时，归一化仍把尺度拉回 1：这正是"批全局统计会误杀小组"的反例
    tight = torch.tensor([[10.0, 10.5, 1000.0, 2000.0]])
    g2 = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    t, t_rms, _m2, _v2 = solve(tight, g2, 2, 0.5, 1, 1e-5)
    _close(t[0], torch.tensor([-0.99992, 0.99992, -1.0, 1.0]), "tight-group")
    _close(t_rms, torch.tensor([0.99996]), "tight-group.rms")
    # 换成批全局统计：组内两条小奖励被压成几乎同值，大组独占全部尺度（选错源的后果）
    tb, _rb, _mb, _vb = solve(tight, g2, 0, 0.5, 1, 1e-5)
    assert abs(float(tb[0][1]) - float(tb[0][0])) < 1e-2, tb
    assert abs(float(t[0][1]) - float(t[0][0])) > 1.9, t


def check_shapes_dtypes_and_ledger_length(solve):
    """输出形状/ dtype 契约：running_* 比批数多 1（第 0 项是初值）。"""
    rb = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0], [9.0, 8.0, 7.0], [0.0, 0.0, 1.0]])
    gid = torch.tensor([0, 1, 1], dtype=torch.long)
    for src in (0, 1, 2):
        n, rms, rm, rv = solve(rb, gid, src, 0.4, 1, 1e-5)
        assert n.shape == (4, 3) and rms.shape == (4,) and rm.shape == (5,) and rv.shape == (5,)
        for t in (n, rms, rm, rv):
            assert t.dtype == torch.float32
        assert float(rm[0]) == 0.0 and float(rv[0]) == 1.0
    # 与 oracle 再对一次（B=3 的奇数组，组 0 只有一项）
    got = solve(rb, gid, 2, 0.4, 1, 1e-5)
    ref = _oracle(rb, gid, 2, 0.4, 1, 1e-5)
    for i, name in enumerate(["n", "rms", "mean", "var"]):
        _close(got[i], ref[i], "shape." + name)
    assert torch.equal(got[0][0][0], torch.tensor(0.0))    # 单条组 ⇒ 恒为 0
