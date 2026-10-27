import math

import torch
import torch.nn.functional as F

# 示例期望值：手写 4 样本 2 类，margin=1.0、p=2.0、swap=False（实测回填）
EX_EMB = [[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [1.0, 2.0]]
EX_LABELS = [0, 0, 1, 1]
EXAMPLE_BATCHHARD = 0.7928932309  # 实测回填
EXAMPLE_SEMIHARD = 0.5986121893   # 实测回填


def _oracle_solve(emb, labels, margin, p, swap, strategy, tie_last=False):
    """独立双循环 python oracle：math 逐维算 p-范数 + 显式挖掘循环。

    tie_last=False 时并列距离取最小下标（与 spec 一致）；True 时取最大下标，
    仅供 tie-break 探针用来验证「取错下标会得到明显不同的损失」。
    """
    n = emb.shape[0]
    E = [[float(v) for v in row] for row in emb]
    L = [int(v) for v in labels]

    def dist(i, j):
        s = 0.0
        for u, v in zip(E[i], E[j]):
            s += abs(u - v) ** p
        return s ** (1.0 / p)

    losses = []
    for a in range(n):
        pos = [j for j in range(n) if j != a and L[j] == L[a]]
        neg = [j for j in range(n) if L[j] != L[a]]
        if not pos or not neg:
            continue
        d_pos, j_pos = None, None
        for j in pos:  # 从小到大扫描；严格比较使并列时保留先出现者（最小 j）
            d = dist(a, j)
            if d_pos is None or (d > d_pos if not tie_last else d >= d_pos):
                d_pos, j_pos = d, j
        if strategy == "semihard":
            cand = [j for j in neg if d_pos < dist(a, j) < d_pos + margin]
            pool = cand if cand else neg
        else:
            pool = neg
        d_neg, j_neg = None, None
        for j in pool:
            d = dist(a, j)
            if d_neg is None or (d < d_neg if not tie_last else d <= d_neg):
                d_neg, j_neg = d, j
        if swap:
            d_neg = min(d_neg, dist(j_pos, j_neg))
        losses.append(max(0.0, margin + d_pos - d_neg))
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def _cmp(solve, emb, labels, margin, p, swap, strategy):
    got = solve(emb, labels, margin=margin, p=p, swap=swap, strategy=strategy)
    exp = _oracle_solve(emb, labels, margin, p, swap, strategy)
    assert got.shape == (), f"应返回 0 维标量张量，got shape {got.shape}"
    assert torch.allclose(got, torch.tensor(exp, dtype=got.dtype), atol=1e-5, rtol=1e-5), \
        (got.item(), exp)
    return got


def _grid_emb(seed, n, d, hi=5):
    """固定种子的整数网格 embeddings：fp32/fp64 下距离精确，避免假性近并列。"""
    torch.manual_seed(seed)
    return torch.randint(0, hi, (n, d)).float()


def check_example(solve):
    emb = torch.tensor(EX_EMB)
    labels = torch.tensor(EX_LABELS)
    bh = solve(emb, labels, strategy="batch_hard")
    sh = solve(emb, labels, strategy="semihard")
    assert bh.shape == () and sh.shape == ()
    assert abs(bh.item() - EXAMPLE_BATCHHARD) < 1e-6, bh.item()
    assert abs(sh.item() - EXAMPLE_SEMIHARD) < 1e-6, sh.item()


check_example.example = True


def check_matches_oracle_random(solve):
    for seed in range(10):
        n = 4 + seed
        d = 2 + seed % 3
        emb = _grid_emb(100 + seed, n, d)
        torch.manual_seed(200 + seed)
        labels = torch.randint(0, 3, (n,))
        for margin in (1.0, 0.5):
            for p in (2.0, 1.0):
                for swap in (False, True):
                    _cmp(solve, emb, labels, margin, p, swap, "batch_hard")
                    _cmp(solve, emb, labels, margin, p, swap, "semihard")
    # 固定种子连续值数据补充两组（float 路径）
    for seed in (77, 78):
        torch.manual_seed(seed)
        emb = torch.randn(8, 4)
        torch.manual_seed(seed + 1)
        labels = torch.randint(0, 2, (8,))
        _cmp(solve, emb, labels, 1.0, 2.0, False, "batch_hard")
        _cmp(solve, emb, labels, 1.0, 2.0, True, "semihard")


def check_swap_formula_matches_torch(solve):
    # 判别性单 triple：d_ap=3, d_an=√10≈3.1623, d_pn=1
    a = torch.tensor([[0.0, 0.0]])
    p = torch.tensor([[3.0, 0.0]])
    n = torch.tensor([[3.0, 1.0]])
    margin = 1.0
    d_ap = math.sqrt(9.0)
    d_an = math.sqrt(10.0)
    d_pn = math.sqrt(1.0)
    f_noswap = F.triplet_margin_loss(a, p, n, margin=margin, swap=False, reduction="none")
    f_swap = F.triplet_margin_loss(a, p, n, margin=margin, swap=True, reduction="none")
    # swap=False: relu(margin + d_ap − d_an)
    assert torch.allclose(f_noswap, torch.tensor(max(0.0, margin + d_ap - d_an)), atol=1e-6)
    # swap=True 的 min 作用在负例距离一侧：relu(margin + d_ap − min(d_an, d_pn))
    assert torch.allclose(f_swap, torch.tensor(max(0.0, margin + d_ap - min(d_an, d_pn))),
                          atol=1e-6), f_swap.item()
    # 反例钉死：若 min 作用在 d_ap 一侧（relu(margin + min(d_ap,d_pn) − d_an)）会得到 0，
    # 与实测 F 输出 3.0 相差甚远
    wrong_side = max(0.0, margin + min(d_ap, d_pn) - d_an)
    assert abs(f_swap.item() - wrong_side) > 1.0
    # 随机批量 per-triplet 对拍（p=1/2 × swap 两种）
    torch.manual_seed(7)
    A, P, N = torch.randn(16, 5), torch.randn(16, 5), torch.randn(16, 5)
    for pv in (2.0, 1.0):
        for swap in (False, True):
            f = F.triplet_margin_loss(A, P, N, margin=0.7, p=int(pv), swap=swap,
                                      reduction="none")
            man = []
            for i in range(16):
                dap = math.sqrt(sum((float(x) - float(y)) ** 2
                                    for x, y in zip(A[i], P[i]))) if pv == 2.0 else \
                    sum(abs(float(x) - float(y)) for x, y in zip(A[i], P[i]))
                dan = math.sqrt(sum((float(x) - float(y)) ** 2
                                    for x, y in zip(A[i], N[i]))) if pv == 2.0 else \
                    sum(abs(float(x) - float(y)) for x, y in zip(A[i], N[i]))
                dpn = math.sqrt(sum((float(x) - float(y)) ** 2
                                    for x, y in zip(P[i], N[i]))) if pv == 2.0 else \
                    sum(abs(float(x) - float(y)) for x, y in zip(P[i], N[i]))
                dneg = min(dan, dpn) if swap else dan
                man.append(max(0.0, 0.7 + dap - dneg))
            assert torch.allclose(f, torch.tensor(man), atol=1e-5), (pv, swap, f.tolist())
    # 端到端：构造每 anchor 恰一个正例/负例的 batch，solve 输出须等于 F 的均值
    emb = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1])
    tri_a, tri_p, tri_n = emb[[0, 1]], emb[[1, 0]], emb[[2, 2]]
    for swap in (False, True):
        ref = F.triplet_margin_loss(tri_a, tri_p, tri_n, margin=1.0, p=2, swap=swap,
                                    reduction="mean")
        got = solve(emb, labels, margin=1.0, p=2.0, swap=swap, strategy="batch_hard")
        assert torch.allclose(got, ref, atol=1e-6), (got.item(), ref.item())


def check_semihard_trigger_and_fallback(solve):
    # 触发例：a0 的负例 {4, 5.5}，d_pos=5，开区间 (5,6) 只含 5.5（非全局最小 4）；
    # a3 的负例 {5.5, 0.5} 区间 (1.5,2.5) 为空走回退（min=0.5）
    emb = torch.tensor([[0.0], [5.0], [4.0], [5.5]])
    labels = torch.tensor([0, 0, 1, 1])
    sh = solve(emb, labels, margin=1.0, p=2.0, strategy="semihard")
    bh = solve(emb, labels, margin=1.0, p=2.0, strategy="batch_hard")
    # sh: a0=relu(1+5−5.5)=0.5, a1=relu(1+5−0.5)=5.5, a2=relu(1+1.5−1)=1.5, a3=relu(1+1.5−0.5)=2
    assert torch.allclose(sh, torch.tensor(2.375), atol=1e-6), sh.item()
    # bh: a0 取全局最小 4 → relu(1+5−4)=2，其余同 sh
    assert torch.allclose(bh, torch.tensor(2.75), atol=1e-6), bh.item()
    assert sh.item() < bh.item(), "semihard 命中区间内负例（5.5）而非全局最难（4）"
    _cmp(solve, emb, labels, 1.0, 2.0, False, "semihard")
    _cmp(solve, emb, labels, 1.0, 2.0, False, "batch_hard")
    # 回退例：区间 (5,6) 内无负例（2、7 都不在）→ 退回全局最难，与 batch_hard 相同
    emb2 = torch.tensor([[0.0], [5.0], [2.0], [7.0]])
    labels2 = torch.tensor([0, 0, 1, 1])
    sh2 = solve(emb2, labels2, margin=1.0, p=2.0, strategy="semihard")
    bh2 = solve(emb2, labels2, margin=1.0, p=2.0, strategy="batch_hard")
    assert torch.allclose(sh2, bh2, atol=1e-6), (sh2.item(), bh2.item())
    assert torch.allclose(sh2, torch.tensor(4.0), atol=1e-6), sh2.item()
    _cmp(solve, emb2, labels2, 1.0, 2.0, False, "semihard")


def check_tie_break_smallest_index(solve):
    # 负例并列（两个 d=2）但 d(pos,neg) 不同：x2=(0,2) → d(x1,x2)=√13、
    # x3=(2,0) → d(x1,x3)=1。正确 tie-break 取 j_neg=2 → swap 后
    # d_neg_eff=min(2,√13)=2 → a0 损失 relu(1+3−2)=2；取错成 x3 → d_neg_eff=1 → 3
    emb = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    got = solve(emb, labels, margin=1.0, p=2.0, swap=True, strategy="batch_hard")
    correct = _oracle_solve(emb, labels, 1.0, 2.0, True, "batch_hard", tie_last=False)
    wrong = _oracle_solve(emb, labels, 1.0, 2.0, True, "batch_hard", tie_last=True)
    assert torch.allclose(got, torch.tensor(2.4142135), atol=1e-5), got.item()
    assert torch.allclose(got, torch.tensor(correct, dtype=got.dtype), atol=1e-5), \
        (got.item(), correct)
    assert abs(got.item() - wrong) > 0.05, \
        f"探针失去判别力：正确 tie-break {correct} 与错误 tie-break {wrong} 应明显不同"
    # 正例并列（两个 d=3）但 d(pos,neg) 不同：j_pos=1（x=(0,3)）→ swap 用
    # d(x1,x3)=√13 → d_neg_eff=min(√10,√13)=√10 → a0 损失 ≈0.8377；
    # 取错成 j_pos=2（x=(3,0)）→ d(x2,x3)=1 → d_neg_eff=1 → a0 损失 3
    emb2 = torch.tensor([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 1.0]])
    labels2 = torch.tensor([0, 0, 0, 1])
    got2 = solve(emb2, labels2, margin=1.0, p=2.0, swap=True, strategy="batch_hard")
    correct2 = _oracle_solve(emb2, labels2, 1.0, 2.0, True, "batch_hard")
    wrong2 = _oracle_solve(emb2, labels2, 1.0, 2.0, True, "batch_hard", tie_last=True)
    assert torch.allclose(got2, torch.tensor(3.1076678), atol=1e-5), got2.item()
    assert torch.allclose(got2, torch.tensor(correct2, dtype=got2.dtype), atol=1e-5), \
        (got2.item(), correct2)
    assert abs(got2.item() - wrong2) > 0.05, (got2.item(), correct2, wrong2)
    # semihard 区间内并列（两个 d=1.5）但 d(pos,neg) 不同：j_neg=2（x=(1.5,0)）→
    # swap 用 d(x1,x2)=0.5 → d_neg_eff=0.5 → a0 损失 1.5；取错成 x3（d=1.803）→ 0.5。
    # a2/a3 的 d_pos=√4.5≈2.1213，区间空走回退
    emb3 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
    labels3 = torch.tensor([0, 0, 1, 1])
    got3 = solve(emb3, labels3, margin=1.0, p=2.0, swap=True, strategy="semihard")
    correct3 = _oracle_solve(emb3, labels3, 1.0, 2.0, True, "semihard", tie_last=False)
    wrong3 = _oracle_solve(emb3, labels3, 1.0, 2.0, True, "semihard", tie_last=True)
    assert torch.allclose(got3, torch.tensor(1.5606602), atol=1e-5), got3.item()
    assert torch.allclose(got3, torch.tensor(correct3, dtype=got3.dtype), atol=1e-5), \
        (got3.item(), correct3)
    assert abs(got3.item() - wrong3) > 0.05, (got3.item(), correct3, wrong3)


def check_margin_zero(solve):
    # margin=0：relu(d_pos − d_neg)，仍有非零损失
    emb = torch.tensor([[0.0], [10.0], [2.0]])
    labels = torch.tensor([0, 0, 1])
    got = solve(emb, labels, margin=0.0, p=2.0)
    assert torch.allclose(got, torch.tensor(5.0), atol=1e-6), got.item()
    _cmp(solve, emb, labels, 0.0, 2.0, False, "batch_hard")
    _cmp(solve, emb, labels, 0.0, 2.0, True, "batch_hard")
    # margin=0 且类别分离良好：全部 anchor 归零
    emb2 = torch.tensor([[0.0], [1.0], [10.0]])
    got2 = solve(emb2, labels, margin=0.0, p=2.0)
    assert got2.item() == 0.0, got2.item()


def check_degenerate_and_errors(solve):
    # 单标签全类：无负例 → 0
    emb = _grid_emb(11, 6, 3)
    labels = torch.zeros(6, dtype=torch.long)
    got = solve(emb, labels)
    assert got.shape == () and got.item() == 0.0, got
    # 全异标签：无正例 → 0
    labels2 = torch.tensor([0, 1, 2, 3, 4, 5])
    got2 = solve(emb, labels2)
    assert got2.shape == () and got2.item() == 0.0, got2
    # N=1 与 N=0 → 0
    got3 = solve(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))
    assert got3.item() == 0.0
    got4 = solve(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
    assert got4.shape == () and got4.item() == 0.0, got4
    # 非法 strategy 抛 ValueError
    try:
        solve(emb, torch.tensor([0, 0, 1, 1, 2, 2]), strategy="random")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 strategy 应抛 ValueError")
