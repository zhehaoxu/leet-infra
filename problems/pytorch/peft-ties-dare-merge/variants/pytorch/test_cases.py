import torch

D1 = torch.tensor([[1.0, -2.0, 3.0],
                   [0.5, 4.0, -1.0]])
D2 = torch.tensor([[2.0, -1.0, -3.0],
                   [0.5, 1.0, 5.0]])
M1 = torch.tensor([[1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0]])
M2 = torch.ones(2, 3)

EXP_DARE = torch.tensor([[3.0, -1.0, 0.0],
                         [1.0, 5.0, 4.0]])
EXP_TIES = torch.tensor([[3.0, -2.0, 0.0],
                         [0.0, 8.0, 0.0]])


def _sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _oracle(delta1, delta2, dare_p, mask1, mask2, keep_n):
    """独立重算：纯 Python + 选择排序挑 top-k（并列取小下标），结构异于参考解。"""
    scale = 1.0 / (1.0 - float(dare_p))
    d1 = [[float(delta1[i, j]) * float(mask1[i, j]) * scale for j in range(delta1.shape[1])]
          for i in range(delta1.shape[0])]
    d2 = [[float(delta2[i, j]) * float(mask2[i, j]) * scale for j in range(delta2.shape[1])]
          for i in range(delta2.shape[0])]
    rows, cols = len(d1), len(d1[0])
    dare = [[(d1[i][j] + d2[i][j]) / 2.0 for j in range(cols)] for i in range(rows)]

    flat1 = [x for row in d1 for x in row]
    flat2 = [x for row in d2 for x in row]
    n = len(flat1)

    def select_topk(vals):
        remaining = list(range(n))
        picked = set()
        while len(picked) < int(keep_n) and remaining:
            best = remaining[0]
            for i in remaining:
                if abs(vals[i]) > abs(vals[best]) or (
                        abs(vals[i]) == abs(vals[best]) and i < best):
                    best = i
            picked.add(best)
            remaining.remove(best)
        return picked

    keep1, keep2 = select_topk(flat1), select_topk(flat2)
    t1 = [flat1[i] if i in keep1 else 0.0 for i in range(n)]
    t2 = [flat2[i] if i in keep2 else 0.0 for i in range(n)]
    out = []
    for a, b in zip(t1, t2):
        gamma = _sgn(_sgn(a) + _sgn(b))
        if gamma == 0:
            out.append(0.0)
        else:
            mem = [x for x in (a, b) if _sgn(x) == gamma]
            out.append(sum(mem) / len(mem))
    return (torch.tensor(dare, dtype=torch.float64).reshape(delta1.shape).to(torch.float32),
            torch.tensor(out, dtype=torch.float64).reshape(delta1.shape).to(torch.float32))


def check_example(solve):
    dare, ties = solve(D1, D2, 0.5, M1, M2, 4)
    assert torch.allclose(dare, EXP_DARE, atol=1e-5, rtol=1e-5), f"dare {dare}"
    assert torch.allclose(ties, EXP_TIES, atol=1e-5, rtol=1e-5), f"ties {ties}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(19)
    d1, d2 = torch.randn(3, 4), torch.randn(3, 4)
    m1 = (torch.arange(12) % 3 != 0).to(torch.float32).reshape(3, 4)   # 确定性 0/1 掩码
    m2 = (torch.arange(12) % 4 != 1).to(torch.float32).reshape(3, 4)
    got = solve(d1, d2, 0.25, m1, m2, 7)
    exp = _oracle(d1, d2, 0.25, m1, m2, 7)
    assert torch.allclose(got[0], exp[0], atol=1e-5, rtol=1e-5), "dare 与独立重算不符"
    assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-5), "ties 与独立重算不符"


def check_p_zero_degenerates_to_average(solve):
    ones = torch.ones(2, 3)
    dare, ties = solve(D1, D2, 0.0, ones, ones, 6)
    assert torch.allclose(dare, (D1 + D2) / 2.0, atol=1e-5, rtol=1e-5), (
        "p=0 且掩码全 1 时 DARE 应退化为普通平均")
    exp_ties = torch.tensor([[1.5, -1.5, 0.0],
                             [0.5, 2.5, 0.0]])   # 同号均值；(0,2)与(1,2)一正一负冲突置零
    assert torch.allclose(ties, exp_ties, atol=1e-5, rtol=1e-5), f"ties {ties}"


def check_sign_conflict_zeroed(solve):
    """一正一负的参数：elect 得 gamma=0，TIES 置零（即便幅度最大）。"""
    v = torch.tensor([[5.0]])
    dare, ties = solve(v, -v, 0.0, torch.ones(1, 1), torch.ones(1, 1), 1)
    assert float(dare[0, 0]) == 0.0 and float(ties[0, 0]) == 0.0


def check_single_survivor_disjoint_mean(solve):
    """单侧幸存：TIES 取幸存者原值（分母=1），而非除以任务总数 2。"""
    d1 = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    d2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    zeros = torch.zeros(2, 2)
    dare, ties = solve(d1, d2, 0.0, zeros + 1, zeros + 1, 4)
    assert torch.allclose(ties, d1 + d2, atol=1e-5, rtol=1e-5), (
        f"ties {ties} 应等于幸存者原值之和（各位置仅一侧非零）")
