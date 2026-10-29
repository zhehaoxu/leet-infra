import math

import torch
import torch.nn.functional as F

STUDENT = torch.tensor(
    [[2.0, 0.0, -1.0, 0.5],
     [0.0, 1.0, 0.5, -0.5],
     [1.0, -2.0, 0.3, 0.0]], dtype=torch.float64)
TEACHER = torch.tensor(
    [[1.0, 1.0, 0.0, -0.5],
     [0.2, 2.0, 0.1, 0.0],
     [-1.0, 0.0, 1.0, 0.5]], dtype=torch.float64)
LABELS = torch.tensor([0, 1, -100])
EXAMPLE_LOSS = 0.31461770251685367


def _lse(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _oracle(sl, tl, labels, lam, beta, tau, ignore=-100):
    tot_jsd = 0.0
    tot_ce = 0.0
    cnt = 0
    for i, y in enumerate(labels):
        if y == ignore:
            continue
        ls = [v / tau for v in sl[i]]
        lt = [v / tau for v in tl[i]]
        zs = _lse(ls)
        zt = _lse(lt)
        ps = [math.exp(v - zs) for v in ls]
        pt = [math.exp(v - zt) for v in lt]
        m = [beta * pt[j] + (1.0 - beta) * ps[j] for j in range(len(ps))]
        kl_t = sum(pt[j] * ((lt[j] - zt) - math.log(m[j])) for j in range(len(ps)))
        kl_s = sum(ps[j] * ((ls[j] - zs) - math.log(m[j])) for j in range(len(ps)))
        tot_jsd += beta * kl_t + (1.0 - beta) * kl_s
        tot_ce += _lse(sl[i]) - sl[i][y]
        cnt += 1
    return (1.0 - lam) * tot_jsd / cnt + lam * tot_ce / cnt


def check_example(solve):
    y = solve(STUDENT, TEACHER, LABELS, 0.5, 0.5, 1.0)
    assert y.dim() == 0 and y.dtype == torch.float64, (y.dim(), y.dtype)
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-12, y.item()


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(19)
    for _ in range(6):
        n, v = 6, 5
        st = torch.randn(n, v, dtype=torch.float64)
        te = torch.randn(n, v, dtype=torch.float64)
        labels = torch.randint(0, v, (n,))
        labels[::4] = -100
        for lam, beta, tau in ((0.5, 0.5, 1.0), (0.2, 0.3, 2.0), (0.9, 0.8, 0.5)):
            got = solve(st, te, labels, lam, beta, tau).item()
            exp = _oracle(st.tolist(), te.tolist(), labels.tolist(), lam, beta, tau)
            assert abs(got - exp) < 1e-12, (lam, beta, tau, got, exp)


def check_lam_one_is_ce(solve):
    # λ=1 时只剩硬标签 CE，且 CE 用未缩放 logits
    valid = LABELS != -100
    exp = F.cross_entropy(STUDENT[valid], LABELS[valid]).item()
    y = solve(STUDENT, TEACHER, LABELS, 1.0, 0.5, 3.0)
    assert abs(y.item() - exp) < 1e-12, (y.item(), exp)


def check_jsd_symmetric_at_half(solve):
    # β=0.5 的 skew-JSD 退化为标准 JSD：交换 teacher/student 不变
    a = solve(STUDENT, TEACHER, LABELS, 0.0, 0.5, 1.0).item()
    b = solve(TEACHER, STUDENT, LABELS, 0.0, 0.5, 1.0).item()
    assert abs(a - b) < 1e-12, (a, b)
    assert a > 0.0


def check_temperature_matters(solve):
    base = solve(STUDENT, TEACHER, LABELS, 0.0, 0.5, 1.0).item()
    for tau in (0.5, 2.0, 4.0):
        y = solve(STUDENT, TEACHER, LABELS, 0.0, 0.5, tau).item()
        assert abs(y - base) > 1e-6, (tau, y, base)


def check_ignore_index_invariance(solve):
    base = solve(STUDENT, TEACHER, LABELS, 0.5, 0.5, 1.0).item()
    st = STUDENT.clone()
    te = TEACHER.clone()
    st[2] = torch.tensor([5.0, -7.0, 3.0, 0.1], dtype=torch.float64)
    te[2] = torch.tensor([-4.0, 2.0, 6.0, -1.0], dtype=torch.float64)
    y = solve(st, te, LABELS, 0.5, 0.5, 1.0).item()
    assert abs(y - base) < 1e-12, (y, base)


def check_beta_endpoints_rejected(solve):
    for beta in (0.0, 1.0, -0.2, 1.5):
        try:
            solve(STUDENT, TEACHER, LABELS, 0.5, beta, 1.0)
        except ValueError:
            continue
        raise AssertionError(f"beta={beta} 应抛 ValueError")


def check_float32_input(solve):
    y = solve(STUDENT.to(torch.float32), TEACHER.to(torch.float32), LABELS, 0.5, 0.5, 1.0)
    assert y.dtype == torch.float64
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-5, y.item()


def check_errors(solve):
    good_st = torch.zeros(2, 3, dtype=torch.float64)
    good_lab = torch.tensor([0, 1])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([0.0], good_st, good_lab, 0.5, 0.5, 1.0), "student 非张量")
    bad(lambda: solve(torch.zeros(2, 3, 1, dtype=torch.float64), torch.zeros(2, 3, 1, dtype=torch.float64), good_lab, 0.5, 0.5, 1.0), "非二维")
    bad(lambda: solve(good_st, torch.zeros(3, 2, dtype=torch.float64), good_lab, 0.5, 0.5, 1.0), "形状不一致")
    bad(lambda: solve(good_st, good_st, torch.tensor([0]), 0.5, 0.5, 1.0), "labels 长度不符")
    bad(lambda: solve(good_st, good_st, torch.tensor([0, 3]), 0.5, 0.5, 1.0), "标签越界")
    bad(lambda: solve(good_st, good_st, torch.tensor([-100, -100]), 0.5, 0.5, 1.0), "全被忽略")
    bad(lambda: solve(good_st, good_st, good_lab, -0.1, 0.5, 1.0), "lam 为负")
    bad(lambda: solve(good_st, good_st, good_lab, 1.1, 0.5, 1.0), "lam 超 1")
    bad(lambda: solve(good_st, good_st, good_lab, 0.5, 0.5, 0.0), "temperature=0")
    bad(lambda: solve(torch.tensor([[float("nan"), 0.0, 0.0]]), good_st[:1], torch.tensor([0]), 0.5, 0.5, 1.0), "含 NaN")
