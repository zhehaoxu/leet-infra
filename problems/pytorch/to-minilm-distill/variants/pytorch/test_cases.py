import math

import torch

TA = torch.tensor(
    [[[[1.0, 0.0, -1.0], [0.5, 1.5, 0.0], [2.0, 0.0, 1.0]],
      [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.5]]],
     [[[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, -1.0, 1.0]],
      [[0.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]]], dtype=torch.float64)
SA = torch.tensor(
    [[[[0.5, 0.5, -0.5], [0.0, 1.0, 0.5], [1.0, 0.5, 1.0]],
      [[1.0, 0.0, 0.0], [0.5, 0.5, 1.0], [0.0, 1.0, 0.0]]],
     [[[0.5, 1.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]],
      [[0.5, 0.5, 0.0], [1.0, 1.0, 0.0], [1.0, 0.5, 1.0]]]], dtype=torch.float64)
TC = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [-1.0, 2.0]],
                   [[0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]], dtype=torch.float64)
SC = torch.tensor([[[0.5, 0.0], [0.0, 1.0], [-1.0, 1.0]],
                   [[0.5, 0.5], [1.0, 0.0], [1.0, -1.0]]], dtype=torch.float64)
MASK = torch.tensor([[True, True, False], [True, True, True]])
EXAMPLE = {"attn_kl": 0.16755623451792415,
           "cls_mse": 0.3541666666666667,
           "loss": 0.5217229011845909}


def _softmax(vals):
    m = max(vals)
    ex = [math.exp(v - m) for v in vals]
    s = sum(ex)
    return [v / s for v in ex]


def _oracle(ta, sa, tc, sc, mask, tau):
    b, h, l, _ = len(ta), len(ta[0]), len(ta[0][0]), len(ta[0][0][0])
    tot = 0.0
    nq = 0
    for bi in range(b):
        keys = [j for j in range(l) if mask[bi][j]]
        for hi in range(h):
            for i in range(l):
                if not mask[bi][i]:
                    continue
                lt = [ta[bi][hi][i][j] / tau for j in keys]
                ls = [sa[bi][hi][i][j] / tau for j in keys]
                pt = _softmax(lt)
                ps = _softmax(ls)
                for k, j in enumerate(keys):
                    tot += pt[k] * (math.log(pt[k]) - math.log(ps[k]))
                nq += 1
    attn = tot / nq  # nq 已按 (batch, head, query) 计数
    se = 0.0
    cnt = 0
    for bi in range(b):
        for k in range(3):
            for d in range(len(tc[0][0])):
                se += (tc[bi][k][d] - sc[bi][k][d]) ** 2
                cnt += 1
    return {"attn_kl": attn, "cls_mse": se / cnt, "loss": attn + se / cnt}


def check_example(solve):
    out = solve(TA, SA, TC, SC, MASK)
    assert set(out) == {"attn_kl", "cls_mse", "loss"}
    for k, v in out.items():
        assert v.dim() == 0 and v.dtype == torch.float64, (k, v.dim(), v.dtype)
        assert abs(v.item() - EXAMPLE[k]) < 1e-12, (k, v.item())
    assert abs(out["loss"].item() - (out["attn_kl"] + out["cls_mse"]).item()) < 1e-15


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(23)
    for _ in range(4):
        b, h, l, d = 3, 2, 4, 3
        ta = torch.randn(b, h, l, l, dtype=torch.float64)
        sa = torch.randn(b, h, l, l, dtype=torch.float64)
        tc = torch.randn(b, 3, d, dtype=torch.float64)
        sc = torch.randn(b, 3, d, dtype=torch.float64)
        m = torch.rand(b, l) > 0.25
        m[:, 0] = True
        for tau in (1.0, 2.5):
            got = solve(ta, sa, tc, sc, m, tau)
            exp = _oracle(ta.tolist(), sa.tolist(), tc.tolist(), sc.tolist(), m.tolist(), tau)
            for k in ("attn_kl", "cls_mse", "loss"):
                assert abs(got[k].item() - exp[k]) < 1e-12, (tau, k, got[k].item(), exp[k])


def check_identical_is_zero(solve):
    out = solve(TA, TA.clone(), TC, TC.clone(), MASK)
    assert out["attn_kl"].item() == 0.0 and out["cls_mse"].item() == 0.0, out


def check_masked_key_invariance(solve):
    base = solve(TA, SA, TC, SC, MASK)
    st = SA.clone()
    st[0, 0, :, 2] = 99.0
    st[0, 1, :, 2] = -99.0
    out = solve(TA, st, TC, SC, MASK)
    assert abs(out["attn_kl"].item() - base["attn_kl"].item()) < 1e-15, (out, base)


def check_masked_query_ignored(solve):
    # 第 0 个 batch 的第 2 个位置被 mask：改它的整行不影响结果
    base = solve(TA, SA, TC, SC, MASK)
    st = SA.clone()
    st[0, :, 2, :] = 7.5
    out = solve(TA, st, TC, SC, MASK)
    assert abs(out["attn_kl"].item() - base["attn_kl"].item()) < 1e-15


def check_cls_independent_of_attn(solve):
    base = solve(TA, SA, TC, SC, MASK)
    out = solve(TA * 3.0, SA - 1.0, TC, SC, MASK)
    assert abs(out["cls_mse"].item() - base["cls_mse"].item()) < 1e-15
    assert abs(out["attn_kl"].item() - base["attn_kl"].item()) > 1e-6


def check_errors(solve):
    good = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    good_cls = torch.zeros(1, 3, 2, dtype=torch.float64)
    good_mask = torch.ones(1, 2, dtype=torch.bool)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(TA.tolist(), SA, TC, SC, MASK), "attn 非张量")
    bad(lambda: solve(good[0], good[0], good_cls, good_cls, good_mask), "attn 非四维")
    bad(lambda: solve(good, torch.zeros(1, 1, 2, 3, dtype=torch.float64), good_cls, good_cls, good_mask), "非方阵")
    bad(lambda: solve(good, good, good_cls, good_cls, good_mask[0]), "mask 非二维")
    bad(lambda: solve(good, good, good_cls, good_cls, torch.zeros(1, 2, dtype=torch.bool)), "mask 全 False")
    bad(lambda: solve(good, good, good_cls, good_cls, good_mask.to(torch.int64)), "mask 非 bool")
    bad(lambda: solve(good, good, torch.zeros(1, 2, 2, dtype=torch.float64), good_cls, good_mask), "CLS 非 3 对")
    bad(lambda: solve(good, good, good_cls, torch.zeros(1, 3, 3, dtype=torch.float64), good_mask), "CLS 形状不符")
    bad(lambda: solve(good, good, good_cls, good_cls, good_mask, 0.0), "temperature=0")
    bad(lambda: solve(torch.full((1, 1, 2, 2), float("nan")), good, good_cls, good_cls, good_mask), "含 NaN")
