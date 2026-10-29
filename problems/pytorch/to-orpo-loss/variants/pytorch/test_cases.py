import math

import torch

LOGW = torch.tensor([-2.0, -1.5], dtype=torch.float64)
LOGL = torch.tensor([-3.0, -2.4], dtype=torch.float64)
LENW = torch.tensor([2, 3])
LENL = torch.tensor([3, 4])
LAM = 0.1
EXAMPLE_LOSS = 0.813742563169928


def _log_odds(lp):
    """独立路径：先算概率再取 log(p/(1-p))，与参考解的 log1p 稳定式结构不同。"""
    p = math.exp(lp)
    return math.log(p / (1.0 - p))


def _oracle(logw, logl, nw, nl, lam):
    tot_sft = 0.0
    tot_or = 0.0
    n = len(logw)
    for i in range(n):
        lw = logw[i] / nw[i]
        ll = logl[i] / nl[i]
        tot_sft += -lw
        d = _log_odds(lw) - _log_odds(ll)
        tot_or += -min(0.0, d) + math.log1p(math.exp(-abs(d)))
    return tot_sft / n + lam * tot_or / n


def check_example(solve):
    y = solve(LOGW, LOGL, LENW, LENL, LAM)
    assert y.dim() == 0 and y.dtype == torch.float64, (y.dim(), y.dtype)
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-12, y.item()


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(11)
    for _ in range(10):
        n = 6
        logw = (-torch.rand(n, dtype=torch.float64) * 4.0 - 0.5).tolist()
        logl = (-torch.rand(n, dtype=torch.float64) * 4.0 - 0.5).tolist()
        nw = torch.randint(1, 9, (n,)).tolist()
        nl = torch.randint(1, 9, (n,)).tolist()
        got = solve(torch.tensor(logw, dtype=torch.float64),
                    torch.tensor(logl, dtype=torch.float64),
                    torch.tensor(nw), torch.tensor(nl), 0.1).item()
        exp = _oracle(logw, logl, nw, nl, 0.1)
        assert abs(got - exp) < 1e-12, (got, exp)


def check_lambda_zero_is_sft(solve):
    # λ=0 只剩 SFT 项：-mean(logp_w/len_w)，与赔率项无关
    y = solve(LOGW, LOGL, LENW, LENL, 0.0)
    exp = -sum(LOGW.tolist()[i] / LENW.tolist()[i] for i in range(2)) / 2
    assert abs(y.item() - exp) < 1e-12, (y.item(), exp)


def check_equal_logodds_gives_ln2(solve):
    # 两侧平均对数概率相同 -> logodds 差为 0 -> 赔率项恰为 ln2
    logw = torch.tensor([-2.0], dtype=torch.float64)
    logl = torch.tensor([-4.0], dtype=torch.float64)
    y = solve(logw, logl, torch.tensor([2]), torch.tensor([4]), 0.1)
    assert abs(y.item() - (1.0 + 0.1 * math.log(2.0))) < 1e-12, y.item()


def check_odds_not_logp(solve):
    # lp_w=-1、lp_l=-2：正确的 log-odds 差 ~1.3133，若误用 lp 差只有 1.0，
    # 两种口径的损失差 > 1e-3，足以区分实现
    logw = torch.tensor([-1.0], dtype=torch.float64)
    logl = torch.tensor([-4.0], dtype=torch.float64)
    nw, nl = torch.tensor([1]), torch.tensor([2])
    lw, ll = -1.0, -2.0
    d_odds = _log_odds(lw) - _log_odds(ll)
    exp_odds = 1.0 + 0.1 * math.log1p(math.exp(-d_odds))
    y = solve(logw, logl, nw, nl, 0.1).item()
    assert abs(y - exp_odds) < 1e-12, (y, exp_odds)
    assert abs(y - (1.0 + 0.1 * math.log1p(math.exp(-1.0)))) > 1e-3, y


def check_float32_input(solve):
    y = solve(LOGW.to(torch.float32), LOGL.to(torch.float32), LENW, LENL, LAM)
    assert y.dtype == torch.float64
    assert abs(y.item() - EXAMPLE_LOSS) < 1e-6, y.item()


def check_length_normalization(solve):
    # 同一序列 logp 但长度翻倍：平均对数概率减半，SFT 项与赔率项都随之改变
    y1 = solve(torch.tensor([-2.0]), torch.tensor([-2.0]),
               torch.tensor([2]), torch.tensor([2]), 0.0)
    y2 = solve(torch.tensor([-2.0]), torch.tensor([-2.0]),
               torch.tensor([4]), torch.tensor([2]), 0.0)
    assert abs(y1.item() - 1.0) < 1e-12 and abs(y2.item() - 0.5) < 1e-12, (y1.item(), y2.item())


def check_errors(solve):
    good_lw = torch.tensor([-2.0], dtype=torch.float64)
    good_ll = torch.tensor([-3.0], dtype=torch.float64)
    good_n = torch.tensor([2])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{desc}: 应抛 ValueError，实际 {type(exc).__name__}")
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([-2.0], good_ll, good_n, good_n), "logp_w 非张量")
    bad(lambda: solve(good_lw.unsqueeze(0), good_ll, good_n, good_n), "logp_w 非一维")
    bad(lambda: solve(good_lw.to(torch.int64), good_ll.to(torch.int64), good_n, good_n), "logp 非浮点")
    bad(lambda: solve(good_lw, torch.tensor([-3.0, -3.0]), good_n, good_n), "形状不一致")
    bad(lambda: solve(good_lw, good_ll, torch.tensor([0]), good_n), "len 为 0")
    bad(lambda: solve(good_lw, good_ll, good_n.to(torch.float32), good_n), "len 非整型")
    bad(lambda: solve(torch.tensor([0.5]), good_ll, good_n, good_n), "平均对数概率 >= 0")
    bad(lambda: solve(good_lw, good_ll, good_n, good_n, -0.1), "lam 为负")
    bad(lambda: solve(torch.tensor([float("nan")]), good_ll, good_n, good_n), "含 NaN")
