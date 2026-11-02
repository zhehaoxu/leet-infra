import math

import torch


def _softmax(row):
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [e / s for e in ex]


def _ce(row, idx):
    m = max(row)
    lse = math.log(sum(math.exp(v - m) for v in row)) + m
    return lse - row[idx]


def _oracle(logits, targets, stride, reg_max):
    """纯 Python 参照：softmax 期望 decode + 左右两个 CE 的加权和。"""
    probs, dist, loss = [], [], []
    for n in range(logits.shape[0]):
        prow, drow, lrow = [], [], []
        for side in range(logits.shape[1]):
            row = logits[n][side].tolist()
            p = _softmax(row)
            prow.append(p)
            drow.append(sum(i * p[i] for i in range(reg_max + 1)) * stride)
            if targets is not None:
                t = targets[n][side].item()
                left = int(t)
                right = left + 1
                lrow.append(_ce(row, left) * (right - t) + _ce(row, right) * (t - left))
        probs.append(prow)
        dist.append(drow)
        if targets is not None:
            loss.append(lrow)
    return probs, dist, loss


def check_uniform_logits(solve):
    out = solve(torch.zeros(1, 4, 8, dtype=torch.float64))
    assert out["dist"].tolist() == [[3.5, 3.5, 3.5, 3.5]], out["dist"]
    assert torch.allclose(out["probs"][0, 0], torch.full((8,), 0.125, dtype=torch.float64), atol=1e-15)
    assert out["bin_index"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def check_peak_bin(solve):
    # 极端 logit 让概率压到单一 bin：期望就是那个 bin 的下标（不是下标+1）
    lg = torch.full((1, 4, 8), -50.0, dtype=torch.float64)
    lg[0, :, 5] = 50.0
    out = solve(lg, stride=4.0)
    assert torch.allclose(out["dist"], torch.full((1, 4), 20.0, dtype=torch.float64), atol=1e-12), out["dist"]
    out2 = solve(torch.full((1, 4, 8), 1e4, dtype=torch.float64))
    assert torch.allclose(out2["probs"][0, 0], torch.full((8,), 0.125, dtype=torch.float64), atol=1e-12)


def check_stride_scales_dist(solve):
    lg = torch.zeros(1, 4, 8, dtype=torch.float64)
    a = solve(lg, stride=1.0)["dist"][0, 0].item()
    b = solve(lg, stride=16.0)["dist"][0, 0].item()
    assert abs(b - 16.0 * a) < 1e-12, (a, b)


def check_integer_target_single_ce(solve):
    # 整数目标：wl=1, wr=0，损失等于单个 bin 的 CE
    lg = torch.zeros(1, 1, 8, dtype=torch.float64)
    t = torch.tensor([[3.0]])
    out = solve(lg, t)
    assert out["w_left"].tolist() == [[1.0]] and out["w_right"].tolist() == [[0.0]], out
    assert abs(out["loss"][0, 0].item() - math.log(8)) < 1e-12, out["loss"]
    assert out["left"].tolist() == [[3]] and out["right"].tolist() == [[4]]


def check_fractional_weights(solve):
    lg = torch.zeros(1, 1, 8, dtype=torch.float64)
    out = solve(lg, torch.tensor([[3.25]]))
    assert abs(out["w_left"][0, 0].item() - 0.75) < 1e-15, out["w_left"]
    assert abs(out["w_right"][0, 0].item() - 0.25) < 1e-15, out["w_right"]
    assert out["left"].tolist() == [[3]] and out["right"].tolist() == [[4]]
    # 两个权重恒和为 1，损失是两条 CE 的凸组合
    assert abs(out["loss"][0, 0].item() - math.log(8)) < 1e-12, out["loss"]


def check_two_ce_not_interpolated_logit(solve):
    # 目标 3.5：0.5/0.5 混合的是两个 CE，不是把 logit 插值后再算一次 CE
    lg = torch.zeros(1, 1, 8, dtype=torch.float64)
    lg[0, 0, 3] = 2.0
    out = solve(lg, torch.tensor([[3.5]]))
    ref = 0.5 * _ce(lg[0, 0].tolist(), 3) + 0.5 * _ce(lg[0, 0].tolist(), 4)
    assert abs(out["loss"][0, 0].item() - ref) < 1e-12, (out["loss"][0, 0].item(), ref)


def check_top_target_allowed(solve):
    # reg_max-1 是允许的最大目标值（右邻 bin 仍可索引）
    out = solve(torch.zeros(1, 1, 8, dtype=torch.float64), torch.tensor([[6.0]]))
    assert out["right"].tolist() == [[7]], out["right"]
    assert abs(out["loss"][0, 0].item() - math.log(8)) < 1e-12


def check_target_out_of_range(solve):
    lg = torch.zeros(1, 1, 8, dtype=torch.float64)
    for bad in ([[7.0]], [[-0.5]], [[6.0001]]):
        try:
            solve(lg, torch.tensor(bad))
        except ValueError:
            continue
        raise AssertionError(f"targets={bad} 应被拒绝")


def check_weight_multiplier(solve):
    lg = torch.zeros(1, 2, 8, dtype=torch.float64)
    t = torch.tensor([[1.0, 2.0]])
    base = solve(lg, t)
    out = solve(lg, t, weights=torch.tensor([[2.0, 0.0]], dtype=torch.float64))
    assert abs(out["loss"][0, 0].item() - 2.0 * base["loss"][0, 0].item()) < 1e-12
    assert out["loss"][0, 1].item() == 0.0
    assert abs(out["loss_mean"].item() - base["loss"][0, 0].item()) < 1e-12


def check_loss_mean_denominator(solve):
    # 归一化分母是元素数（N*4），不是正样本数
    lg = torch.zeros(1, 4, 8, dtype=torch.float64)
    t = torch.ones(1, 4, dtype=torch.float64)
    out = solve(lg, t)
    assert abs(out["loss_mean"].item() - math.log(8)) < 1e-12, out["loss_mean"]
    lg2 = torch.zeros(2, 4, 8, dtype=torch.float64)
    out2 = solve(lg2, torch.ones(2, 4, dtype=torch.float64))
    assert abs(out2["loss_mean"].item() - math.log(8)) < 1e-12


def check_validation(solve):
    ok = torch.zeros(1, 4, 8, dtype=torch.float64)
    bad = [
        dict(reg_max=0),
        dict(reg_max=True),
        dict(stride=0.0),
        dict(stride=-1.0),
        dict(stride=float("inf")),
    ]
    for kw in bad:
        args = dict(reg_max=7)
        args.update(kw)
        try:
            solve(ok, **args)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {kw}")
    for bad_logits in (torch.zeros(4, 8, dtype=torch.float64), torch.zeros(1, 4, 7, dtype=torch.float64)):
        try:
            solve(bad_logits)
        except ValueError:
            continue
        raise AssertionError("形状错误应被拒绝")
    # bool 是 int 子类：用通道数恰好匹配的输入单独验，避免被维度检查掩盖
    try:
        solve(torch.zeros(1, 1, 2, dtype=torch.float64), reg_max=True)
    except ValueError:
        pass
    else:
        raise AssertionError("reg_max=True 应被拒绝")
    try:
        solve(ok, torch.zeros(1, 3, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("targets 形状错误应被拒绝")
    try:
        solve(ok, torch.zeros(1, 4, dtype=torch.float64), weights=torch.zeros(1, 3, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("weights 形状错误应被拒绝")


def check_decode_only(solve):
    out = solve(torch.zeros(1, 4, 8, dtype=torch.float64))
    for key in ("loss", "loss_mean", "left", "w_left"):
        assert key not in out, key


def check_no_alias(solve):
    lg = torch.zeros(1, 4, 8, dtype=torch.float64)
    t = torch.ones(1, 4, dtype=torch.float64)
    out = solve(lg, t)
    assert lg.tolist()[0][0][0] == 0.0 and t.tolist() == [[1.0] * 4], "入参不应被改"
    out["probs"][0, 0, 0] = 99.0
    assert out["dist"][0, 0].item() == 3.5, "dist 不应随 probs 原地改动"
    out["left"][0, 0] = 7
    assert out["right"][0, 0].item() == 2, "left/right 不应共享存储"


def check_against_oracle(solve):
    torch.manual_seed(20261106)
    for trial in range(30):
        n = int(torch.randint(1, 4, (1,)).item())
        reg_max = [3, 7, 15][int(torch.randint(0, 3, (1,)).item())]
        lg = torch.randn(n, 4, reg_max + 1, dtype=torch.float64) * 3.0
        stride = [1.0, 8.0, 32.0][int(torch.randint(0, 3, (1,)).item())]
        tg = (torch.rand(n, 4, dtype=torch.float64) * (reg_max - 1))
        tg = tg.round(decimals=2) if trial % 2 else tg
        out = solve(lg, tg, stride=stride, reg_max=reg_max)
        probs, dist, loss = _oracle(lg, tg, stride, reg_max)
        assert torch.allclose(out["probs"], torch.tensor(probs, dtype=torch.float64), atol=1e-12), trial
        assert torch.allclose(out["dist"], torch.tensor(dist, dtype=torch.float64), atol=1e-11), trial
        assert torch.allclose(out["loss"], torch.tensor(loss, dtype=torch.float64), atol=1e-11), trial
        assert abs(out["loss_mean"].item() - sum(sum(r) for r in loss) / (n * 4)) < 1e-11, trial


check_uniform_logits.example = True
check_integer_target_single_ce.example = True
