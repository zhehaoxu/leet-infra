import copy
import math

import torch

JUNK = 2


def _draw_plan(with_junk):
    """与题目同序重放全局 RNG：6 个训练步各 randn(1,2)+randn(1)；
    with_junk=True 时在第 3 步与第 4 步之间额外 randn(2)。
    torch 随机流无法用 numpy 复现，数据抽取是本仓库 framework 题对拍口径中
    唯一共享的原语；其余数学全部纯 Python 重算。"""
    torch.manual_seed(1234)
    pairs = []
    for i in range(6):
        if with_junk and i == 3:
            torch.randn(JUNK)
        x = torch.randn(1, 2).tolist()[0]
        y = float(torch.randn(1).item())
        pairs.append((x, y))
    return pairs


def _fresh_state():
    return {
        "W1": [[0.5, -0.25], [0.125, 0.25]],
        "b1": [0.0, 0.5],
        "w2": [0.25, -0.5],
        "b2": 0.0,
        "mW1": [[0.0, 0.0], [0.0, 0.0]],
        "mb1": [0.0, 0.0],
        "mw2": [0.0, 0.0],
        "mb2": 0.0,
    }


def _run_steps(pairs, st, start, end):
    """纯 Python 双循环重算：前向/反向/动量 SGD/StepLR，全程 float64。"""
    losses = []
    for i in range(start, end):
        x, y = pairs[i]
        lr = 0.1 * 0.5 ** (i // 2)          # 第 i 步生效的 lr
        W1, b1, w2, b2 = st["W1"], st["b1"], st["w2"], st["b2"]
        mW1, mb1, mw2 = st["mW1"], st["mb1"], st["mw2"]
        z = [W1[r][0] * x[0] + W1[r][1] * x[1] + b1[r] for r in range(2)]
        a = [math.tanh(v) for v in z]
        out = w2[0] * a[0] + w2[1] * a[1] + b2
        d = out - y
        losses.append(d * d)
        go = 2.0 * d
        da = [go * w2[0], go * w2[1]]
        dz = [da[r] * (1.0 - a[r] * a[r]) for r in range(2)]
        for r in range(2):
            for c in range(2):
                mW1[r][c] = 0.9 * mW1[r][c] + dz[r] * x[c]
                W1[r][c] = W1[r][c] - lr * mW1[r][c]
            mb1[r] = 0.9 * mb1[r] + dz[r]
            b1[r] = b1[r] - lr * mb1[r]
        g2 = [go * a[0], go * a[1]]
        for c in range(2):
            mw2[c] = 0.9 * mw2[c] + g2[c]
            w2[c] = w2[c] - lr * mw2[c]
        st["mb2"] = 0.9 * st["mb2"] + go
        st["b2"] = b2 - lr * st["mb2"]
    return losses


def _oracle():
    pairs_clean = _draw_plan(with_junk=False)
    st = _fresh_state()
    head = _run_steps(pairs_clean, st, 0, 3)
    st_at_save = copy.deepcopy(st)
    tail = _run_steps(pairs_clean, st, 3, 6)
    losses_full = head + tail

    pairs_shift = _draw_plan(with_junk=True)
    stb = _fresh_state()
    head_b = _run_steps(pairs_shift, stb, 0, 3)
    tail_b = _run_steps(pairs_shift, copy.deepcopy(st_at_save), 3, 6)
    losses_no_rng = head_b + tail_b
    return losses_full, losses_no_rng


ORACLE_FULL, ORACLE_NO_RNG = _oracle()

# 参考解实跑 dump（float32 loss.item()）
EX_FULL = [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
           0.0066848150454461575, 0.05954299122095108, 0.5288273692131042]
EX_NO_RNG = [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
             1.0938212871551514, 0.1766050010919571, 1.190037727355957]


def check_example(solve):
    out = solve()
    assert torch.allclose(torch.tensor(out["losses_full"]), torch.tensor(EX_FULL),
                          atol=1e-5, rtol=1e-5), f"losses_full 不符: {out['losses_full']}"
    assert out["losses_resumed"] == out["losses_full"], "恢复轨迹必须逐位一致"
    assert torch.allclose(torch.tensor(out["losses_no_rng"]), torch.tensor(EX_NO_RNG),
                          atol=1e-5, rtol=1e-5), f"losses_no_rng 不符: {out['losses_no_rng']}"
    assert out["resumed_bitwise_identical"] is True
    assert out["no_rng_tail_diverges"] is True
    assert out["sched_last_epoch_resumed"] == 6


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    for key, exp in (("losses_full", ORACLE_FULL), ("losses_no_rng", ORACLE_NO_RNG)):
        got = torch.tensor(out[key])
        assert torch.allclose(got, torch.tensor(exp), atol=1e-5, rtol=1e-5), \
            f"{key} 与纯 Python 重算不符: {out[key]} vs {exp}"
    # 续跑账必须与黄金轨迹逐位（Python float 列表 ==），不只是 allclose
    assert out["losses_resumed"] == out["losses_full"]


def check_diverge_only_in_tail(solve):
    out = solve()
    assert len(out["losses_no_rng"]) == 6
    assert out["losses_no_rng"][:3] == out["losses_full"][:3], \
        "存档前的 3 步走同一条随机流，必须逐位相同"
    for a, b in zip(out["losses_no_rng"][3:], out["losses_full"][3:]):
        assert abs(a - b) > 1e-3, "漏恢复 RNG 后尾部每一步数据都错位"
    assert out["no_rng_tail_diverges"] is True


def check_resume_ledgers(solve):
    out = solve()
    assert out["resumed_bitwise_identical"] is True
    assert out["momentum_restored"] is True, "动量 buffer 必须来自 ckpt 而非重跑污染"
    assert out["final_weight_identical"] is True
    # oracle 的续跑尾部只有在「第 3 步动量正确恢复」前提下才与解一致：
    # 若解的 ckpt 被两次 load_state_dict 原地共享污染，losses_no_rng 会偏离
    assert torch.allclose(torch.tensor(out["losses_no_rng"][3:]),
                          torch.tensor(ORACLE_NO_RNG[3:]), atol=1e-5, rtol=1e-5)


def check_losses_sane(solve):
    out = solve()
    vals = out["losses_full"] + out["losses_resumed"] + out["losses_no_rng"]
    assert len(out["losses_full"]) == len(out["losses_resumed"]) == len(out["losses_no_rng"]) == 6
    assert all(math.isfinite(v) and v > 0.0 for v in vals), "平方损失必须有限且为正"
