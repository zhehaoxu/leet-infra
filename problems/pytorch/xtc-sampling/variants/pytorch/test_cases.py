import torch


def _oracle(logits, probability, threshold, min_keep, u_gate, u):
    """独立重算：纯 Python 排序 + 手工前缀扫描，不复用任何张量排序/散射。"""
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    flat = probs.reshape(-1, v)
    n = flat.shape[0]
    ug = u_gate.reshape(-1)
    uu = u.reshape(-1)
    out = torch.zeros_like(flat)
    applied = torch.zeros(n, dtype=torch.bool)
    chosen = []
    for r in range(n):
        p = [float(x) for x in flat[r]]
        keep = [1.0] * v
        fired = False
        if float(probability) > 0.0 and float(threshold) <= 0.5 and v >= 2:
            order = sorted(range(v), key=lambda i: (-p[i], i))
            cnt = 0
            for t in order:
                if p[t] >= float(threshold):
                    cnt += 1
                else:
                    break
            pos_last = cnt - 1
            if float(ug[min(r, ug.numel() - 1)]) <= float(probability) \
                    and pos_last > 0 and v - pos_last >= int(min_keep):
                for t in order[:pos_last]:
                    keep[t] = 0.0
                fired = True
        applied[r] = fired
        tot = sum(p[i] * keep[i] for i in range(v))
        row = [p[i] * keep[i] / tot for i in range(v)]
        acc, c = 0.0, v - 1
        uv = float(uu[min(r, uu.numel() - 1)])
        for i, x in enumerate(row):
            acc += x
            if acc >= uv:
                c = i
                break
        out[r] = torch.tensor(row)
        chosen.append(c)
    return (out.reshape(probs.shape),
            torch.tensor(chosen).reshape(probs.shape[:-1]),
            applied.reshape(probs.shape[:-1]))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([3.0, 2.5, 2.0, 0.0, -1.0, -2.0])
    # probs=[0.488,0.296,0.180,0.024,0.009,0.003]，threshold=0.05 → 降序前缀 3 个达标
    # pos_last=2 > 0 → 剔除秩 0,1（token 0,1），保留 token 2 + 尾部并归一化
    filtered, chosen, applied = solve(logits, 1.0, 0.05, 0, torch.tensor(0.5), torch.tensor(0.4))
    exp = torch.tensor([0.0, 0.0, 0.8310, 0.1125, 0.0414, 0.0152])
    assert torch.allclose(filtered, exp, atol=1e-4), filtered.tolist()
    assert chosen.item() == 2, chosen.item()
    assert bool(applied) is True


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        logits = torch.randn(5, 12)
        for probability in (0.0, 0.5, 1.0):
            for threshold in (0.01, 0.05, 0.3):
                for min_keep in (0, 3, 8):
                    ug = torch.rand(5)
                    u = torch.rand(5)
                    got = solve(logits, probability, threshold, min_keep, ug, u)
                    exp = _oracle(logits, probability, threshold, min_keep, ug, u)
                    assert torch.allclose(got[0], exp[0], atol=1e-5, rtol=1e-5), (probability, threshold, min_keep)
                    assert torch.equal(got[1], exp[1]), (probability, threshold, min_keep)
                    assert torch.equal(got[2], exp[2]), (probability, threshold, min_keep)


def check_gate_off_is_identity(solve):
    # u_gate > probability → 剔除不触发，filtered 就是原始 softmax，applied=False
    torch.manual_seed(1)
    logits = torch.randn(6)
    probs = torch.softmax(logits, dim=-1)
    f, c, ap = solve(logits, 0.5, 0.05, 0, torch.tensor(0.9), torch.tensor(0.2))
    assert bool(ap) is False
    assert torch.allclose(f, probs, atol=1e-6), f.tolist()
    assert c.item() == 0, c.item()


def check_disabled_params(solve):
    torch.manual_seed(2)
    logits = torch.randn(4, 7)
    probs = torch.softmax(logits, dim=-1)
    ug = torch.rand(4)
    u = torch.rand(4)
    f, _, ap = solve(logits, 0.0, 0.05, 0, ug, u)        # probability<=0
    assert bool(ap.any()) is False and torch.allclose(f, probs, atol=1e-6)
    f, _, ap = solve(logits, 1.0, 0.6, 0, ug, u)          # threshold>0.5 → 早退
    assert bool(ap.any()) is False and torch.allclose(f, probs, atol=1e-6)
    f1, _, ap1 = solve(torch.zeros(1, 1), 1.0, 0.01, 0,   # 候选数 < 2 → 早退
                       torch.tensor([[0.1]]), torch.tensor([[0.1]]))
    assert bool(ap1.any()) is False
    assert torch.allclose(f1, torch.ones(1, 1), atol=1e-6)


def check_only_one_above_threshold_no_cut(solve):
    # 只有最高概率项过阈值 → pos_last=0 → 不砍（XTC 需要「至少两个头部候选」才动刀）
    logits = torch.tensor([9.0, -1.0, -2.0, -3.0])
    f, _, ap = solve(logits, 1.0, 0.5, 0, torch.tensor(0.1), torch.tensor(0.1))
    assert bool(ap) is False, ap
    assert f[0].item() > 0.9, f.tolist()


def check_min_keep_blocks_cut(solve):
    # 尾部存活数不足 min_keep → 不砍
    logits = torch.tensor([3.0, 2.5, 2.0, 0.0, -1.0, -2.0])
    _, _, ap_small = solve(logits, 1.0, 0.05, 5, torch.tensor(0.5), torch.tensor(0.4))
    assert bool(ap_small) is False, ap_small          # 存活 4 < 5
    _, _, ap_ok = solve(logits, 1.0, 0.05, 4, torch.tensor(0.5), torch.tensor(0.4))
    assert bool(ap_ok) is True, ap_ok


def check_cuts_head_not_tail(solve):
    # 与 top-p/min-p 相反：被清零的是**头部**（概率最大的候选），尾部仍保留质量
    logits = torch.tensor([3.0, 2.5, 2.0, 0.0, -1.0, -2.0])
    f, _, ap = solve(logits, 1.0, 0.05, 0, torch.tensor(0.5), torch.tensor(0.4))
    assert bool(ap) is True
    assert f[0].item() == 0.0 and f[1].item() == 0.0, f.tolist()
    assert f[3].item() > 0.0 and f[5].item() > 0.0, f.tolist()


check_example.example = True
