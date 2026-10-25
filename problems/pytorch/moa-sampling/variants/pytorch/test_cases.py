import math

import torch


def _softmax(xs):
    mx = max(xs)
    ex = [math.exp(x - mx) for x in xs]
    tot = math.fsum(ex)
    return [x / tot for x in ex]


def _oracle(logits, weights, temps, top_ps, m, u_experts, u):
    """独立重算：纯 Python 逐专家 softmax + 前缀扫描 + 手工逆变换，不使用任何张量排序/索引。"""
    e_n, v = int(logits.shape[0]), int(logits.shape[1])
    m = int(m)

    q = []
    for e in range(e_n):
        lg = [float(x) for x in logits[e].tolist()]
        t = float(temps[e])
        if t <= 0.0:
            best, bi = lg[0], 0
            for i in range(1, v):
                if lg[i] > best:
                    best, bi = lg[i], i
            p = [0.0] * v
            p[bi] = 1.0
        else:
            p = _softmax([x / t for x in lg])
        # top-p：降序（并列按索引升序）累积到首次 >= top_ps[e]，至少 1 个
        order = sorted(range(v), key=lambda i: (-p[i], i))
        sp = [p[i] for i in order]
        cum, cut = 0.0, 1
        for i in range(len(sp)):
            cum += sp[i]
            if cum >= float(top_ps[e]):
                cut = i + 1
                break
        else:
            cut = len(sp)
        kept = sp[:cut]
        s = math.fsum(kept)
        row = [0.0] * v
        for i in range(cut):
            row[order[i]] = kept[i] / s
        q.append(row)

    union = set()
    for e in range(e_n):
        for j in range(m):
            uv = float(u_experts[e][j])
            acc, c = 0.0, v - 1
            for i, x in enumerate(q[e]):
                acc += x
                if acc >= uv:
                    c = i
                    break
            union.add(c)
    u_idx = sorted(union)

    ws = [float(x) for x in weights.tolist()]
    sw = math.fsum(ws)
    ws = [x / sw for x in ws]

    mix = [0.0] * v
    for e in range(e_n):
        denom = math.fsum(q[e][t] for t in u_idx)
        if denom <= 0.0:
            continue
        for t in u_idx:
            mix[t] += ws[e] * q[e][t] / denom
    tot = math.fsum(mix[t] for t in u_idx)
    filtered = [0.0] * v
    if tot > 0.0:
        for t in u_idx:
            filtered[t] = mix[t] / tot

    acc, c = 0.0, v - 1
    uv = float(u.reshape(-1)[0]) if u.dim() else float(u)
    for i, x in enumerate(filtered):
        acc += x
        if acc >= uv:
            c = i
            break
    return (torch.tensor(filtered), torch.tensor(c, dtype=torch.long),
            torch.tensor([i in union for i in range(v)], dtype=torch.bool))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0],
                           [0.0, 1.0, 2.0, 1.0, 0.0]])
    filt, chosen, in_union = solve(
        logits, torch.tensor([0.7, 0.3]), torch.tensor([1.0, 0.8]),
        torch.tensor([0.9, 0.9]), 2,
        torch.tensor([[0.1, 0.6], [0.3, 0.95]]), torch.tensor(0.5))
    exp = torch.tensor([0.616558, 0.0, 0.316632, 0.066810, 0.0])
    assert torch.allclose(filt, exp, atol=1e-5, rtol=1e-5), filt.tolist()
    assert chosen.item() == 0, chosen.item()
    assert in_union.tolist() == [1, 0, 1, 1, 0], in_union.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(25):
        e_n = torch.randint(1, 4, ()).item()
        v = torch.randint(3, 10, ()).item()
        m = torch.randint(1, 4, ()).item()
        logits = torch.randn(e_n, v) * 1.5
        weights = torch.rand(e_n) + 0.05
        temps = torch.where(torch.rand(e_n) < 0.3, torch.zeros(e_n),
                            torch.rand(e_n) * 1.5 + 0.3)      # 混入 temp<=0 的贪心专家
        top_ps = torch.rand(e_n) * 0.9 + 0.1
        ue = torch.rand(e_n, m)
        u = torch.rand(1)
        got = solve(logits, weights, temps, top_ps, m, ue, u)
        exp = _oracle(logits, weights, temps, top_ps, m, ue, u)
        assert torch.allclose(got[0], exp[0], atol=1e-5, rtol=1e-5), (e_n, v, m)
        assert torch.equal(got[1], exp[1]), (e_n, v, m)
        assert torch.equal(got[2], exp[2]), (e_n, v, m)


def check_union_and_mass_invariants(solve):
    torch.manual_seed(1)
    for _ in range(20):
        e_n, v, m = 3, 9, 2
        logits = torch.randn(e_n, v)
        ue = torch.rand(e_n, m)
        filt, chosen, in_union = solve(logits, torch.full((e_n,), 1.0 / 3),
                                       torch.ones(e_n), torch.ones(e_n), m, ue,
                                       torch.rand(1))
        assert int(in_union.sum()) <= e_n * m, int(in_union.sum())
        assert float(filt[~in_union].abs().sum()) == 0.0          # U 外严格为 0
        assert abs(float(filt.sum()) - 1.0) < 1e-6, filt.sum().item()
        assert bool(in_union[int(chosen)]) is True                 # chosen 必在联合候选集内


def check_single_expert_one_sample_is_onehot(solve):
    # E=1, m=1：联合候选集只有 1 个 token ⇒ 混合分布退化成 one-hot，且必为该专家自己的抽样
    logits = torch.tensor([[3.0, 1.0, 0.5, -2.0]])
    for uv in (0.05, 0.4, 0.7, 0.95):
        filt, chosen, in_union = solve(logits, torch.tensor([1.0]), torch.tensor([1.0]),
                                       torch.tensor([1.0]), 1, torch.tensor([[uv]]),
                                       torch.tensor(0.5))
        assert int(in_union.sum()) == 1
        assert filt[int(chosen)].item() == 1.0, filt.tolist()


def check_zero_weight_expert_only_affects_union(solve):
    # 权重为 0 的专家仍贡献联合候选集（过采样发生在加权之前），但不贡献混合质量
    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    ue = torch.tensor([[0.1], [0.1]])
    filt, _, in_union = solve(logits, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]),
                              torch.tensor([1.0, 1.0]), 1, ue, torch.tensor(0.5))
    assert in_union.tolist() == [True, False, False, True], in_union.tolist()
    q1 = torch.softmax(logits[1], dim=-1)
    ref = q1 * in_union.to(q1.dtype)
    ref = ref / ref.sum()                       # 混合 = 有效专家在 U 上的再归一化
    assert torch.allclose(filt, ref, atol=1e-6), (filt.tolist(), ref.tolist())
    assert abs(float(filt[1]) + float(filt[2])) < 1e-12


def check_greedy_expert_temp_zero(solve):
    # temp<=0 的专家＝one-hot(argmax)（llama.cpp 的 temp 口径），多次过采样也只会抽到同一个 token
    logits = torch.tensor([[-1.0, 5.0, 0.0, 2.0]])
    ue = torch.tensor([[0.2, 0.6, 0.99]])
    filt, chosen, in_union = solve(logits, torch.tensor([1.0]), torch.tensor([0.0]),
                                   torch.tensor([1.0]), 3, ue, torch.tensor(0.5))
    assert in_union.tolist() == [0, 1, 0, 0], in_union.tolist()
    assert int(chosen) == 1 and filt[1].item() == 1.0, filt.tolist()

check_example.example = True
