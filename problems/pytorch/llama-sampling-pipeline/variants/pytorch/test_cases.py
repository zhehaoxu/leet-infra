import math

import torch


def _softmax_py(lg, alive):
    """存活候选上的 softmax（纯 Python，返回整长列表，死位为 0）。"""
    v = len(lg)
    out = [0.0] * v
    cc = [i for i in range(v) if alive[i]]
    if not cc:
        return out
    mx = max(lg[i] for i in cc)
    ex = {i: math.exp(lg[i] - mx) for i in cc}
    tot = math.fsum(ex.values())
    for i in cc:
        out[i] = ex[i] / tot
    return out


def _oracle(logits, history, bias_tokens, bias_values, banned,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            penalty_repeat, penalty_freq, penalty_present, penalty_last_n,
            xtc_probability, xtc_threshold, min_keep, top_k, top_p, min_p,
            temp, temp_first, u_xtc, u):
    """独立重算：纯 Python 列表逐步模拟 + 朴素最长重复匹配（不用 Z-algorithm）。"""
    v = int(logits.shape[0])
    lg = [float(x) for x in logits.tolist()]
    alive = [True] * v
    masks = [[False] * v for _ in range(8)]
    mk = int(min_keep)

    def snap(row):
        for i in range(v):
            masks[row][i] = alive[i]

    def order():
        return sorted([i for i in range(v) if alive[i]], key=lambda i: (-lg[i], i))

    def restrict(keep):
        kset = set(keep)
        return [i in kset for i in range(v)]

    # ---- 0. logit-bias（逐条累加）+ banned（强制掩码）----
    for t, b in zip(list(bias_tokens), list(bias_values)):
        tid = int(t)
        if 0 <= tid < v:
            lg[tid] += float(b)
    for t in list(banned):
        tid = int(t)
        if 0 <= tid < v:
            alive[tid] = False
    snap(0)

    # ---- 1. penalties（窗口内计数；<=0 乘、>0 除，再减 count*freq + present）----
    ln = int(penalty_last_n)
    win = [int(x) for x in history] if ln < 0 else ([] if ln == 0 else [int(x) for x in history][-ln:])
    counts = {}
    for t in win:
        counts[t] = counts.get(t, 0) + 1
    if not (float(penalty_repeat) == 1.0 and float(penalty_freq) == 0.0
            and float(penalty_present) == 0.0):
        for tid, c in counts.items():
            if 0 <= tid < v and alive[tid] and c > 0:
                x = lg[tid]
                x = x * float(penalty_repeat) if x <= 0.0 else x / float(penalty_repeat)
                lg[tid] = x - c * float(penalty_freq) - float(penalty_present)
    snap(1)

    # ---- 2. dry：对每个偏移 k 朴素求「rev 与 rev[k:] 的最长公共前缀」（与 Z-algorithm 等价，
    #      但对齐 dry-sampling 已接受口径：重复段必须**贴着窗口末尾**，故在逆序窗口上匹配）----
    al = int(dry_allowed_length)
    ln2 = int(dry_penalty_last_n)
    last = [int(x) for x in history] if ln2 < 0 else ([] if ln2 == 0 else [int(x) for x in history][-ln2:])
    n = len(last)
    if float(dry_multiplier) != 0.0 and float(dry_base) >= 1.0 and ln2 != 0 and n > al:
        rev = last[::-1]
        best = {}
        for k in range(1, n):
            c = 0
            while k + c < n and rev[c] == rev[k + c]:
                c += 1
            if c >= al:
                tok = last[n - k]
                # 与已接受的 dry-sampling 口径一致：只登记严格更长的重复（长度 0 不登记，
                # 故 dry_allowed_length=0 时「零长匹配」不会被罚，见 notes 的口径说明）
                if c > best.get(tok, 0):
                    best[tok] = c
        for tok, rl in best.items():
            if 0 <= tok < v and alive[tok]:
                lg[tok] -= float(dry_multiplier) * float(dry_base) ** (rl - al)
    snap(2)

    # ---- 7. temp（时机由 temp_first 决定；<=0 只留最大 logit）----
    def apply_temp():
        nonlocal alive
        if float(temp) > 0.0:
            for i in order():
                lg[i] = lg[i] / float(temp)
        else:
            cc = order()
            if cc:
                alive = restrict(cc[:1])
        snap(7)

    if bool(temp_first):
        apply_temp()

    # ---- 3. top-k ----
    if int(top_k) > 0:
        cc = order()
        if len(cc) > int(top_k):
            alive = restrict(cc[:int(top_k)])
    snap(3)

    # ---- 4. top-p：降序累积首次 >= p（且已保留数 >= min_keep）处截断 ----
    if float(top_p) < 1.0:
        cc = order()
        if len(cc) > 1:
            p = _softmax_py(lg, alive)
            cum = 0.0
            cut = len(cc)
            for i, t in enumerate(cc):
                cum += p[t]
                if cum >= float(top_p) and i + 1 >= mk:
                    cut = i + 1
                    break
            alive = restrict(cc[:cut])
    snap(4)

    # ---- 5. min-p：logit 域地板 = 最大 logit + ln(p) ----
    if float(min_p) > 0.0:
        cc = order()
        if cc:
            thr = lg[cc[0]] + math.log(float(min_p))
            i = 1
            while i < len(cc):
                if lg[cc[i]] < thr and i >= mk:
                    break
                i += 1
            alive = restrict(cc[:i])
    snap(5)

    # ---- 6. xtc：达标头部前缀（除最后一个）被砍，尾部保留 ----
    if float(xtc_probability) > 0.0 and float(xtc_threshold) <= 0.5:
        cc = order()
        if len(cc) >= 2 and float(u_xtc) <= float(xtc_probability):
            p = _softmax_py(lg, alive)
            pos_last = 0
            for i, t in enumerate(cc):
                if p[t] >= float(xtc_threshold):
                    pos_last = i
                else:
                    break
            if pos_last > 0 and len(cc) - pos_last >= mk:
                alive = restrict(cc[pos_last:])
    snap(6)

    if not bool(temp_first):
        apply_temp()

    p = _softmax_py(lg, alive)
    acc, c = 0.0, 0
    uv = float(u.reshape(-1)[0]) if u.dim() else float(u)
    for i, x in enumerate(p):
        acc += x
        if acc >= uv:
            c = i
            break
    else:
        cc = order()
        c = cc[-1] if cc else 0
    return (torch.tensor(masks, dtype=torch.bool), torch.tensor(p),
            torch.tensor(c, dtype=torch.long))


def _kw(**over):
    d = dict(history=[3, 4, 5, 3, 4, 5],
             bias_tokens=torch.tensor([2]), bias_values=torch.tensor([0.7]),
             banned=torch.tensor([0]),
             dry_multiplier=0.8, dry_base=1.4, dry_allowed_length=2,
             dry_penalty_last_n=-1,
             penalty_repeat=1.1, penalty_freq=0.2, penalty_present=0.1,
             penalty_last_n=-1, xtc_probability=0.0, xtc_threshold=0.05,
             min_keep=1, top_k=8, top_p=0.6, min_p=0.0, temp=0.5, temp_first=False)
    d.update(over)
    return d


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    masks, probs, chosen = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.55), **_kw())
    assert masks.shape == (8, 10)
    # temp_first=False ⇒ 第 7 行（temp）在全部截断之后执行；temp=0.5>0 不删候选，故与第 6 行同集合。
    assert [int(masks[i].sum()) for i in range(8)] == [9, 9, 9, 8, 2, 2, 2, 2]
    exp = torch.tensor([0.0, 0.645656, 0.354344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(probs, exp, atol=1e-5, rtol=1e-5), probs.tolist()
    assert chosen.item() == 1, chosen.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for trial in range(10):
        v = 14
        logits = torch.randn(v) * 3
        hist = torch.randint(0, v, (12,)).tolist()
        for temp_first in (False, True):
            for xtc_probability in (0.0, 1.0):
                for temp in (0.5, 1.0, 0.0):
                    for top_p in (0.6, 0.95, 1.0):
                        for min_p in (0.0, 0.1):
                            kw = _kw(history=hist, temp_first=temp_first,
                                     bias_tokens=torch.tensor([2, 5, 99]),
                                     bias_values=torch.tensor([0.7, -1.0, 5.0]),
                                     banned=torch.tensor([0, 13]),
                                     penalty_last_n=6, xtc_probability=xtc_probability,
                                     top_p=top_p, min_p=min_p, temp=temp)
                            got = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.3), **kw)
                            exp = _oracle(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.3), **kw)
                            tag = (trial, temp_first, xtc_probability, temp, top_p, min_p)
                            assert torch.equal(got[0], exp[0]), tag
                            assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-5), tag
                            assert torch.equal(got[2], exp[2]), tag


def check_hf_vs_llama_temperature_order(solve):
    # 专项断言：同参数下 llama.cpp 口径（temp 在截断之后）与 HF 口径（temp 在截断之前）的
    # top-p 存活集合不同 —— llama.cpp 的截断看到的是**未过温度**的概率，高温不放大截断。
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    m_l, p_l, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.55), **_kw())
    m_h, p_h, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.55),
                        **_kw(temp_first=True))
    assert m_l[4].tolist() == [False, True, True] + [False] * 7, m_l[4].tolist()
    assert m_h[4].tolist() == [False, True] + [False] * 8, m_h[4].tolist()
    assert not torch.equal(m_l[4], m_h[4])
    assert bool(((m_l[7] & m_l[6]) == m_l[7]).all())        # llama：temp 是最后一个截断
    # HF 口径：temp 先执行（第 7 行快照 = 全部 9 个存活），随后 top-p 在**已过温度**的
    # 概率上累积，直接塌到单 token，于是后续 min-p / xtc 都无候选可删 ⇒ 4/5/6 三行相同。
    assert int(m_h[7].sum()) == 9, int(m_h[7].sum())
    assert torch.equal(m_h[4], m_h[5]) and torch.equal(m_h[5], m_h[6])
    assert abs(float(p_l.sum()) - 1.0) < 1e-6 and abs(float(p_h.sum()) - 1.0) < 1e-6
    assert not torch.allclose(p_l, p_h, atol=1e-6)


def check_all_disabled_is_plain_softmax(solve):
    # 所有旋钮关掉 → 每一步都不删候选、不改 logit，最终分布就是原 logits 的 softmax
    torch.manual_seed(1)
    logits = torch.randn(9)
    off = _kw(history=[0, 1, 2], bias_tokens=torch.tensor([], dtype=torch.long),
              bias_values=torch.tensor([]), banned=torch.tensor([], dtype=torch.long),
              dry_multiplier=0.0, dry_base=1.0, dry_allowed_length=4,
              dry_penalty_last_n=0, penalty_repeat=1.0, penalty_freq=0.0,
              penalty_present=0.0, penalty_last_n=0, xtc_probability=0.0,
              xtc_threshold=0.1, min_keep=0, top_k=0, top_p=1.0, min_p=0.0,
              temp=1.0)
    for tf in (False, True):
        off["temp_first"] = tf          # 就地覆盖，避免向 _kw 重复传同名关键字
        masks, probs, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.2),
                                **off)
        assert bool(masks.all()), masks
        assert torch.allclose(probs, torch.softmax(logits, dim=-1), atol=1e-6)


def check_temp_zero_greedy_and_nested(solve):
    # temp=0 → temp 步只留 logit 最大的候选（llama.cpp 把其余置 -INF），dist 退化成贪心；
    # 顺带校验存活集合沿执行序逐阶段嵌套（截断只删不加）。
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    masks, probs, chosen = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.55),
                                 **_kw(temp=0.0))
    assert [int(masks[i].sum()) for i in range(8)] == [9, 9, 9, 8, 2, 2, 2, 1]
    assert chosen.item() == 1 and probs[1].item() == 1.0, probs.tolist()
    for a, b in zip(range(7), range(1, 8)):
        assert bool(((masks[b] & masks[a]) == masks[b]).all()), (a, b)


def check_banned_and_bias_effects(solve):
    # ban 掉的 token 从第 0 行起恒为 False、最终概率恰为 0；bias 只改 logit、不删候选，
    # 且同一 token 出现两次则累加（+2 两次 == +4）。
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    kw = _kw(top_k=0, top_p=1.0, min_p=0.0, temp=1.0, banned=torch.tensor([1, 2]),
             bias_tokens=torch.tensor([3, 3]), bias_values=torch.tensor([2.0, 2.0]),
             history=[], penalty_repeat=1.0, penalty_freq=0.0, penalty_present=0.0,
             penalty_last_n=0, dry_multiplier=0.0, xtc_probability=0.0)
    masks, probs, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.5), **kw)
    assert bool(masks[:, 1].any()) is False and bool(masks[:, 2].any()) is False
    assert probs[1].item() == 0.0 and probs[2].item() == 0.0
    assert int(masks[0].sum()) == 8 and bool(masks.all(dim=0)[3]) is True
    ref = torch.tensor([5.0, 4.0, 3.0, 6.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    r = ref.clone()
    r[[1, 2]] = -float("inf")
    assert torch.allclose(probs, torch.softmax(r, dim=-1), atol=1e-6), probs.tolist()


def check_dry_penalizes_repeat_follower(solve):
    # DRY 只改 logit、不删候选（penalties 行与 dry 行存活集合相同），但循环的「下一个 token」
    # 概率被显著压低（history=[1,2,3,1,2,3] → 延续循环的是 last[n-k]=1）
    logits = torch.full((6,), 2.0)
    kw = _kw(history=[1, 2, 3, 1, 2, 3], bias_tokens=torch.tensor([], dtype=torch.long),
             bias_values=torch.tensor([]), banned=torch.tensor([], dtype=torch.long),
             penalty_repeat=1.0, penalty_freq=0.0, penalty_present=0.0, penalty_last_n=0,
             xtc_probability=0.0, min_keep=0, top_k=0, top_p=1.0, min_p=0.0, temp=1.0)
    off = dict(kw, dry_multiplier=0.0)
    _, p_off, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.5), **off)
    masks, p_on, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.5), **kw)
    assert bool((masks[1] == masks[2]).all()) is True
    assert abs(float(p_off[1]) - 1.0 / 6) < 1e-6, p_off.tolist()
    assert p_on[1].item() < p_off[1].item(), (p_on.tolist(), p_off.tolist())
    assert int(torch.argmin(p_on)) == 1, p_on.tolist()
    # 退化探针：把 dry 窗口收到比重复段更短（2 < 3）→ 无重复可罚，分布退回 dry 关闭时的样子
    short = dict(kw, dry_penalty_last_n=2)
    m_s, p_s, _ = solve(logits, u_xtc=torch.tensor(0.5), u=torch.tensor(0.5), **short)
    assert bool(m_s.all()), m_s                      # 无删候选，8 行全 True
    assert torch.equal(m_s[1], m_s[2])
    assert torch.allclose(p_s, p_off, atol=1e-6), (p_s.tolist(), p_off.tolist())


def check_empty_vocab_min_keep_and_xtc_gate(solve):
    # 退化/不变量探针三则，全部同时校验 solution 与独立 oracle 的口径一致。
    clean = dict(bias_tokens=torch.tensor([], dtype=torch.long), bias_values=torch.tensor([]),
                 banned=torch.tensor([], dtype=torch.long), dry_multiplier=0.0,
                 dry_penalty_last_n=0, penalty_repeat=1.0, penalty_freq=0.0,
                 penalty_present=0.0, penalty_last_n=0, xtc_probability=0.0)

    # ① 整个词表都被 ban：存活集合恒空 ⇒ 8 行全 False、最终分布全 0、chosen 兜底 0
    lg0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    kw0 = dict(_kw(history=[0, 1]), top_k=2, top_p=0.5, min_p=0.2, temp=0.7)
    kw0.update(bias_tokens=torch.tensor([0]), bias_values=torch.tensor([1.0]))
    kw0["banned"] = torch.tensor([0, 1, 2, 3, 99])   # 越界 id 无副作用
    m0, p0, c0 = solve(lg0, u_xtc=torch.tensor(0.5), u=torch.tensor(0.5), **kw0)
    e0 = _oracle(lg0, u_xtc=torch.tensor(0.5), u=torch.tensor(0.5), **kw0)
    assert not bool(m0.any()), m0
    assert torch.equal(m0, e0[0]) and torch.equal(c0, e0[2])
    assert float(p0.sum()) == 0.0 and int(c0) == 0, (p0.tolist(), int(c0))

    # ② min_keep 抬高截断下限（llama.cpp 的 `i + 1 >= min_keep` / `i >= min_keep` 判据）
    lg1 = torch.tensor([9.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    for mk in (0, 1, 3, 6, 9):
        for tf in (False, True):
            kw1 = dict(_kw(history=[], **clean))
            kw1.update(min_keep=mk, top_k=5, top_p=0.5, min_p=0.1, temp=0.8,
                       xtc_probability=1.0, xtc_threshold=0.05, temp_first=tf)
            g = solve(lg1, u_xtc=torch.tensor(0.5), u=torch.tensor(0.4), **kw1)
            e = _oracle(lg1, u_xtc=torch.tensor(0.5), u=torch.tensor(0.4), **kw1)
            assert torch.equal(g[0], e[0]), (mk, tf, g[0].tolist(), e[0].tolist())
            assert torch.allclose(g[1], e[1], atol=1e-5, rtol=1e-5), (mk, tf)
            assert torch.equal(g[2], e[2]), (mk, tf)
    # 同一份 logits：min_keep=1 时 top-p 首元素就够 0.5 ⇒ 只留 1 个；min_keep=3 ⇒ 至少留 3 个
    k_a = dict(_kw(history=[], **clean)); k_a.update(min_keep=1, top_k=5, top_p=0.5, min_p=0.1, temp=0.8)
    k_b = dict(_kw(history=[], **clean)); k_b.update(min_keep=3, top_k=5, top_p=0.5, min_p=0.1, temp=0.8)
    m_a, _, _ = solve(lg1, u_xtc=torch.tensor(0.5), u=torch.tensor(0.4), **k_a)
    m_b, _, _ = solve(lg1, u_xtc=torch.tensor(0.5), u=torch.tensor(0.4), **k_b)
    assert int(m_a[4].sum()) == 1 and int(m_b[4].sum()) == 3, (m_a[4].tolist(), m_b[4].tolist())
    # min_keep=6 已超过 top-k 后的候选数 5 ⇒ `i + 1 >= min_keep` 永不成立，后续截断全成 no-op；
    # min_keep 只是下限，不能复活被 top-k 删掉的候选。
    k_c = dict(_kw(history=[], **clean)); k_c.update(min_keep=6, top_k=5, top_p=0.5, min_p=0.1, temp=0.8)
    m_c, _, _ = solve(lg1, u_xtc=torch.tensor(0.5), u=torch.tensor(0.4), **k_c)
    assert int(m_c[3].sum()) == 5, m_c.tolist()
    assert torch.equal(m_c[4], m_c[3]) and torch.equal(m_c[5], m_c[3]) and torch.equal(m_c[6], m_c[3])

    # ③ xtc 的两个早退：概率门关（u_xtc > probability）与 threshold > 0.5 ⇒ 第 6 行 == 第 5 行
    lg2 = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    base2 = dict(_kw(history=[], **clean))
    base2.update(top_k=6, top_p=1.0, min_p=0.0, temp=1.0, min_keep=1)
    m_off, _, _ = solve(lg2, u_xtc=torch.tensor(0.9), u=torch.tensor(0.4),
                        **dict(base2, xtc_probability=0.0, xtc_threshold=0.05))
    # 门开着但 u_xtc 落在门外
    m_gate, _, _ = solve(lg2, u_xtc=torch.tensor(0.9), u=torch.tensor(0.4),
                         **dict(base2, xtc_probability=0.5, xtc_threshold=0.05))
    # 门开着（u_xtc=0.1 <= 1.0）但 threshold>0.5 ⇒ 早退
    m_thr, _, _ = solve(lg2, u_xtc=torch.tensor(0.1), u=torch.tensor(0.4),
                        **dict(base2, xtc_probability=1.0, xtc_threshold=0.7))
    # 真正触发一次，作为对照（砍掉头部前缀，尾部保留）
    m_fire, _, _ = solve(lg2, u_xtc=torch.tensor(0.1), u=torch.tensor(0.4),
                        **dict(base2, xtc_probability=1.0, xtc_threshold=0.05))
    assert torch.equal(m_off[5], m_off[6]) and torch.equal(m_gate[5], m_gate[6])
    assert torch.equal(m_thr[5], m_thr[6])
    assert not torch.equal(m_fire[5], m_fire[6]), m_fire.tolist()
    assert int(m_fire[6].sum()) < int(m_fire[5].sum())


check_example.example = True
