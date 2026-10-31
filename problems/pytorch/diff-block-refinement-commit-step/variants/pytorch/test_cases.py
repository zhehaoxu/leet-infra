import math

import torch


NEG_INF = float("-inf")


def _t(nested, dtype=torch.float32):
    return torch.tensor(nested, dtype=dtype)


def _f32(v):
    """把标量降到 float32 再升回来——参考实现里比较发生在这个精度上（阈值也一样）。"""
    return float(torch.tensor(float(v), dtype=torch.float32).item())


def _flat(x):
    return x.tolist() if isinstance(x, torch.Tensor) else x


def call_mask(solve, tokens, probs, sample, t, N, mask_id=9, th=0.9, et=None, pm=None):
    prompt = None if pm is None else _t(pm, torch.bool)
    return solve(_t(tokens, torch.int64), _t(probs), _t(sample, torch.int64), t, N,
                 mask_id, float(th), et, prompt, None, None)


def call_uniform(solve, tokens, probs, sample, t, N, th=0.9, et=None, committed=None, rnd=None, pm=None):
    prompt = None if pm is None else _t(pm, torch.bool)
    return solve(_t(tokens, torch.int64), _t(probs), _t(sample, torch.int64), t, N,
                 None, float(th), et, prompt, _t(committed, torch.bool), _t(rnd, torch.int64))


def oracle(tokens, probs, sample, t, N, mask_id, threshold, editing_threshold, prompt_mask,
           committed, random_tokens):
    """独立 oracle：纯 Python 的名次/集合推演，不借任何 torch 算子。"""
    rows, length = len(sample), len(sample[0])
    p = [[_f32(probs[r][i]) for i in range(length)] for r in range(rows)]
    threshold = _f32(threshold)
    if editing_threshold is not None:
        editing_threshold = _f32(editing_threshold)
    transfer, editing, prev = [], [], []

    if mask_id is None:
        if t == 0:
            base = [[False] * length for _ in range(rows)]
        else:
            base = [[bool(v) for v in committed[r]] for r in range(rows)]
        target = ((t + 1) * length + N - 1) // N
        for r in range(rows):
            keyed = [p[r][i] if not base[r][i] else NEG_INF for i in range(length)]
            order = sorted(range(length), key=lambda i: (-keyed[i], i))
            rank = [0] * length
            for pos, i in enumerate(order):
                rank[i] = pos
            needed = max(0, target - sum(base[r]))
            tr = [(not base[r][i]) and (rank[i] < needed or p[r][i] > threshold) for i in range(length)]
            ed = []
            for i in range(length):
                if editing_threshold is None:
                    ed.append(False)
                else:
                    ed.append(base[r][i] and tokens[r][i] != sample[r][i]
                              and p[r][i] > editing_threshold)
            after = [base[r][i] or tr[i] for i in range(length)]
            row_prev = []
            for i in range(length):
                if tr[i] or ed[i]:
                    row_prev.append(tokens[r][i])
                elif after[i]:
                    row_prev.append(sample[r][i])
                else:
                    row_prev.append(random_tokens[r][i])
            transfer.append(tr)
            editing.append(ed)
            prev.append(row_prev)
        return prev, transfer, editing

    any_mask = any(sample[r][i] == mask_id for r in range(rows) for i in range(length))
    base, rem = divmod(length, N)
    schedule = [base + 1] * rem + [base] * (N - rem)
    quota = schedule[min(t, N - 1)]
    pm = [False] * length if prompt_mask is None else [bool(v) for v in prompt_mask]
    editing_on = editing_threshold is not None and _f32(editing_threshold) > 0.0
    for r in range(rows):
        active = [sample[r][i] == mask_id for i in range(length)]
        keyed = [p[r][i] if active[i] else NEG_INF for i in range(length)]
        tr = [False] * length
        if any_mask:
            over = [i for i in range(length) if keyed[i] > threshold]
            if len(over) >= quota:
                for i in over:
                    tr[i] = True
            else:
                k = min(quota, sum(active))
                order = sorted(range(length), key=lambda i: (-keyed[i], i))
                for i in order[:k]:
                    tr[i] = True
        ed = [False] * length
        if editing_on:
            for i in range(length):
                ed[i] = (not active[i]) and (not pm[i]) and p[r][i] > editing_threshold \
                    and tokens[r][i] != sample[r][i]
        prev.append([tokens[r][i] if (tr[i] or ed[i]) else sample[r][i] for i in range(length)])
        transfer.append(tr)
        editing.append(ed)
    return prev, transfer, editing


def _as_lists(out):
    prev, tr, ed = out
    return _flat(prev), _flat(tr), _flat(ed)


def _assert_match(solve, args, tag):
    got = _as_lists(solve(*args))
    want = [list(map(list, _as_lists(oracle(*args))[k])) for k in range(3)]
    assert got == want, (tag, got, want)


# ---------------------------------------------------------------- 示例

EX_MASK = dict(
    tokens=[[4, 8, 1, 6, 2, 0], [0, 5, 2, 7, 3, 8]],
    probs=[[0.93, 0.40, 0.99, 0.61, 0.88, 0.12], [0.10, 0.93, 0.30, 0.95, 0.91, 0.05]],
    sample=[[9, 9, 3, 9, 7, 9], [9, 9, 9, 9, 9, 9]],
    pm=[False, False, True, False, False, False],
)

EX_UNIFORM = dict(
    tokens=[[1, 9, 3, 6, 8, 2], [5, 7, 4, 0, 9, 3]],
    probs=[[0.95, 0.20, 0.88, 0.61, 0.30, 0.72], [0.10, 0.55, 0.70, 0.25, 0.92, 0.40]],
    sample=[[7, 2, 9, 4, 5, 1], [6, 3, 0, 8, 2, 4]],
    committed=[[True, True, False, False, True, False], [False, False, False, False, False, False]],
    rnd=[[1, 8, 6, 2, 3, 9], [7, 1, 5, 8, 0, 6]],
)


def check_example(solve):
    prev, tr, ed = call_mask(solve, EX_MASK["tokens"], EX_MASK["probs"], EX_MASK["sample"],
                             1, 4, mask_id=9, th=0.9, et=0.6, pm=EX_MASK["pm"])
    assert _flat(prev) == [[4, 9, 3, 6, 2, 9], [9, 5, 9, 7, 3, 9]], _flat(prev)
    assert _flat(tr) == [[True, False, False, True, False, False],
                         [False, True, False, True, True, False]], _flat(tr)
    assert _flat(ed) == [[False, False, False, False, True, False],
                         [False, False, False, False, False, False]], _flat(ed)

    prev, tr, ed = call_uniform(solve, EX_UNIFORM["tokens"], EX_UNIFORM["probs"], EX_UNIFORM["sample"],
                                2, 4, th=0.9, et=0.5, committed=EX_UNIFORM["committed"], rnd=EX_UNIFORM["rnd"])
    assert _flat(prev) == [[1, 2, 3, 2, 5, 2], [7, 7, 4, 0, 9, 3]], _flat(prev)
    assert _flat(tr) == [[False, False, True, False, False, True],
                         [False, True, True, True, True, True]], _flat(tr)
    assert _flat(ed) == [[True, False, False, False, False, False],
                         [False, False, False, False, False, False]], _flat(ed)


check_example.example = True


# ---------------------------------------------------------------- 两种配额口径

def check_two_quota_regimes_differ(solve):
    """L=10、N=6：掩码模式每步定量 [2,2,2,2,1,1]，均匀模式累计 [2,4,5,7,9,10]。"""
    L, N = 10, 6
    assert [L // N + 1] * (L % N) + [L // N] * (N - L % N) == [2, 2, 2, 2, 1, 1]
    assert [((k + 1) * L + N - 1) // N for k in range(N)] == [2, 4, 5, 7, 9, 10]
    probs = [[0.5] * L]
    sample = [[9] * L]
    tokens = [[3] * L]
    counts = []
    for t in range(N):
        counts.append(int(call_mask(solve, tokens, probs, sample, t, N, mask_id=9, th=0.95)[1].sum(-1)[0]))
    assert counts == [2, 2, 2, 2, 1, 1], counts

    committed = [[False] * L]
    rnd = [[7] * L]
    cum = []
    for t in range(N):
        _, tr, _ = call_uniform(solve, tokens, probs, committed and sample, t, N,
                                th=0.95, et=None, committed=committed, rnd=rnd)
        committed = [[bool(a or b) for a, b in zip(committed[0], tr.tolist()[0])]]
        cum.append(sum(sum(row) for row in committed))
    assert cum == [2, 4, 5, 7, 9, 10], cum


def check_uniform_target_is_not_schedule_prefix_sum(solve):
    """把累计目标写成每步表的前缀和：L=10、N=6 时第 3、4 步各多要一个。"""
    L, N = 10, 6
    prefix = [2, 4, 6, 8, 9, 10]
    probs = [[0.5] * L]
    sample = [[1] * L]
    tokens = [[3] * L]
    rnd = [[7] * L]
    committed = [[False] * L]
    for t in range(N):
        _, tr, _ = call_uniform(solve, tokens, probs, sample, t, N, th=0.95, committed=committed, rnd=rnd)
        committed = [[bool(a or b) for a, b in zip(committed[0], tr.tolist()[0])]]
        total = sum(sum(row) for row in committed)
        assert total == [((k + 1) * L + N - 1) // N for k in range(N)][t], (t, total)
        assert total != prefix[t] or t in (0, 1, 4, 5), (t, total, prefix)


# ---------------------------------------------------------------- 掩码模式分支

def check_commit_all_above_threshold(solve):
    """|S| >= quota 时提交的是全部过阈位置，不做截断。"""
    L, N = 8, 4  # 配额 [2,2,2,2]
    probs = [[0.99, 0.98, 0.97, 0.96, 0.10, 0.10, 0.10, 0.10]]
    sample = [[9] * L]
    tokens = [[1, 2, 3, 4, 5, 6, 7, 8]]
    _, tr, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.95)
    assert _flat(tr) == [[True, True, True, True, False, False, False, False]], _flat(tr)
    # 配额 0（N > L 的后段）也走这支：有过阈就提交
    L2, N2 = 3, 6
    sched = [L2 // N2 + 1] * (L2 % N2) + [L2 // N2] * (N2 - L2 % N2)
    assert sched == [1, 1, 1, 0, 0, 0], sched
    probs2 = [[0.99, 0.10, 0.97]]
    _, tr2, _ = call_mask(solve, [[1, 2, 3]], probs2, [[9, 9, 9]], 3, N2, mask_id=9, th=0.95)
    assert _flat(tr2) == [[True, False, True]], _flat(tr2)
    _, tr3, _ = call_mask(solve, [[1, 2, 3]], probs2, [[9, 9, 9]], 5, N2, mask_id=9, th=0.05)
    assert _flat(tr3) == [[True, True, True]], _flat(tr3)


def check_topk_fallback_ignores_unmasked(solve):
    """过阈数不足时按名次补到 quota，未掩码的高概率位永远不参与。"""
    L, N = 6, 3  # 配额 [2,2,2]
    probs = [[0.10, 1.00, 0.55, 0.45, 0.05, 0.30]]
    sample = [[9, 1, 9, 9, 9, 9]]  # 下标 1 未掩码，却是全行概率最大
    tokens = [[1, 2, 3, 4, 5, 6]]
    # 活动位里的前两名是 2(0.55) 与 3(0.45)，1.00 那一格因为没掩码被当成 -inf
    _, tr, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.95)
    assert _flat(tr) == [[False, False, True, True, False, False]], _flat(tr)
    _, tr2, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.999)
    assert _flat(tr2) == [[False, False, True, True, False, False]], _flat(tr2)


def check_inactive_positions_compare_as_negative_infinity(solve):
    """阈值为负时非活动位靠 -inf 而不是 0 落榜；用 0 填充会把它们也算进 S。"""
    L, N = 4, 4  # 配额 [1,1,1,1]
    probs = [[0.10, 0.20, 0.30, 0.40]]
    sample = [[9, 1, 2, 9]]  # 活动位 0、3
    tokens = [[5, 6, 7, 8]]
    _, tr, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=-0.5)
    assert _flat(tr) == [[True, False, False, True]], _flat(tr)
    prev, _, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=-0.5)
    assert _flat(prev) == [[5, 1, 2, 8]], _flat(prev)


def check_row_without_masks_only_blanks_that_row(solve):
    L, N = 5, 2  # 配额 [3,2]
    probs = [[0.99, 0.99, 0.99, 0.10, 0.10], [0.99, 0.99, 0.99, 0.99, 0.99]]
    sample = [[9, 9, 9, 1, 2], [3, 4, 5, 6, 7]]  # 第二行一个掩码都没有
    tokens = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
    prev, tr, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.5)
    assert _flat(tr) == [[True, True, True, False, False], [False] * 5], _flat(tr)
    # 整块都没有掩码：本步 transfer 恒空，哪怕全都过阈
    sample2 = [[3, 4, 5, 6, 7], [1, 2, 3, 4, 5]]
    prev2, tr2, _ = call_mask(solve, tokens, probs, sample2, 0, N, mask_id=9, th=0.5)
    assert not tr2.any(), _flat(tr2)
    assert _flat(prev2) == sample2, _flat(prev2)


def check_clamped_step_uses_last_quota(solve):
    """t >= N 不报错，配额取表尾；本例表尾是 1。"""
    L, N = 5, 2  # [3,2]
    probs = [[0.4, 0.9, 0.4, 0.8, 0.4]]
    sample = [[9, 9, 9, 9, 9]]
    tokens = [[1, 2, 3, 4, 5]]
    got_t1 = _flat(call_mask(solve, tokens, probs, sample, 1, N, mask_id=9, th=0.99)[1])
    for t in (2, 5, 40):
        assert _flat(call_mask(solve, tokens, probs, sample, t, N, mask_id=9, th=0.99)[1]) == got_t1, t
    assert got_t1 == [[False, True, False, True, False]], got_t1


# ---------------------------------------------------------------- 编辑开关

def check_editing_threshold_asymmetry(solve):
    """掩码模式要 >0 才开编辑；均匀模式只要 not None，0.0 与负数照样开。"""
    L, N = 6, 3
    probs = [[0.99, 0.99, 0.99, 0.80, 0.70, 0.60]]
    mask_sample = [[9, 9, 9, 1, 2, 3]]
    tokens = [[4, 5, 6, 7, 8, 9]]
    for et in (0.0, -0.5):
        _, _, ed = call_mask(solve, tokens, probs, mask_sample, 0, N, mask_id=9, th=0.95, et=et)
        assert not ed.any(), (et, _flat(ed))
    _, _, ed_pos = call_mask(solve, tokens, probs, mask_sample, 0, N, mask_id=9, th=0.95, et=0.5)
    assert _flat(ed_pos) == [[False, False, False, True, True, True]], _flat(ed_pos)
    _, _, ed_none = call_mask(solve, tokens, probs, mask_sample, 0, N, mask_id=9, th=0.95, et=None)
    assert not ed_none.any()

    uni_sample = [[1, 2, 3, 4, 5, 6]]
    committed = [[False, False, False, True, True, True]]
    rnd = [[0, 0, 0, 0, 0, 0]]
    for et in (0.0, -0.5):
        _, _, ed = call_uniform(solve, tokens, probs, uni_sample, 3, N, th=0.999, et=et,
                                committed=committed, rnd=rnd)
        assert _flat(ed) == [[False, False, False, True, True, True]], (et, _flat(ed))
    _, _, ed = call_uniform(solve, tokens, probs, uni_sample, 3, N, th=0.999, et=None,
                            committed=committed, rnd=rnd)
    assert not ed.any()


def check_editing_requires_token_change_and_non_active(solve):
    L, N = 6, 3
    probs = [[0.95, 0.95, 0.95, 0.95, 0.95, 0.95]]
    sample = [[9, 1, 1, 1, 1, 1]]
    tokens = [[2, 1, 5, 1, 6, 1]]  # 下标 1、3、5 没换 token
    _, tr, ed = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.99, et=0.5)
    assert _flat(tr) == [[True, False, False, False, False, False]], _flat(tr)
    assert _flat(ed) == [[False, False, True, False, True, False]], _flat(ed)
    prev, _, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.99, et=0.5)
    assert _flat(prev) == [[2, 1, 5, 1, 6, 1]], _flat(prev)


def check_prompt_mask_only_applies_to_mask_mode(solve):
    L, N = 6, 3
    probs = [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]
    sample = [[9, 9, 1, 2, 3, 4]]
    tokens = [[5, 6, 7, 8, 9, 0]]
    pm = [False, False, True, True, False, False]
    _, _, ed_off = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.95, et=0.5, pm=None)
    assert _flat(ed_off) == [[False, False, True, True, True, True]], _flat(ed_off)
    _, _, ed_on = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.95, et=0.5, pm=pm)
    assert _flat(ed_on) == [[False, False, False, False, True, True]], _flat(ed_on)
    # 均匀模式：prompt_mask 完全不参与，两个结果必然相同
    committed = [[False, False, True, True, True, True]]
    rnd = [[1] * L]
    base = call_uniform(solve, tokens, probs, sample, 2, N, th=0.95, et=0.5, committed=committed, rnd=rnd)
    with_pm = call_uniform(solve, tokens, probs, sample, 2, N, th=0.95, et=0.5, committed=committed,
                           rnd=rnd, pm=pm)
    assert _as_lists(base) == _as_lists(with_pm)


# ---------------------------------------------------------------- 均匀模式细节

def check_committed_is_reset_at_t_zero(solve):
    L, N = 6, 3
    probs = [[0.10, 0.90, 0.55, 0.40, 0.30, 0.20]]
    sample = [[1, 2, 3, 4, 5, 6]]
    tokens = [[7, 8, 9, 0, 1, 2]]
    rnd = [[9, 9, 3, 7, 9, 9]]
    all_committed = [[True] * L]
    prev, tr, ed = call_uniform(solve, tokens, probs, sample, 0, N, th=0.999, et=0.5,
                                committed=all_committed, rnd=rnd)
    target = (1 * L + N - 1) // N
    assert target == 2, target
    # 若 t=0 真把 committed 当用，needed 会被 clamp 成 0，整行空、全落进 random_tokens
    assert _flat(tr) == [[False, True, True, False, False, False]], _flat(tr)
    assert _flat(prev) == [[9, 8, 9, 7, 9, 9]], _flat(prev)
    assert not ed.any(), _flat(ed)
    # t=1 才轮到外部状态生效：全 True 的 committed 把名额吃光，一个都不新提交
    prev1, tr1, _ = call_uniform(solve, tokens, probs, sample, 1, N, th=0.999, et=0.5,
                                 committed=all_committed, rnd=rnd)
    assert not tr1.any(), _flat(tr1)
    assert _flat(prev1) == [[1, 8, 9, 4, 5, 6]], _flat(prev1)  # 不新提交，只编辑 1 与 2


def check_renoise_protects_this_steps_commits(solve):
    L, N = 6, 3
    probs = [[0.10, 0.20, 0.30, 0.40, 0.50, 0.60]]
    sample = [[1] * L]
    tokens = [[2] * L]
    rnd = [[7] * L]
    committed = [[True, True, False, False, False, False]]
    prev, tr, _ = call_uniform(solve, tokens, probs, sample, 1, N, th=0.999,
                               committed=committed, rnd=rnd)
    # target = (2*6+2)//3 = 4，已提交 2 -> needed 2：未提交里最高的两格是 5(0.60) 与 4(0.50)
    assert _flat(tr) == [[False, False, False, False, True, True]], _flat(tr)
    assert _flat(prev) == [[1, 1, 7, 7, 2, 2]], _flat(prev)
    # 刚定稿的两格没被 random_tokens 冲掉，仍未定稿的两格被冲掉
    prev2, tr2, _ = call_uniform(solve, tokens, probs, sample, 2, N, th=0.999,
                                 committed=[[True, True, True, True, True, False]], rnd=rnd)
    assert _flat(tr2) == [[False, False, False, False, False, True]], _flat(tr2)
    assert _flat(prev2) == [[1, 1, 1, 1, 1, 2]], _flat(prev2)
    # needed 已经是 0：一个都不新提交，未定稿的格子全部重加噪
    prev3, tr3, _ = call_uniform(solve, tokens, probs, sample, 2, N, th=0.999,
                                 committed=[[True] * L], rnd=rnd)
    assert not tr3.any() and _flat(prev3) == [[1] * L], _flat(prev3)


def check_uniform_threshold_adds_to_quota(solve):
    """needed 与过阈是并集：过阈的即使名次落在 needed 之外也照样提交。"""
    L, N = 8, 16  # target(1) = (8+15)//16 = 1 -> needed 只有 1 个名额
    probs = [[0.97, 0.96, 0.955, 0.90, 0.10, 0.09, 0.08, 0.07]]
    sample = [[0] * L]
    tokens = [[i + 1 for i in range(L)]]
    rnd = [[9] * L]
    prev, tr, _ = call_uniform(solve, tokens, probs, sample, 0, N, th=0.95,
                               committed=[[False] * L], rnd=rnd)
    assert _flat(tr) == [[True, True, True, False, False, False, False, False]], _flat(tr)
    assert _flat(prev) == [[1, 2, 3, 9, 9, 9, 9, 9]], _flat(prev)
    # 已提交的那格虽然 0.97 过阈，也被 ~base 挡住；剩下的两个过阈位照样进并集
    prev2, tr2, _ = call_uniform(solve, tokens, probs, sample, 1, N, th=0.95,
                                 committed=[[True] + [False] * 7], rnd=rnd)
    assert _flat(tr2) == [[False, True, True, False, False, False, False, False]], _flat(tr2)
    assert _flat(prev2) == [[0, 2, 3, 9, 9, 9, 9, 9]], _flat(prev2)


def check_t_beyond_steps_commits_everything(solve):
    L, N = 5, 3
    probs = [[0.10, 0.20, 0.30, 0.40, 0.50]]
    sample = [[0] * L]
    tokens = [[9] * L]
    rnd = [[7] * L]
    committed = [[False, True, False, False, False]]
    t = 7  # target = (8*5+2)//3 = 13 > L
    prev, tr, _ = call_uniform(solve, tokens, probs, sample, t, N, th=0.999, committed=committed, rnd=rnd)
    assert _flat(tr) == [[True, False, True, True, True]], _flat(tr)
    assert _flat(prev) == [[9, 0, 9, 9, 9]], _flat(prev)  # 已提交的保持原 token，其余全部定稿


# ---------------------------------------------------------------- 精度与并列

def check_float64_probs_are_downcast_before_compare(solve):
    L, N = 4, 4  # 每步配额 1
    probs64 = _t([[0.9500000000000001, 0.96, 0.5, 0.4]], torch.float64)
    tokens = _t([[1, 2, 3, 4]], torch.int64)
    sample = _t([[9, 9, 9, 9]], torch.int64)
    tr = solve(tokens, probs64, sample, 0, N, 9, 0.95, None, None, None, None)[1]
    assert _flat(tr) == [[False, True, False, False]], _flat(tr)
    # 不降精度会得到 [True, True, ...]：过阈数从 1 变成 2
    tr32 = solve(tokens, probs64.to(torch.float32), sample, 0, N, 9, 0.95, None, None, None, None)[1]
    assert _flat(tr32) == [[False, True, False, False]], _flat(tr32)


def check_ties_take_smaller_index(solve):
    L, N = 6, 3
    probs = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    tokens = [[1, 2, 3, 4, 5, 6]]
    sample = [[9] * L]
    _, tr, _ = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.99)
    assert _flat(tr) == [[True, True, False, False, False, False]], _flat(tr)
    # 下标 0 也是活动位，所以并列里最下的两个活动位是 0 与 2
    _, tr2, _ = call_mask(solve, tokens, probs, [[9, 1, 9, 9, 9, 9]], 0, N, mask_id=9, th=0.99)
    assert _flat(tr2) == [[True, False, True, False, False, False]], _flat(tr2)
    _, tu, _ = call_uniform(solve, tokens, probs, [[1, 2, 3, 4, 5, 6]], 0, N, th=0.99,
                            committed=[[False] * L], rnd=[[0] * L])
    assert _flat(tu) == [[True, True, False, False, False, False]], _flat(tu)
    # 已提交的位置排在 -inf 组里，不占用名额，也不把并列顺序打乱
    _, tu2, _ = call_uniform(solve, tokens, probs, [[1, 2, 3, 4, 5, 6]], 1, N, th=0.99,
                             committed=[[False, True, False, False, False, False]], rnd=[[0] * L])
    # target(2) = (12+2)//3 = 4，已提交 1 -> needed 3，并列按 0、2、3 的顺序取
    assert _flat(tu2) == [[True, False, True, True, False, False]], _flat(tu2)


def check_threshold_is_strict(solve):
    """恰好等于阈值不算过：p > threshold 是严格大于（阈值也按 float32 参与比较）。"""
    L, N = 4, 8  # target(1) = (4+7)//8 = 1 -> 只有一个名额
    probs = [[0.95, 0.96, 0.10, 0.05]]
    sample = [[1, 2, 3, 4]]
    tokens = [[5, 6, 7, 8]]
    rnd = [[9, 9, 9, 9]]
    _, tr, _ = call_uniform(solve, tokens, probs, sample, 0, N, th=0.95,
                            committed=[[False] * L], rnd=rnd)
    assert _flat(tr) == [[False, True, False, False]], _flat(tr)  # 写成 >= 就会多一个下标 0
    _, tr2, _ = call_uniform(solve, tokens, probs, sample, 0, N, th=0.949999,
                             committed=[[False] * L], rnd=rnd)
    assert _flat(tr2) == [[True, True, False, False]], _flat(tr2)
    # 编辑阈值同样严格：非掩码位的 0.95 在 et=0.95 下不动，在 et=0.94 下被改写
    mask_sample = [[9, 1, 2, 3]]
    probs3 = [[0.10, 0.95, 0.40, 0.30]]
    _, _, ed = call_mask(solve, tokens, probs3, mask_sample, 0, N, mask_id=9, th=0.99, et=0.95)
    assert _flat(ed) == [[False, False, False, False]], _flat(ed)
    _, _, ed2 = call_mask(solve, tokens, probs3, mask_sample, 0, N, mask_id=9, th=0.99, et=0.94)
    assert _flat(ed2) == [[False, True, False, False]], _flat(ed2)


# ---------------------------------------------------------------- 结构与批量

def check_batch_rows_are_independent(solve):
    L, N = 6, 3
    probs = [[0.9, 0.1, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]]
    sample = [[9, 9, 9, 1, 2, 3], [1, 2, 3, 9, 9, 9]]
    tokens = [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5]]
    whole = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.95, et=0.5)
    for r in range(2):
        single = call_mask(solve, [tokens[r]], [probs[r]], [sample[r]], 0, N, mask_id=9, th=0.95, et=0.5)
        for a, b in zip(_as_lists(whole), _as_lists(single)):
            assert a[r] == b[0], (r, a[r], b[0])


def check_transfer_and_editing_are_disjoint(solve):
    L, N = 8, 4
    torch.manual_seed(7)
    for _ in range(40):
        sample = torch.where(torch.rand(2, L) < 0.5, torch.full((2, L), 9, dtype=torch.long),
                             torch.randint(0, 9, (2, L)))
        probs = torch.rand(2, L, dtype=torch.float32)
        tokens = torch.randint(0, 10, (2, L), dtype=torch.long)
        out = solve(tokens, probs, sample, int(torch.randint(0, N + 2, (1,))), N, 9,
                    float(probs.mean()), 0.3, None, None, None)
        assert not (out[1] & out[2]).any()
        mask = out[1] | out[2]
        assert torch.equal(out[0][mask], tokens[mask])
        assert torch.equal(out[0][~mask], sample[~mask])
        committed = torch.rand(2, L) < 0.4
        rnd = torch.randint(0, 10, (2, L), dtype=torch.long)
        for t in range(N + 1):
            u = solve(tokens, probs, sample if t == 0 else out[0], t, N, None,
                      float(probs.mean()), 0.3, None, committed, rnd)
            assert not (u[1] & u[2]).any(), t


# ---------------------------------------------------------------- 对拍与类型

def check_against_oracle(solve):
    tried = 0
    for L in (1, 3, 5, 6, 10):
        for N in (1, 2, 3, 7):
            torch.manual_seed(L * 13 + N)
            for mask_mode in (True, False):
                for et in (None, 0.0, 0.3, -0.25):
                    sample = torch.where(torch.rand(3, L) < 0.5, torch.full((3, L), 9, dtype=torch.long),
                                         torch.randint(0, 9, (3, L), dtype=torch.long))
                    probs = torch.rand(3, L, dtype=torch.float32)
                    tokens = torch.randint(0, 10, (3, L), dtype=torch.long)
                    pm = (torch.arange(L) % 3 == 0).tolist()
                    committed = torch.rand(3, L) < 0.45
                    rnd = torch.randint(0, 10, (3, L), dtype=torch.long)
                    for t in (0, 1, N - 1, N, N + 3):
                        for th in (0.05, 0.5, 0.95):
                            args = (tokens.tolist(), probs.tolist(), sample.tolist(), t, N,
                                    9 if mask_mode else None, float(th), et,
                                    pm, committed.tolist() if not mask_mode else None,
                                    rnd.tolist() if not mask_mode else None)
                            if mask_mode:
                                out = call_mask(solve, tokens.tolist(), probs.tolist(), sample.tolist(),
                                                t, N, mask_id=9, th=th, et=et, pm=pm)
                            else:
                                out = call_uniform(solve, tokens.tolist(), probs.tolist(), sample.tolist(),
                                                   t, N, th=th, et=et, committed=committed.tolist(),
                                                   rnd=rnd.tolist(), pm=pm)
                            want = oracle(*args)
                            assert [list(x) for x in _as_lists(out)] == [_flat(w) for w in want], (L, N, t, th, et, mask_mode)
                            tried += 1
    assert tried >= 400, tried


def check_rollout_matches_oracle(solve):
    """把单步串成多步：均匀模式的外部状态必须由上一步的 transfer 推出来。"""
    for L, N in ((6, 4), (10, 3), (5, 7)):
        torch.manual_seed(L)
        probs = torch.rand(2, L, dtype=torch.float32)
        tokens = torch.randint(0, 10, (2, L), dtype=torch.long)
        sample = torch.randint(0, 10, (2, L), dtype=torch.long)
        rnd = torch.randint(0, 10, (2, L), dtype=torch.long)
        committed = torch.zeros(2, L, dtype=torch.bool)
        for t in range(N + 2):
            prev, tr, ed = solve(tokens, probs, sample, t, N, None, 0.6, 0.4, None,
                                 committed, rnd)
            args = (tokens.tolist(), probs.tolist(), sample.tolist(), t, N, None, 0.6, 0.4, None,
                    committed.tolist(), rnd.tolist())
            want = oracle(*args)
            assert [list(x) for x in _as_lists((prev, tr, ed))] == [_flat(w) for w in want], (L, N, t)
            if t == 0:
                committed = torch.zeros(2, L, dtype=torch.bool)
            committed = committed | tr
            sample = prev
        assert committed.all(), (L, N, committed.tolist())


def check_dtypes_and_shapes(solve):
    L, N = 5, 2
    probs = [[0.9, 0.4, 0.3, 0.2, 0.1]]
    tokens = [[1, 2, 3, 4, 5]]
    sample = [[9, 9, 9, 9, 9]]
    out = call_mask(solve, tokens, probs, sample, 0, N, mask_id=9, th=0.5, et=0.2)
    prev, tr, ed = out
    assert prev.dtype == torch.int64 and tr.dtype == torch.bool and ed.dtype == torch.bool
    assert tuple(prev.shape) == tuple(tr.shape) == tuple(ed.shape) == (1, L)
    # int32 / uint8 的 token 也接受，结果按 int64 返回
    p32 = solve(_t(tokens, torch.int32), _t(probs), _t(sample, torch.int32), 0, N, 9, 0.5, 0.2,
                None, None, None)
    assert p32[0].dtype == torch.int64 and torch.equal(p32[0], prev)
    # 标量 threshold 传 int 也接受
    out_i = call_mask(solve, tokens, [[1.0, 0.4, 0.3, 0.2, 0.1]], sample, 0, N, mask_id=9, th=0, et=None)
    assert int(out_i[1].sum()) == L, _flat(out_i[1])
    # L=1 的极端形状
    one = call_mask(solve, [[1]], [[0.99]], [[9]], 0, 1, mask_id=9, th=0.5)
    assert _flat(one[0]) == [[1]] and _flat(one[1]) == [[True]]


def check_invalid_inputs(solve):
    L, N = 4, 2
    tok, pr, sm = [[1, 2, 3, 4]], [[0.5] * 4], [[9, 9, 1, 2]]
    committed, rnd = [[False, False, True, True]], [[0, 0, 0, 1]]

    def _keep(v, dtype):
        return v if isinstance(v, torch.Tensor) else _t(v, dtype)

    def run_mask(**kw):
        a = dict(tokens=tok, probs=pr, sample=sm, t=0, N=N, mask=9, th=0.5, et=None, pm=None)
        a.update(kw)
        solve(_keep(a["tokens"], torch.int64), _keep(a["probs"], torch.float32),
              _keep(a["sample"], torch.int64),
              a["t"], a["N"], a["mask"], a["th"], a["et"], a["pm"], None, None)

    def run_uniform(**kw):
        a = dict(tokens=tok, probs=pr, sample=sm, t=1, N=N, th=0.5, et=None,
                 cm=committed, rnd=rnd, pm=None)
        a.update(kw)
        solve(_keep(a["tokens"], torch.int64), _keep(a["probs"], torch.float32),
              _keep(a["sample"], torch.int64),
              a["t"], N, None, a["th"], a["et"], a["pm"],
              _keep(a["cm"], torch.bool), _keep(a["rnd"], torch.int64))

    bad_mask = [
        ("tokens 非张量", lambda: solve(tok, _t(pr), _t(sm, torch.int64), 0, N, 9, 0.5, None,
                                        None, None, None)),
        ("tokens 浮点", lambda: run_mask(tokens=_t(tok))),
        ("tokens 含负", lambda: run_mask(tokens=_t([[1, -2, 3, 4]], torch.int64))),
        ("probs 整数", lambda: run_mask(probs=_t([[1, 2, 3, 4]], torch.int64))),
        ("probs 含 NaN", lambda: run_mask(probs=_t([[float("nan"), 0.2, 0.3, 0.4]]))),
        ("probs 大于 1", lambda: run_mask(probs=_t([[1.5, 0.2, 0.3, 0.4]]))),
        ("probs 为负", lambda: run_mask(probs=_t([[-0.1, 0.2, 0.3, 0.4]]))),
        ("probs 形状不符", lambda: run_mask(probs=[[0.5] * 3])),
        ("sample 是 bool", lambda: run_mask(sample=_t([[1, 0, 1, 0]], torch.bool))),
        ("sample 三维", lambda: run_mask(sample=_t([[[9, 9, 1, 2]]], torch.int64))),
        ("两行 batch 不一致", lambda: run_mask(sample=_t([[9, 9, 1, 2], [9, 9]], torch.int64))),
        ("timestep 浮点", lambda: run_mask(t=0.0)),
        ("timestep 为 bool", lambda: run_mask(t=False)),
        ("timestep 负", lambda: run_mask(t=-1)),
        ("num_inference_steps 为 0", lambda: run_mask(N=0)),
        ("mask_token_id 浮点", lambda: run_mask(mask=9.0)),
        ("mask_token_id 负", lambda: run_mask(mask=-1)),
        ("threshold 非数", lambda: run_mask(th="0.5")),
        ("threshold NaN", lambda: run_mask(th=float("nan"))),
        ("threshold inf", lambda: run_mask(th=float("inf"))),
        ("editing_threshold NaN", lambda: run_mask(et=float("nan"))),
        ("prompt_mask 形状错", lambda: run_mask(pm=[False, True])),
        ("prompt_mask 是 python list", lambda: run_mask(pm=[False, True, False, False])),
        ("prompt_mask 非 bool", lambda: run_mask(pm=_t([0, 1, 0, 0], torch.int64))),
    ]
    for name, fn in bad_mask:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"掩码模式未对 {name} 报错")

    bad_uniform = [
        ("缺 committed", lambda: solve(_t(tok, torch.int64), _t(pr), _t(sm, torch.int64), 1, N, None,
                                       0.5, None, None, None, _t(rnd, torch.int64))),
        ("缺 random_tokens", lambda: solve(_t(tok, torch.int64), _t(pr), _t(sm, torch.int64), 1, N, None,
                                           0.5, None, None, _t(committed, torch.bool), None)),
        ("committed 是整数", lambda: run_uniform(cm=_t([[0, 0, 1, 1]], torch.int64))),
        ("committed 形状错", lambda: run_uniform(cm=[[False, False], [True, True]])),
        ("random_tokens 浮点", lambda: run_uniform(rnd=_t([[0.0, 0, 0, 1]]))),
        ("random_tokens 含负", lambda: run_uniform(rnd=[[-1, 0, 0, 1]])),
        ("editing_threshold 是字符串", lambda: run_uniform(et="0.5")),
    ]
    for name, fn in bad_uniform:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"均匀模式未对 {name} 报错")

    # t=0 时 committed 可以是全 False 的合法张量（被忽略），但不给就报错
    solve(_t(tok, torch.int64), _t(pr), _t(sm, torch.int64), 0, N, None, 0.5, None, None,
          _t([[False] * L], torch.bool), _t(rnd, torch.int64))
