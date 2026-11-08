"""848 判题用例：EViT token 重组织账。

四个 check_* 函数，均以 solve 作为唯一参数（judge 内部以 fn(solve) 调用）：
  check_cls_attn_score  —— cls_attn 的打分口径（cls 行 / 去 cls 列 / 跨 head 均值）
  check_select_tokens   —— topk 选 token + keep_rate 的 ceil 与触发条件
  check_fuse_token      —— 非保留 token 的加权【求和】与 extra_token
  check_reorganize      —— 一次完整的 Block 级重组（顺序 / 位置 / 长度）

每个 check 都带一份【独立 oracle】（不依赖参考解，按官方算式重写一遍），
并且每条关键结论都做【双向断言】：
  ① 结果必须等于 oracle；
  ② 结果必须【不等于】那个「看起来也合理」的错误候选。
"""

import math


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------


def _flat(value):
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _f(value):
    return [float(v) for v in _flat(value)]


def _close(a, b, tol=1e-9):
    a = _f(a)
    b = _f(b)
    if len(a) != len(b):
        return False, "length mismatch: %d vs %d" % (len(a), len(b))
    worst = 0.0
    worst_i = -1
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(float(x) - float(y))
        if d > worst:
            worst = d
            worst_i = i
    if worst > tol:
        return False, "max abs diff %.3e at index %d" % (worst, worst_i)
    return True, ""


def _must_differ(a, b, what, tol=1e-9):
    """断言两个候选【必须不同】—— 防止用例丧失区分能力。"""
    ok, _ = _close(a, b, tol=tol)
    assert not ok, "%s: the wrong candidate equals the right one" % what


def _get(res, key):
    """兼容 solve 返回 dict 或对象属性两种风格。"""
    if isinstance(res, dict):
        if key not in res:
            raise AssertionError("missing key: %s" % key)
        return res[key]
    if not hasattr(res, key):
        raise AssertionError("missing attribute: %s" % key)
    return getattr(res, key)


# --------------------------------------------------------------------------
# 独立 oracle（按官方算式重写，不依赖参考解）
# --------------------------------------------------------------------------


def _oracle_cls_attn(attn):
    """官方：attn[:, :, 0, 1:].mean(dim=1)。

    attn 按 [H][N][N] 给出：取每个 head 的【第 0 行】（cls 作为 query），
    列从 1 开始（去掉 cls 自身那一列），再跨 head 求【算术平均】。
    """
    h = len(attn)
    n = len(attn[0])
    out = []
    for j in range(1, n):
        s = 0.0
        for hi in range(h):
            s += float(attn[hi][0][j])
        out.append(s / h)
    return out


def _oracle_topk_order(scores, k):
    """官方 torch.topk(largest=True, sorted=True)：按分数降序，并列按下标升序。"""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return order[:k]


def _oracle_complement(idx, dim):
    """官方 complement_idx(idx, dim)：补集，按下标升序。"""
    keep = set(int(i) for i in idx)
    return [i for i in range(dim) if i not in keep]


def _oracle_fuse(non_cls, cls_attn, complement):
    """官方：extra_token = sum(non_topk * non_topk_attn)（加权【求和】）。"""
    c = len(non_cls[0])
    extra = [0.0] * c
    for i in complement:
        w = float(cls_attn[i])
        for d in range(c):
            extra[d] += w * float(non_cls[i][d])
    return extra


def _oracle_reorganize(hidden, idx, extra_token, fuse):
    """官方：cat([cls, gather(non_cls, idx), (extra,)])。"""
    cls_row = list(hidden[0])
    patches = hidden[1:]
    out = [cls_row] + [list(patches[i]) for i in idx]
    if fuse:
        out = out + [list(extra_token)]
    return out


# --------------------------------------------------------------------------
# mode 1: cls_attn_score
# --------------------------------------------------------------------------

# 3 heads × 5 tokens（每行是 softmax 后的注意力分布，和为 1）
ATTN3 = [
    # head 0
    [[0.5, 0.1, 0.2, 0.1, 0.1],
     [0.1, 0.4, 0.2, 0.2, 0.1],
     [0.1, 0.1, 0.4, 0.2, 0.2],
     [0.1, 0.1, 0.2, 0.4, 0.2],
     [0.1, 0.2, 0.1, 0.2, 0.4]],
    # head 1
    [[0.2, 0.3, 0.2, 0.2, 0.1],
     [0.2, 0.2, 0.3, 0.2, 0.1],
     [0.1, 0.2, 0.2, 0.3, 0.2],
     [0.2, 0.1, 0.2, 0.2, 0.3],
     [0.3, 0.1, 0.2, 0.1, 0.3]],
    # head 2
    [[0.4, 0.1, 0.2, 0.2, 0.1],
     [0.1, 0.3, 0.3, 0.2, 0.1],
     [0.2, 0.1, 0.3, 0.2, 0.2],
     [0.1, 0.2, 0.2, 0.3, 0.2],
     [0.2, 0.1, 0.2, 0.2, 0.3]],
]

# 单 head（退化为跨 head 均值 = 本身）
ATTN1 = [
    [[0.25, 0.25, 0.25, 0.25],
     [0.1, 0.4, 0.3, 0.2],
     [0.2, 0.2, 0.3, 0.3],
     [0.3, 0.2, 0.2, 0.3]],
]


def check_cls_attn_score(solve):
    """cls_attn：只取第 0 行 + 去掉 cls 列 + 跨 head 算术平均。"""

    # ---- 多 head 基本情形 ----------------------------------------------
    r = solve("cls_attn_score", attn=ATTN3)
    got = _f(_get(r, "cls_attn"))
    exp = _oracle_cls_attn(ATTN3)
    ok, msg = _close(got, exp)
    assert ok, "cls_attn mismatch: %s" % msg

    assert int(_get(r, "n_heads")) == 3, (
        "n_heads must be 3, got %r" % (_get(r, "n_heads"),))
    assert int(_get(r, "n_tokens")) == 5, (
        "n_tokens must be 5 (cls + 4 patch), got %r" % (_get(r, "n_tokens"),))

    # ★ 长度必须是 N-1（去掉 cls 自身那一列）
    assert len(got) == 4, (
        "cls_attn length must be N-1 = 4, got %d" % len(got))

    # ★ 双向断言 1：不能把 cls 自身那一列也算进去
    wrong_with_cls = []
    for j in range(5):
        s = 0.0
        for hi in range(3):
            s += float(ATTN3[hi][0][j])
        wrong_with_cls.append(s / 3)
    _must_differ(got, wrong_with_cls,
                 "cls_attn must exclude the cls column (col 0)")

    # ★ 双向断言 2：不能跨 head 求【和】而不是均值
    wrong_sum = [v * 3 for v in exp]
    _must_differ(got, wrong_sum, "cls_attn must average over heads, not sum")

    # ★ 双向断言 3：不能对【所有行】求均值（而不是只取 cls 那一行）
    wrong_all_rows = []
    for j in range(1, 5):
        s = 0.0
        cnt = 0
        for hi in range(3):
            for rrow in ATTN3[hi]:
                s += float(rrow[j])
                cnt += 1
        wrong_all_rows.append(s / cnt)
    _must_differ(got, wrong_all_rows,
                 "cls_attn must use row 0 only (cls as query)")

    # ★ 双向断言 4：不能跨 head 求【最大】而不是均值
    wrong_max = []
    for j in range(1, 5):
        wrong_max.append(max(float(ATTN3[hi][0][j]) for hi in range(3)))
    _must_differ(got, wrong_max, "cls_attn must average over heads, not max")

    # ---- 单 head：均值退化为本身 ----------------------------------------
    r1 = solve("cls_attn_score", attn=ATTN1)
    got1 = _f(_get(r1, "cls_attn"))
    exp1 = _oracle_cls_attn(ATTN1)
    ok, msg = _close(got1, exp1)
    assert ok, "single-head cls_attn mismatch: %s" % msg
    assert int(_get(r1, "n_heads")) == 1
    assert len(got1) == 3

    # ---- head 数不同的两个用例必须给出不同的结果 -----------------------
    # 同一个 head0，换成 2 个 head（第二个 head 全是 0.5 的均匀分布）
    attn2 = [ATTN3[0],
             [[0.2, 0.2, 0.2, 0.2, 0.2]] * 5]
    r2 = solve("cls_attn_score", attn=attn2)
    got2 = _f(_get(r2, "cls_attn"))
    exp2 = _oracle_cls_attn(attn2)
    ok, msg = _close(got2, exp2)
    assert ok, "two-head cls_attn mismatch: %s" % msg
    _must_differ(got2, got, "adding a head must change the mean")

    # ---- cls 行是「唯一峰值」的情形 ------------------------------------
    attn_peak = [
        [[0.1, 0.4, 0.2, 0.1, 0.1],
         [0.2, 0.2, 0.2, 0.2, 0.2],
         [0.2, 0.2, 0.2, 0.2, 0.2],
         [0.2, 0.2, 0.2, 0.2, 0.2],
         [0.2, 0.2, 0.2, 0.2, 0.2]],
    ]
    rp = solve("cls_attn_score", attn=attn_peak)
    gotp = _f(_get(rp, "cls_attn"))
    ok, msg = _close(gotp, _oracle_cls_attn(attn_peak))
    assert ok, "peak cls_attn mismatch: %s" % msg
    assert abs(gotp[0] - 0.4) < 1e-12, (
        "cls_attn[0] must be the cls row's column 1, got %.6f" % gotp[0])


# --------------------------------------------------------------------------
# mode 2: select_tokens
# --------------------------------------------------------------------------


def check_select_tokens(solve):
    """select_tokens：触发条件 / ceil / tokens 覆盖 / 提前返回 / topk 降序。"""

    ca = [0.1, 0.5, 0.3, 0.05, 0.05]        # 5 个 patch

    # ---- 基本情形：self_keep_rate < 1 且 keep_rate < 1 ------------------
    r = solve("select_tokens", cls_attn=ca, keep_rate=0.5,
              self_keep_rate=0.5, tokens=None)
    assert int(_get(r, "left_tokens")) == 3, (
        "left_tokens must be ceil(0.5 * 5) = 3, got %r"
        % (_get(r, "left_tokens"),))
    idx = [int(i) for i in _get(r, "idx")]
    assert idx == [1, 2, 0], (
        "idx must be topk order (score descending) = [1, 2, 0], got %s" % idx)
    comp = [int(i) for i in _get(r, "complement")]
    assert comp == [3, 4], (
        "complement must be ascending = [3, 4], got %s" % comp)

    # ★ 双向断言：不能按下标升序返回
    _must_differ(idx, [0, 1, 2],
                 "idx must be in topk (descending) order, not ascending")

    # ★ 双向断言：ceil 而不是 floor
    r_floor = solve("select_tokens", cls_attn=ca, keep_rate=0.3,
                    self_keep_rate=0.5, tokens=None)
    assert int(_get(r_floor, "left_tokens")) == 2, (
        "left_tokens must be ceil(0.3 * 5) = 2, got %r"
        % (_get(r_floor, "left_tokens"),))
    assert len(_get(r_floor, "idx")) == 2, (
        "idx length must equal left_tokens (ceil, not floor)")

    # ---- ★ 触发条件：self_keep_rate 是【构造时】的值 --------------------
    r_self1 = solve("select_tokens", cls_attn=ca, keep_rate=0.5,
                    self_keep_rate=1.0, tokens=None)
    assert int(_get(r_self1, "left_tokens")) == 5, (
        "self_keep_rate=1.0 must NOT trigger the keep_rate branch; "
        "left_tokens must stay N-1 = 5, got %r"
        % (_get(r_self1, "left_tokens"),))
    assert _get(r_self1, "idx") is None, (
        "when not triggered, idx must be None")
    assert _get(r_self1, "complement") is None, (
        "when not triggered, complement must be None")

    # ---- ★ 触发条件：keep_rate == 1.0 也不触发 -------------------------
    r_kr1 = solve("select_tokens", cls_attn=ca, keep_rate=1.0,
                  self_keep_rate=0.5, tokens=None)
    assert int(_get(r_kr1, "left_tokens")) == 5, (
        "keep_rate=1.0 must NOT trigger; left_tokens must stay 5, got %r"
        % (_get(r_kr1, "left_tokens"),))
    assert _get(r_kr1, "idx") is None

    # ★ 双向断言：把 self_keep_rate 忽略掉的错误实现会在这里触发
    wrong = solve("select_tokens", cls_attn=ca, keep_rate=0.5,
                  self_keep_rate=1.0, tokens=None)
    assert _get(wrong, "idx") is None, (
        "ignoring self_keep_rate would wrongly select tokens")

    # ---- ★ tokens 单独触发，并【覆盖】keep_rate 的 ceil 结果 -----------
    r_tok = solve("select_tokens", cls_attn=ca, keep_rate=0.9,
                  self_keep_rate=1.0, tokens=2)
    assert int(_get(r_tok, "left_tokens")) == 2, (
        "tokens must override left_tokens (2), got %r"
        % (_get(r_tok, "left_tokens"),))
    idx_tok = [int(i) for i in _get(r_tok, "idx")]
    assert idx_tok == [1, 2], (
        "idx for tokens=2 must be top-2 by score = [1, 2], got %s" % idx_tok)

    # ★ 双向断言：tokens 覆盖 vs ceil(keep_rate * (N-1)) = ceil(0.9*5) = 5
    assert int(_get(r_tok, "left_tokens")) != 5, (
        "tokens must override the ceil result (5), not be ignored")
    # ★ tokens 也能在 self_keep_rate == 1.0 时单独触发
    r_tok2 = solve("select_tokens", cls_attn=ca, keep_rate=0.5,
                   self_keep_rate=1.0, tokens=3)
    assert int(_get(r_tok2, "left_tokens")) == 3, (
        "tokens alone must trigger the branch, got %r"
        % (_get(r_tok2, "left_tokens"),))

    # ★ tokens 【大于】ceil 结果时也必须直接覆盖（不是取 min）
    r_tok3 = solve("select_tokens", cls_attn=ca, keep_rate=0.3,
                   self_keep_rate=0.5, tokens=4)
    assert int(_get(r_tok3, "left_tokens")) == 4, (
        "tokens=4 must override ceil(0.3 * 5) = 2 outright, got %r"
        % (_get(r_tok3, "left_tokens"),))
    idx_tok3 = [int(i) for i in _get(r_tok3, "idx")]
    assert idx_tok3 == [1, 2, 0, 3], (
        "idx for tokens=4 must be top-4 by score = [1, 2, 0, 3], got %s"
        % idx_tok3)
    # ★ 双向断言：min(tokens, ceil) 会得到 2，与正确的 4 不同
    assert int(_get(r_tok3, "left_tokens")) != 2, (
        "tokens must override the ceil result, not be clamped to it")

    # ---- ★ left_tokens == N-1 时提前返回 --------------------------------
    r_full = solve("select_tokens", cls_attn=ca, keep_rate=0.9,
                   self_keep_rate=0.5, tokens=None)
    assert int(_get(r_full, "left_tokens")) == 5, (
        "ceil(0.9 * 5) = 5 must be returned as left_tokens")
    assert _get(r_full, "idx") is None, (
        "left_tokens == N-1 must early-return with idx = None")
    assert _get(r_full, "complement") is None

    # ---- ★ ceil 的边界：0.8 -> 4（触发），0.9 -> 5（提前返回） -----------
    r_b1 = solve("select_tokens", cls_attn=ca, keep_rate=0.8,
                 self_keep_rate=0.5, tokens=None)
    assert int(_get(r_b1, "left_tokens")) == 4, (
        "ceil(0.8 * 5) = 4, got %r" % (_get(r_b1, "left_tokens"),))
    assert _get(r_b1, "idx") is not None, (
        "left_tokens=4 < N-1 must actually select tokens")

    # ---- ★ 基数是 N-1 而不是 N ----------------------------------------
    # ca10 有 10 个 patch ⇒ N = 11，基数 N-1 = 10
    ca10 = [0.01 * (i + 1) for i in range(10)]
    r_base = solve("select_tokens", cls_attn=ca10, keep_rate=0.85,
                   self_keep_rate=0.5, tokens=None)
    assert int(_get(r_base, "left_tokens")) == 9, (
        "left_tokens must use N-1 = 10 as the base: ceil(0.85 * 10) = 9, "
        "got %r" % (_get(r_base, "left_tokens"),))
    # 若错误地用 N=11 当基数，会得到 ceil(0.85*11) = 10 —— 恰好不同，
    # 换一个能区分的 keep_rate 再测一次。
    r_base2 = solve("select_tokens", cls_attn=ca10, keep_rate=0.75,
                    self_keep_rate=0.5, tokens=None)
    assert int(_get(r_base2, "left_tokens")) == 8, (
        "left_tokens must use N-1 = 10 as the base: ceil(0.75 * 10) = 8, "
        "got %r" % (_get(r_base2, "left_tokens"),))
    # ★ 双向断言：两种错误基数的结果都跟正确值不同
    for wrong_base, tag in ((11, "N"), (9, "N-2")):
        assert int(_get(r_base2, "left_tokens")) != int(
            math.ceil(0.75 * wrong_base)), (
            "the %s-as-base candidate must differ from the right one" % tag)

    # ---- ★ topk 是【降序】输出（与下标升序不同） ------------------------
    ca_desc = [0.5, 0.1, 0.9, 0.2]
    r_desc = solve("select_tokens", cls_attn=ca_desc, keep_rate=0.5,
                   self_keep_rate=0.5, tokens=None)
    idx_desc = [int(i) for i in _get(r_desc, "idx")]
    assert idx_desc == [2, 0], (
        "idx must follow score descending = [2, 0], got %s" % idx_desc)
    _must_differ(idx_desc, [0, 2],
                 "topk output must be descending, not ascending")

    # ---- ★ 并列时按下标升序（全部并列时最稳） --------------------------
    ca_tie = [0.3, 0.3, 0.3, 0.3]
    r_tie = solve("select_tokens", cls_attn=ca_tie, keep_rate=0.5,
                  self_keep_rate=0.5, tokens=None)
    idx_tie = [int(i) for i in _get(r_tie, "idx")]
    assert idx_tie == [0, 1], (
        "fully tied scores must break by ascending index = [0, 1], got %s"
        % idx_tie)
    comp_tie = [int(i) for i in _get(r_tie, "complement")]
    assert comp_tie == [2, 3], (
        "complement must be ascending = [2, 3], got %s" % comp_tie)

    # ---- 分数与补集都必须与 oracle 一致 --------------------------------
    r_or = solve("select_tokens", cls_attn=ca, keep_rate=0.6,
                 self_keep_rate=0.5, tokens=None)
    exp_left = int(math.ceil(0.6 * 5))
    assert int(_get(r_or, "left_tokens")) == exp_left
    exp_idx = _oracle_topk_order(ca, exp_left)
    got_idx = [int(i) for i in _get(r_or, "idx")]
    assert got_idx == exp_idx, (
        "idx mismatch: expected %s got %s" % (exp_idx, got_idx))
    exp_comp = _oracle_complement(exp_idx, 5)
    got_comp = [int(i) for i in _get(r_or, "complement")]
    assert got_comp == exp_comp, (
        "complement mismatch: expected %s got %s" % (exp_comp, got_comp))

    # ---- n_tokens 恒为 N（= patch 数 + 1） -----------------------------
    for res in (r, r_self1, r_tok, r_full):
        assert int(_get(res, "n_tokens")) == 6, (
            "n_tokens must be N = 6 (cls + 5 patch), got %r"
            % (_get(res, "n_tokens"),))

    # ---- ★ left_tokens 必须 >= 1（官方 assert left_tokens >= 1）----------
    # tokens=0 会让 left_tokens 变成 0，官方直接断言失败。
    raised = False
    try:
        solve("select_tokens", cls_attn=ca, keep_rate=0.5,
              self_keep_rate=0.5, tokens=0)
    except Exception:
        raised = True
    assert raised, (
        "left_tokens = 0 must be rejected (official asserts "
        "left_tokens >= 1)")


# --------------------------------------------------------------------------
# mode 3: fuse_token
# --------------------------------------------------------------------------

NON_CLS = [[1.0], [2.0], [3.0], [4.0]]
CLS_ATTN = [0.1, 0.2, 0.3, 0.4]


def check_fuse_token(solve):
    """fuse_token：加权【求和】（不是平均）、权重不归一化、补集升序。"""

    idx = [0, 1]
    comp = [2, 3]
    r = solve("fuse_token", non_cls=NON_CLS, cls_attn=CLS_ATTN,
              idx=idx, complement=comp)
    got = _f(_get(r, "extra_token"))
    exp = _oracle_fuse(NON_CLS, CLS_ATTN, comp)
    ok, msg = _close(got, exp)
    assert ok, "extra_token mismatch: %s" % msg

    # 手算：0.3 * 3.0 + 0.4 * 4.0 = 0.9 + 1.6 = 2.5
    assert abs(got[0] - 2.5) < 1e-12, (
        "extra_token must be 0.3*3 + 0.4*4 = 2.5, got %.12f" % got[0])

    # ★ 双向断言 1：不是加权【平均】
    wrong_avg = [2.5 / 0.7]
    _must_differ(got, wrong_avg, "extra_token must be a weighted SUM")

    # ★ 双向断言 2：不是「权重归一化后的加权和」（等价于加权平均）
    _must_differ(got, wrong_avg,
                 "extra_token must NOT normalize the weights")

    # ★ 双向断言 3：不是「不加权的求和」
    wrong_plain = [3.0 + 4.0]
    _must_differ(got, wrong_plain, "extra_token must apply the weights")

    # ★ 双向断言 4：不是「只用第一个权重」
    wrong_first = [0.3 * 3.0]
    _must_differ(got, wrong_first, "extra_token must sum over all dropped tokens")

    # ---- weight_sum 必须是【未归一化】的权重和 --------------------------
    ws = float(_get(r, "weight_sum"))
    assert abs(ws - 0.7) < 1e-12, (
        "weight_sum must be the raw sum 0.3 + 0.4 = 0.7, got %.12f" % ws)

    # ---- non_topk / non_topk_attn 必须按 complement 的顺序 --------------
    nt = _f(_get(r, "non_topk"))
    ok, msg = _close(nt, [3.0, 4.0])
    assert ok, "non_topk must gather in complement order [3, 4]: %s" % msg
    nta = _f(_get(r, "non_topk_attn"))
    ok, msg = _close(nta, [0.3, 0.4])
    assert ok, "non_topk_attn must be [0.3, 0.4]: %s" % msg

    # ---- 多通道：逐通道独立加权 ----------------------------------------
    nc2 = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    r2 = solve("fuse_token", non_cls=nc2, cls_attn=CLS_ATTN,
               idx=idx, complement=comp)
    got2 = _f(_get(r2, "extra_token"))
    exp2 = _oracle_fuse(nc2, CLS_ATTN, comp)
    ok, msg = _close(got2, exp2)
    assert ok, "multi-channel extra_token mismatch: %s" % msg
    assert abs(got2[0] - 2.5) < 1e-12 and abs(got2[1] - 25.0) < 1e-12, (
        "multi-channel extra_token must be [2.5, 25.0], got %s" % got2)

    # ---- 补集为空的极端：extra_token = 0 -------------------------------
    r_empty = solve("fuse_token", non_cls=[[1.0], [2.0]], cls_attn=[0.4, 0.6],
                    idx=[0, 1], complement=[])
    got_empty = _f(_get(r_empty, "extra_token"))
    ok, msg = _close(got_empty, [0.0])
    assert ok, "empty complement must give extra_token = 0: %s" % msg

    # ---- 权重全为 0：extra_token 也全为 0 ------------------------------
    r_zero = solve("fuse_token", non_cls=[[1.0], [2.0], [3.0]],
                   cls_attn=[0.0, 0.0, 0.0], idx=[0], complement=[1, 2])
    got_zero = _f(_get(r_zero, "extra_token"))
    ok, msg = _close(got_zero, [0.0])
    assert ok, "zero weights must give extra_token = 0: %s" % msg

    # ★ 双向断言：权重全 0 时，加权平均是 0/0（无定义），
    #   加权和是 0 —— 用「和」这一侧
    assert abs(float(_get(r_zero, "weight_sum"))) < 1e-12, (
        "weight_sum must be 0 when all weights are 0")


# --------------------------------------------------------------------------
# mode 4: reorganize
# --------------------------------------------------------------------------

HIDDEN = [[0.0, 0.0],      # cls
          [1.0, 2.0],      # patch 0
          [3.0, 4.0],      # patch 1
          [5.0, 6.0],      # patch 2
          [7.0, 8.0]]      # patch 3


def check_reorganize(solve):
    """reorganize：[cls] + [按 idx 顺序 gather 的 patch] + ([extra] 在末尾)。"""

    idx = [2, 0]
    extra = [9.0, 9.0]
    r = solve("reorganize", hidden=HIDDEN, idx=idx, extra_token=extra,
              fuse=True)
    got = _f(_get(r, "out"))
    exp = _oracle_reorganize(HIDDEN, idx, extra, True)
    ok, msg = _close(got, exp)
    assert ok, "reorganize(fuse=True) mismatch: %s" % msg

    # ★ 长度：fuse 时 = 1 + len(idx) + 1
    assert int(_get(r, "n_tokens")) == 3, (
        "n_tokens must be output length - 1 = 3, got %r"
        % (_get(r, "n_tokens"),))
    out_rows = _get(r, "out")
    assert len(out_rows) == 4, (
        "fused output must have 1 + len(idx) + 1 = 4 rows, got %d"
        % len(out_rows))

    # ★ extra_token 必须在【末尾】（不是最前）
    last = _f(out_rows[-1])
    ok, msg = _close(last, [9.0, 9.0])
    assert ok, "extra_token must be the LAST row: %s" % msg
    first = _f(out_rows[0])
    ok, msg = _close(first, [0.0, 0.0])
    assert ok, "the FIRST row must still be the cls token: %s" % msg

    # ★ 双向断言：extra 放在 cls 之后的错误候选必须不同
    wrong_front = _f([extra, HIDDEN[0], HIDDEN[3], HIDDEN[1]])
    _must_differ(got, wrong_front,
                 "extra_token must be appended at the END, not inserted")

    # ★ gather 必须按 idx 的顺序（分数降序），不是下标升序
    ok, msg = _close(_f(out_rows[1]), [5.0, 6.0])
    assert ok, "row 1 must be patch 2 (idx[0] = 2): %s" % msg
    ok, msg = _close(_f(out_rows[2]), [1.0, 2.0])
    assert ok, "row 2 must be patch 0 (idx[1] = 0): %s" % msg
    wrong_asc = _f([HIDDEN[0], HIDDEN[1], HIDDEN[3], extra])
    _must_differ(got, wrong_asc, "patches must follow idx order, not ascending")

    # ★ kept_positions：patch i 在原序列中的位置是 i + 1（cls 占 0）
    kp = [int(i) for i in _get(r, "kept_positions")]
    assert kp == [3, 1], (
        "kept_positions must be the original positions [3, 1], got %s" % kp)

    # ---- 不 fuse：不带 extra_token -------------------------------------
    r_nf = solve("reorganize", hidden=HIDDEN, idx=idx, fuse=False)
    got_nf = _f(_get(r_nf, "out"))
    exp_nf = _oracle_reorganize(HIDDEN, idx, None, False)
    ok, msg = _close(got_nf, exp_nf)
    assert ok, "reorganize(fuse=False) mismatch: %s" % msg
    assert len(_get(r_nf, "out")) == 3, (
        "non-fused output must have 1 + len(idx) = 3 rows, got %d"
        % len(_get(r_nf, "out")))
    assert int(_get(r_nf, "n_tokens")) == 2, (
        "non-fused n_tokens must be 2, got %r" % (_get(r_nf, "n_tokens"),))
    assert _get(r_nf, "extra_token") is None, (
        "non-fused extra_token must be None")
    # ★ kept_positions 在非 fuse 分支也必须是【原序列位置】(i + 1)
    kp_nf = [int(i) for i in _get(r_nf, "kept_positions")]
    assert kp_nf == [3, 1], (
        "kept_positions must be the original positions [3, 1] in the "
        "non-fused branch too, got %s" % kp_nf)

    # ---- 单通道 -------------------------------------------------------
    h1 = [[0.0], [1.0], [2.0], [3.0]]
    r1 = solve("reorganize", hidden=h1, idx=[2, 1], extra_token=[7.0],
               fuse=True)
    got1 = _f(_get(r1, "out"))
    ok, msg = _close(got1, [0.0, 3.0, 2.0, 7.0])
    assert ok, "single-channel reorganize mismatch: %s" % msg

    # ---- idx 覆盖全部 patch（输出只剩 cls + extra） ---------------------
    r_all = solve("reorganize", hidden=HIDDEN, idx=[0, 1, 2, 3],
                   extra_token=[1.0, 2.0], fuse=True)
    got_all = _f(_get(r_all, "out"))
    ok, msg = _close(got_all, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                               7.0, 8.0, 1.0, 2.0])
    assert ok, "full-idx reorganize mismatch: %s" % msg
    assert int(_get(r_all, "n_tokens")) == 5, (
        "full-idx n_tokens must be 5, got %r" % (_get(r_all, "n_tokens"),))
