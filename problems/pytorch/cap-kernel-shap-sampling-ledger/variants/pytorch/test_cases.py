"""849 判题用例：Captum Kernel SHAP 采样与评估次数账。

五个 check_* 函数，均以 solve 作为唯一参数（judge 内部以 fn(solve) 调用）：
  check_select_distribution —— p(k) = (M-1)/(k*(M-k)) 的构造、probs[0]=0、
                               float32 口径与 M=1 的 NaN 退化
  check_similarity_weight  —— 端点 inf_weight 与内部 1.0 的相似度权重
  check_perturb_sample     —— 二进制扰动向量的生成（kthvalue 阈值 + 严格大于）
  check_perturb_sequence   —— 两个固定端点 + 随机样本的完整序列
  check_n_evaluations      —— 前向评估次数（采样版 / 精确版 / 全排列枚举）

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


def _oracle_probs(M):
    """官方 p(k) 的构造：denom = k*(M-k)，probs = (M-1)/denom，probs[0]=0。"""
    out = []
    for k in range(M):
        d = k * (M - k)
        out.append(0.0 if d == 0 else (M - 1) / d)
    out[0] = 0.0
    return out


def _oracle_normalized(probs):
    s = sum(probs)
    if s == 0:
        return None
    return [v / s for v in probs]


def _oracle_similarity(sample, M):
    """官方：端点（全 0 / 全 1）→ inf_weight=1e6，其余 → 1.0。"""
    n = sum(sample)
    if n == 0 or n == M:
        return 1000000.0
    return 1.0


def _oracle_perturb(rand_vals, k, M):
    """官方：threshold = 第 (M-k) 小，比较是【严格大于】。"""
    ordered = sorted(rand_vals)
    threshold = ordered[M - k - 1]
    return [1 if v > threshold else 0 for v in rand_vals], threshold


def _oracle_n_eval(total, n_samples, ppe, exact):
    batches = -(-total // ppe)
    factor = math.factorial(total) if exact else n_samples
    return batches * factor


# --------------------------------------------------------------------------
# mode 1: select_distribution
# --------------------------------------------------------------------------


def check_select_distribution(solve):
    """select_distribution：p(k) 的构造、probs[0]=0、float32 口径、M=1 退化。"""

    for M in [2, 3, 5, 8]:
        r = solve("select_distribution", num_interp_features=M)
        probs = _f(_get(r, "probs"))
        exp = _oracle_probs(M)
        ok, msg = _close(probs, exp)
        assert ok, "probs mismatch at M=%d: %s" % (M, msg)

        # ★ 长度必须是 M（k 取 0..M-1，不含 M）
        assert len(probs) == M, (
            "probs length must be M = %d (k ranges over 0..M-1), got %d"
            % (M, len(probs)))

        # ★ denom = k*(M-k)
        denom = _f(_get(r, "denom"))
        exp_denom = [float(k * (M - k)) for k in range(M)]
        ok, msg = _close(denom, exp_denom)
        assert ok, "denom mismatch at M=%d: %s" % (M, msg)

        # ★ probs[0] 必须是 0（不是 inf）
        assert probs[0] == 0.0, (
            "probs[0] must be 0.0 (official overwrites the inf), got %r"
            % (probs[0],))

        # ★ probs 不归一化：M=5 时总和是 10/3，不是 1
        total = float(_get(r, "total"))
        ok, msg = _close([total], [sum(exp)])
        assert ok, "total mismatch at M=%d: %s" % (M, msg)
        if M == 5:
            assert abs(total - 10.0 / 3.0) < 1e-12, (
                "probs must NOT be normalized: total must be 10/3, got %.12f"
                % total)
            _must_differ([total], [1.0],
                         "the raw probs sum must not be 1")

        # normalized（归一化后）和为 1
        norm = _f(_get(r, "normalized"))
        exp_norm = _oracle_normalized(exp)
        ok, msg = _close(norm, exp_norm)
        assert ok, "normalized mismatch at M=%d: %s" % (M, msg)
        assert abs(sum(norm) - 1.0) < 1e-12, (
            "normalized must sum to 1, got %.12f" % sum(norm))

        # ★ k=M-1 的 probs 是 1.0（不是无穷）
        if M >= 2:
            assert abs(probs[M - 1] - 1.0) < 1e-12, (
                "probs[M-1] must be exactly 1.0 (denom=(M-1)*1), got %.12f"
                % probs[M - 1])

        # ★ 双向断言：k=M-1 若按「理论无穷」处理会得到 inf
        _must_differ([probs[M - 1]], [float("inf")],
                     "probs[M-1] must be 1.0, not inf")

        # ★ 采样范围：k 只能是 1..M-1（probs[0]=0 且 k=M 不在范围内）
        rng = [int(v) for v in _get(r, "sampling_range")]
        assert rng == [1, M - 1], (
            "sampling_range must be [1, M-1] = [1, %d], got %s"
            % (M - 1, rng))

    # ---- ★ float32 口径：官方全程 float32 --------------------------------
    # 官方：torch.arange(M, dtype=torch.float) ⇒ denom / probs 都是 float32
    # M=16 时精确值与 float32 值差约 4.3e-08
    r16 = solve("select_distribution", num_interp_features=16)
    f32 = _f(_get(r16, "probs_f32"))
    exact16 = _oracle_probs(16)
    assert len(f32) == 16
    # float32 口径必须与精确值「接近但不相等」—— 用一个能区分的 M
    diff = max(abs(a - b) for a, b in zip(f32, exact16))
    assert diff > 0.0, (
        "probs_f32 must be the float32-rounded values (they differ from the "
        "exact rationals)")
    assert diff < 1e-6, (
        "probs_f32 must be within float32 precision of the exact values, "
        "diff = %.3e" % diff)

    # ---- ★ M=1 的退化：probs 全 0 ⇒ 归一化出 NaN -------------------------
    r1 = solve("select_distribution", num_interp_features=1)
    p1 = _f(_get(r1, "probs"))
    assert p1 == [0.0], "M=1 probs must be [0.0], got %s" % p1
    assert float(_get(r1, "total")) == 0.0, (
        "M=1 total must be 0, got %r" % (_get(r1, "total"),))
    assert _get(r1, "normalized") is None, (
        "M=1 normalized must be None (official Categorical raises on the "
        "NaN simplex)")
    assert _get(r1, "normalized_f32") is None, (
        "M=1 normalized_f32 must be None too")
    rng1 = [int(v) for v in _get(r1, "sampling_range")]
    assert rng1 == [0, -1], (
        "M=1 sampling_range must be an empty interval [0, -1], got %s" % rng1)


# --------------------------------------------------------------------------
# mode 2: similarity_weight
# --------------------------------------------------------------------------


def check_similarity_weight(solve):
    """similarity_weight：端点 1e6、内部 1.0。"""

    for M in [2, 3, 5]:
        for k in range(M + 1):
            sample = [1] * k + [0] * (M - k)
            r = solve("similarity_weight", interpretable_sample=sample,
                      num_interp_features=M)
            w = float(_get(r, "weight"))
            exp = _oracle_similarity(sample, M)
            assert abs(w - exp) < 1e-12, (
                "M=%d k=%d: weight %.6f != %.6f" % (M, k, w, exp))

            assert int(_get(r, "num_selected")) == k, (
                "num_selected must be %d, got %r" % (k, _get(r, "num_selected")))
            assert bool(_get(r, "is_endpoint")) == (k == 0 or k == M), (
                "is_endpoint must be True only for k=0 or k=M")

    # ---- ★ 端点权重必须是 1000000.0（不是 inf）---------------------------
    r0 = solve("similarity_weight", interpretable_sample=[0, 0, 0],
               num_interp_features=3)
    w0 = float(_get(r0, "weight"))
    assert w0 == 1000000.0, (
        "endpoint weight must be exactly 1000000.0 (inf_weight), got %r" % w0)
    # ★ 双向断言：不是真正的无穷
    assert w0 != float("inf"), (
        "the endpoint weight must be the finite 1e6 surrogate, not inf")

    r1 = solve("similarity_weight", interpretable_sample=[1, 1, 1],
               num_interp_features=3)
    assert float(_get(r1, "weight")) == 1000000.0, (
        "the all-ones endpoint must also get 1e6")

    # ---- 中间样本权重必须是 1.0 ------------------------------------------
    rm = solve("similarity_weight", interpretable_sample=[1, 0, 1],
               num_interp_features=3)
    assert float(_get(rm, "weight")) == 1.0, (
        "interior samples must get weight 1.0, got %r" % (_get(rm, "weight"),))

    # ---- ★ 权重只取决于「选中个数」，与位置无关 --------------------------
    a = solve("similarity_weight", interpretable_sample=[1, 1, 0, 0],
              num_interp_features=4)
    b = solve("similarity_weight", interpretable_sample=[0, 0, 1, 1],
              num_interp_features=4)
    assert float(_get(a, "weight")) == float(_get(b, "weight")) == 1.0, (
        "the weight must depend only on the selected count, not the positions")

    # ---- 自定义 inf_weight -----------------------------------------------
    rc = solve("similarity_weight", interpretable_sample=[0, 0],
               num_interp_features=2, inf_weight=42.0)
    assert float(_get(rc, "weight")) == 42.0, (
        "a custom inf_weight must be honored, got %r" % (_get(rc, "weight"),))

    # ---- 域约束：长度必须等于 num_interp_features ------------------------
    raised = False
    try:
        solve("similarity_weight", interpretable_sample=[1, 0],
              num_interp_features=3)
    except ValueError:
        raised = True
    assert raised, (
        "a sample whose length differs from num_interp_features must be "
        "rejected")

    # ---- 域约束：必须是 0/1 向量 -----------------------------------------
    raised = False
    try:
        solve("similarity_weight", interpretable_sample=[1, 2, 0],
              num_interp_features=3)
    except ValueError:
        raised = True
    assert raised, "a non-binary interpretable sample must be rejected"


# --------------------------------------------------------------------------
# mode 3: perturb_sample
# --------------------------------------------------------------------------

# (rand_vals, k, M)
P_CASES = [
    ([1.541, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986], 2, 6),
    ([0.6614, 0.2669, 0.0617, 0.6213, -0.4519, -0.1661], 2, 6),
    ([0.3923, -0.2236, -0.3195, -1.205, 1.0445, -0.6332], 2, 6),
    ([0.8033, 0.1748, 0.089, -0.6137, 0.0462, -1.3683], 2, 6),
    ([-1.6053, 0.2325, 2.2399, 0.8473, 1.2006, -0.4016], 2, 6),
    ([-0.4868, -0.6038, -0.5581, 0.6675, -0.1974, 1.9428], 2, 6),
    ([3.0, 2.0, 1.0], 1, 3),
    ([3.0, 2.0, 1.0], 2, 3),
    ([5.0, 5.0, 5.0, 1.0], 2, 4),           # 并列 ⇒ 少选
    ([1.0, 2.0, 3.0, 4.0, 5.0], 1, 5),
    ([1.0, 2.0, 3.0, 4.0, 5.0], 4, 5),
    ([0.5, 0.5, 0.5, 0.5], 2, 4),           # 全并列 ⇒ 少选
    ([1.0, 2.0], 1, 2),
    ([-1.0, 0.0, 1.0, 2.0, 3.0], 3, 5),
]


def check_perturb_sample(solve):
    """perturb_sample：kthvalue 阈值 + 严格大于的 top-k 选择。"""

    for ci, (rv, k, M) in enumerate(P_CASES):
        r = solve("perturb_sample", rand_vals=rv, num_selected_features=k,
                  num_interp_features=M)
        sample = [int(v) for v in _get(r, "sample")]
        exp, exp_t = _oracle_perturb(rv, k, M)

        ok, msg = _close(sample, exp)
        assert ok, "perturb_sample#%d mismatch: %s" % (ci, msg)

        # ★ 阈值是第 (M-k) 小（不是第 k 大）
        assert abs(float(_get(r, "threshold")) - exp_t) < 1e-12, (
            "perturb_sample#%d threshold %.12f != %.12f"
            % (ci, float(_get(r, "threshold")), exp_t))

        assert int(_get(r, "num_selected")) == sum(exp), (
            "perturb_sample#%d num_selected must be %d, got %r"
            % (ci, sum(exp), _get(r, "num_selected")))

    # ---- ★ 双向断言：第 k 大的候选必须不同 ------------------------------
    # 用一组非对称数据，让「第 (M-k) 小」与「第 k 大」落在不同的值上
    rv = [9.0, 1.0, 7.0, 3.0, 5.0]
    k, M = 2, 5
    r = solve("perturb_sample", rand_vals=rv, num_selected_features=k,
              num_interp_features=M)
    sample = _f(_get(r, "sample"))
    wrong_threshold = sorted(rv)[k - 1]          # 第 k 大 = 7.0
    wrong = [1 if v > wrong_threshold else 0 for v in rv]
    _must_differ(sample, wrong,
                 "the threshold must be the (M-k)-th smallest, "
                 "not the k-th largest")
    # 反向确认：正确阈值下选中的确实不同
    right_threshold = sorted(rv)[M - k - 1]       # 第 3 小 = 5.0
    right = [1 if v > right_threshold else 0 for v in rv]
    assert sample == right, (
        "the (M-k)-th smallest must be the threshold, got %r" % (sample,))

    # ---- ★ 并列时【少选】：严格大于 vs 大于等于 --------------------------
    r = solve("perturb_sample", rand_vals=[5.0, 5.0, 5.0, 1.0],
              num_selected_features=2, num_interp_features=4)
    assert int(_get(r, "num_selected")) == 0, (
        "with tied values, the strict > comparison must select 0, got %r"
        % (_get(r, "num_selected"),))
    # ★ 双向断言：用 >= 会选 3 个
    ge = [1 if v >= 0.0 else 0 for v in [5.0, 5.0, 5.0, 1.0]]
    _must_differ(_f(_get(r, "sample")), ge,
                 "the comparison must be strictly greater than")

    r = solve("perturb_sample", rand_vals=[0.5, 0.5, 0.5, 0.5],
              num_selected_features=2, num_interp_features=4)
    assert int(_get(r, "num_selected")) == 0, (
        "fully tied values must select 0 under the strict > comparison")

    # ---- 无并列时正好选 k 个 ---------------------------------------------
    for rv, k, M in ([3.0, 2.0, 1.0], 1, 3), ([3.0, 2.0, 1.0], 2, 3), \
                     ([1.0, 2.0, 3.0, 4.0, 5.0], 4, 5):
        r = solve("perturb_sample", rand_vals=rv, num_selected_features=k,
                  num_interp_features=M)
        assert int(_get(r, "num_selected")) == k, (
            "without ties exactly k features must be selected, got %r"
            % (_get(r, "num_selected"),))

    # ---- ★ 域约束：k == M 必须被拒绝（官方 kthvalue 索引 0 越界）--------
    raised = False
    try:
        solve("perturb_sample", rand_vals=[1.0, 2.0], num_selected_features=2,
              num_interp_features=2)
    except ValueError:
        raised = True
    assert raised, (
        "num_selected_features == num_interp_features must be rejected "
        "(official kthvalue index 0 is out of range)")

    # ---- 域约束：rand_vals 长度必须等于 M --------------------------------
    raised = False
    try:
        solve("perturb_sample", rand_vals=[1.0, 2.0], num_selected_features=1,
              num_interp_features=3)
    except ValueError:
        raised = True
    assert raised, "rand_vals whose length differs from M must be rejected"


# --------------------------------------------------------------------------
# mode 4: perturb_sequence
# --------------------------------------------------------------------------


def check_perturb_sequence(solve):
    """perturb_sequence：两个固定端点 + 随机样本的完整序列。"""

    for M in [1, 3, 6]:
        r = solve("perturb_sample", kind="endpoints",
                  num_interp_features=M)
        samples = _get(r, "samples")
        assert len(samples) == 2, (
            "the endpoint list must have exactly 2 entries, got %d"
            % len(samples))
        assert samples[0] == [1] * M, (
            "the first endpoint must be all-ones, got %s" % (samples[0],))
        assert samples[1] == [0] * M, (
            "the second endpoint must be all-zeros, got %s" % (samples[1],))
        assert int(_get(r, "num_endpoints")) == 2

    # ---- 完整序列 -------------------------------------------------------
    r = solve("perturb_sample", kind="sequence",
              rand_vals_list=[[3.0, 2.0, 1.0], [1.0, 5.0, 2.0]],
              selected_counts=[1, 2], num_interp_features=3)
    samples = _get(r, "samples")
    assert len(samples) == 4, (
        "the sequence must have 2 endpoints + 2 random samples = 4, got %d"
        % len(samples))
    assert samples[0] == [1, 1, 1], "first entry must be the all-ones endpoint"
    assert samples[1] == [0, 0, 0], "second entry must be the all-zeros endpoint"

    exp0, _ = _oracle_perturb([3.0, 2.0, 1.0], 1, 3)
    exp1, _ = _oracle_perturb([1.0, 5.0, 2.0], 2, 3)
    assert samples[2] == exp0, (
        "the first random sample mismatch: %s != %s" % (samples[2], exp0))
    assert samples[3] == exp1, (
        "the second random sample mismatch: %s != %s" % (samples[3], exp1))
    assert int(_get(r, "num_endpoints")) == 2
    assert int(_get(r, "num_random")) == 2

    # ★ 双向断言：端点必须在【最前】，不能在最后
    wrong_order = samples[2:] + samples[:2]
    _must_differ([x for s in samples for x in s],
                 [x for s in wrong_order for x in s],
                 "the two endpoints must come first")

    # ---- 域约束：两个列表长度必须一致 ------------------------------------
    raised = False
    try:
        solve("perturb_sample", kind="sequence",
              rand_vals_list=[[1.0, 2.0]], selected_counts=[1, 2],
              num_interp_features=2)
    except ValueError:
        raised = True
    assert raised, "mismatched list lengths must be rejected"


# --------------------------------------------------------------------------
# mode 5: n_evaluations
# --------------------------------------------------------------------------


def check_n_evaluations(solve):
    """n_evaluations：采样版 ceil*n_samples vs 精确版 ceil*factorial。"""

    for total in [3, 5, 8]:
        for ns in [1, 4, 25]:
            for ppe in [1, 2, 3, 7]:
                # ---- 采样版 ------------------------------------------------
                r = solve("n_evaluations", total_features=total,
                          n_samples=ns, perturbations_per_eval=ppe)
                v = int(_get(r, "n_evaluations"))
                exp = _oracle_n_eval(total, ns, ppe, False)
                assert v == exp, (
                    "sampling n_evaluations(%d, %d, %d) = %d, expected %d"
                    % (total, ns, ppe, v, exp))
                assert _get(r, "mode") == "sampling"
                assert int(_get(r, "permutation_factor")) == ns

                # ---- 精确版 ------------------------------------------------
                re_ = solve("n_evaluations", total_features=total,
                            n_samples=ns, perturbations_per_eval=ppe,
                            exact=True)
                ve = int(_get(re_, "n_evaluations"))
                expe = _oracle_n_eval(total, ns, ppe, True)
                assert ve == expe, (
                    "exact n_evaluations(%d, %d, %d) = %d, expected %d"
                    % (total, ns, ppe, ve, expe))
                assert _get(re_, "mode") == "exact"
                assert int(_get(re_, "permutation_factor")) == math.factorial(total)

                # ★ 双向断言：精确版必须【忽略 n_samples】
                if ns != 1:
                    r1 = solve("n_evaluations", total_features=total,
                               n_samples=1, perturbations_per_eval=ppe,
                               exact=True)
                    assert int(_get(r1, "n_evaluations")) == ve, (
                        "the exact variant must ignore n_samples entirely")

    # ---- ★ ceil 在【乘法之前】 -------------------------------------------
    # total=5, ppe=2 ⇒ ceil(5/2)=3；total=5, ppe=3 ⇒ ceil(5/3)=2
    r2 = solve("n_evaluations", total_features=5, n_samples=10,
               perturbations_per_eval=2)
    assert int(_get(r2, "n_evaluations")) == 30, (
        "must be ceil(5/2) * 10 = 30, got %r" % (_get(r2, "n_evaluations"),))
    r3 = solve("n_evaluations", total_features=5, n_samples=10,
               perturbations_per_eval=3)
    assert int(_get(r3, "n_evaluations")) == 20, (
        "must be ceil(5/3) * 10 = 20, got %r" % (_get(r3, "n_evaluations"),))
    # ★ 双向断言：floor 会得到 2*10=20 / 1*10=10，与正确值不同
    assert int(_get(r2, "n_evaluations")) != 20, (
        "the batch count must be a ceiling, not a floor")

    # ---- ★ ppe 不影响乘法因子 -------------------------------------------
    a = solve("n_evaluations", total_features=8, n_samples=4,
              perturbations_per_eval=1)
    b = solve("n_evaluations", total_features=8, n_samples=4,
              perturbations_per_eval=3)
    assert int(_get(a, "permutation_factor")) == \
        int(_get(b, "permutation_factor")) == 4, (
        "perturbations_per_eval must not touch the multiplication factor")

    # ---- 全排列枚举 -----------------------------------------------------
    for M in [1, 2, 3, 5]:
        r = solve("all_permutations", num_features=M, num_samples=999)
        perms = _get(r, "permutations")
        assert int(_get(r, "count")) == math.factorial(M), (
            "the permutation count must be M! = %d, got %r"
            % (math.factorial(M), _get(r, "count")))
        assert len(perms) == math.factorial(M)
        # ★ 每个排列必须是 0..M-1 的一个置换
        for p in perms:
            assert sorted(p) == list(range(M)), (
                "each permutation must be a permutation of 0..M-1, got %s" % p)
        # ★ num_samples 被完全忽略
        r2 = solve("all_permutations", num_features=M, num_samples=1)
        assert int(_get(r2, "count")) == math.factorial(M), (
            "num_samples must be ignored by the exhaustive permutation "
            "generator, got %r" % (_get(r2, "count"),))
        assert bool(_get(r2, "num_samples_ignored")) is True
        # 字典序：第一项是升序
        assert [int(v) for v in perms[0]] == list(range(M)), (
            "the permutations must be in itertools order (identity first), "
            "got %s" % (perms[0],))

    # ---- 域约束：total_features 必须 >= 1 --------------------------------
    raised = False
    try:
        solve("n_evaluations", total_features=0, n_samples=4,
              perturbations_per_eval=1)
    except ValueError:
        raised = True
    assert raised, "total_features = 0 must be rejected"
