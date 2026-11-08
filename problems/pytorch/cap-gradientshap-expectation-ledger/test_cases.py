"""853 判题用例（judge 注入 solve）。

约定：顶层 check_* 函数由 judge 发现并以 fn(solve) 调用；本文件不 import
solution，不依赖 captum / torch / numpy，只用标准库。
"""
import math


# ------------------------------------------------------------------ 工具
def _must(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _eq(a, b, tol=1e-9):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        if len(a) != len(b):
            return False
        return all(_eq(x, y, tol) for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def _raises(fn, *needles):
    try:
        fn()
    except Exception as e:
        msg = str(e)
        for n in needles:
            _must(n in msg, "exception message %r missing keyword %r" % (msg, n))
        return
    _must(False, "expected an exception containing %r" % (needles,))


def _case(bsz=1, d=3, k=2, forward_kind="quadratic", mul=True,
          stdevs=0.0, ret=True, tup=False, num_ref=4, seed=1):
    n_in = 2 if tup else 1

    def mk_in(off):
        return [[float((i + 1) * 0.5 + t * 0.25 + off) for t in range(d)]
                for i in range(bsz)]

    def mk_bl(off):
        return [[float(j * 0.5 - t * 0.25 + off) for t in range(d)]
                for j in range(num_ref)]

    inputs = [mk_in(0.0), mk_in(2.0)] if tup else mk_in(0.0)
    baselines = [mk_bl(0.0), mk_bl(1.0)] if tup else mk_bl(0.0)
    w = [float(0.5 + t * 0.25) for t in range(d)]
    indices = [(i * 3 + seed) % num_ref for i in range(k * bsz * n_in)]
    alphas = [0.25 + ((i * 2 + seed) % 3) * 0.25 for i in range(k * bsz)]
    # 每个元素都不同：常数噪声会让多 tensor 的 noise 切片与
    # (i*k+j)/(j*bsz+i) 布局变异退化为不可观测（历史教训）。
    noises = [((i * 3 + seed) % 7 - 3) * 0.25
              for i in range(k * bsz * d * n_in)]
    return dict(
        inputs=inputs, baselines=baselines, w=w, forward_kind=forward_kind,
        n_samples=k, stdevs=float(stdevs), multiply_by_inputs=mul,
        return_convergence_delta=ret, tuple_inputs=tup,
        indices=indices, alphas=alphas, noises=noises,
    )


def _exp_gs(cfg):
    """独立期望实现（多项式展开 + 逐样本显式循环 + 保存中间量），
    与被测 solve 的分步列表推导路径不同。"""
    w = list(cfg["w"])
    fk = cfg["forward_kind"]
    k = cfg["n_samples"]
    mul = cfg["multiply_by_inputs"]
    ret = cfg["return_convergence_delta"]
    tup = cfg["tuple_inputs"]
    ins = cfg["inputs"] if tup else [cfg["inputs"]]
    bls = cfg["baselines"] if tup else [cfg["baselines"]]
    bsz = len(ins[0])
    d = len(ins[0][0])
    n_in = len(ins)
    idx = cfg["indices"]
    alp = cfg["alphas"]
    noi = cfg["noises"]

    all_pert = []
    all_drawn = []
    all_attr = []
    for ti in range(n_in):
        x = ins[ti]
        bl = bls[ti]
        idt = idx[ti * k * bsz:(ti + 1) * k * bsz]
        nt = noi[ti * k * bsz * d:(ti + 1) * k * bsz * d]
        pert_rows = []
        drawn_rows = []
        attr_rows = []
        for i in range(bsz):
            for j in range(k):
                r = i * k + j
                pert = [x[i][t] + nt[r * d + t] for t in range(d)]
                b = list(bl[idt[r]])
                a = alp[r]
                row = []
                for t in range(d):
                    sc = a * pert[t] + (1 - a) * b[t]
                    gd = w[t] if fk == "linear" else 2 * w[t] * sc
                    row.append((pert[t] - b[t]) * gd if mul else gd)
                pert_rows.append(pert)
                drawn_rows.append(b)
                attr_rows.append(row)
        all_pert.append(pert_rows)
        all_drawn.append(drawn_rows)
        all_attr.append(attr_rows)

    outs = []
    for ti in range(n_in):
        outs.append([
            [sum(all_attr[ti][i * k + j][t] for j in range(k)) / k
             for t in range(d)]
            for i in range(bsz)
        ])
    out = {"attributions": outs if tup else outs[0]}
    if ret:
        delta = []
        for r in range(k * bsz):
            attr_sum = sum(all_attr[ti][r][t]
                           for ti in range(n_in) for t in range(d))
            f_end = 0.0
            f_start = 0.0
            for ti in range(n_in):
                for t in range(d):
                    if fk == "linear":
                        f_end += w[t] * all_pert[ti][r][t]
                        f_start += w[t] * all_drawn[ti][r][t]
                    else:
                        f_end += w[t] * all_pert[ti][r][t] ** 2
                        f_start += w[t] * all_drawn[ti][r][t] ** 2
            delta.append(attr_sum - (f_end - f_start))
        out["delta"] = delta
    return out


# ================================================================ check 1
def check_baseline_draw(solve):
    """np.random.choice 有放回 / 扁平 k*bsz 索引 / 非 distrib 的 repeat_interleave。"""
    bl = [[0., 0., 0.], [10., 10., 10.], [20., 20., 20.], [30., 30., 30.]]
    idx = [0, 1, 2, 3, 3, 2, 1, 0]
    r = solve("baseline_draw", baselines=bl, indices=idx, n_samples=4, bsz=2)
    _must(r["num_ref"] == 4, "num_ref")
    _must(r["feature_dim"] == 3, "feature_dim")
    _must(r["k"] == 4 and r["bsz"] == 2, "k/bsz")
    _must(r["expected_indices"] == 8, "expected_indices must be k*bsz")
    _must(r["indices"] == idx, "indices must be returned verbatim")
    _must(r["layout"] == "flat_k_times_bsz", "distrib layout")
    _must(r["replace"] is True, "np.random.choice defaults to replace=True")
    _must(r["rng"] == "np.random.choice(num_ref, k*bsz)", "rng tag")
    _must(r["validate"] == "shape[1:] only (distrib)", "distrib validation")
    _must(r["drawn_shape"] == [8, 3], "drawn shape")
    _must(_eq(r["drawn"], [bl[i] for i in idx]), "drawn == baseline[indices]")
    # 有放回：索引可重复
    r_rep = solve("baseline_draw", baselines=bl, indices=[2, 2, 2, 2],
                  n_samples=2, bsz=2)
    _must(_eq(r_rep["drawn"], [bl[2]] * 4), "with-replacement draws")

    # 非 distrib：repeat_interleave(k)（仅当 num_ref == bsz 且 > 1）
    bl2 = [[1., 1., 1.], [2., 2., 2.]]
    r2 = solve("baseline_draw", baselines=bl2,
               indices=[0, 1, 0, 1, 0, 1, 0, 1], n_samples=4, bsz=2,
               draw_baseline_from_distrib=False)
    _must(r2["layout"] == "repeat_interleave", "non-distrib layout")
    _must(r2["non_distrib"]["applicable"] is True, "non-distrib applicable")
    _must(_eq(r2["drawn"], [[1., 1., 1.]] * 4 + [[2., 2., 2.]] * 4),
          "repeat_interleave order")
    # 非 distrib 不适用时原样
    r3 = solve("baseline_draw", baselines=bl, indices=idx, n_samples=4,
               bsz=2, draw_baseline_from_distrib=False)
    _must(r3["layout"] == "unchanged", "non-distrib unchanged layout")
    _must(r3["non_distrib"]["applicable"] is False, "non-distrib not applicable")
    _must(r3["drawn_shape"] == [4, 3],
          "unchanged drawn_shape must stay [num_ref, d], not [num_ref*k, d]")
    _must(len(r3["drawn"]) == 4, "unchanged drawn rows must equal num_ref")

    # num_ref == bsz == 1 时官方 applicable 仍需 num_ref > 1，故 unchanged
    r4 = solve("baseline_draw", baselines=[[7., 8., 9.]],
               indices=[0, 0], n_samples=2, bsz=1,
               draw_baseline_from_distrib=False)
    _must(r4["non_distrib"]["applicable"] is False,
          "num_ref==bsz==1 must NOT be applicable (needs num_ref > 1)")
    _must(r4["layout"] == "unchanged", "bsz=1 non-distrib layout")
    _must(r4["drawn_shape"] == [1, 3], "bsz=1 unchanged drawn_shape")
    _must(_eq(r4["drawn"], [[7., 8., 9.]]), "bsz=1 unchanged drawn")

    # 域约束
    _raises(lambda: solve("baseline_draw", baselines=bl,
                          indices=[0, 1, 2, 9], n_samples=2, bsz=2),
            "baseline index out of range")
    _raises(lambda: solve("baseline_draw", baselines=bl,
                          indices=[0.0, 1.0], n_samples=1, bsz=2),
            "indices must be ints")
    _raises(lambda: solve("baseline_draw", baselines=bl,
                          indices=[0, 1], n_samples=4, bsz=2),
            "indices length")
    _raises(lambda: solve("baseline_draw", baselines=bl, indices=idx,
                          n_samples=4, bsz=2, input_feature_dim=4),
            "shape", "baseline")


# ================================================================ check 2
def check_scale_input(solve):
    """alpha.view((n,1,1,..)) 广播 + alpha*x+(1-alpha)*b + requires_grad_。"""
    x = [[1., 2., 3.], [4., 5., 6.]]
    b = [[0., 0., 0.], [1., 1., 1.]]
    a = [0.25, 0.75]
    r = solve("scale_input", inputs=x, baselines=b, alphas=a)
    _must(r["batch"] == 2 and r["feature_dim"] == 3, "batch/feature_dim")
    _must(r["coeff_shape"] == [2, 1], "2D coeff view (n,1)")
    _must(_eq(r["scaled"], [[0.25, 0.5, 0.75], [3.25, 4.0, 4.75]]),
          "scaled interpolation values")
    _must(r["interpolation"] == "alpha*input + (1-alpha)*baseline",
          "interpolation formula tag")
    _must(r["rng"] == "np.random.uniform(0.0, 1.0, batch)", "rng tag")
    _must(r["requires_grad"] is True, "_scale_input calls requires_grad_()")
    _must(solve("scale_input", inputs=x, baselines=b, alphas=a,
                ndim=3)["coeff_shape"] == [2, 1, 1], "3D coeff view")
    _must(solve("scale_input", inputs=x, baselines=b, alphas=a,
                ndim=4)["coeff_shape"] == [2, 1, 1, 1], "4D coeff view")
    r2 = solve("scale_input", inputs=[[2.0, 4.0]], baselines=[[0.0, 2.0]],
               alphas=[0.5])
    _must(_eq(r2["scaled"], [[1.0, 3.0]]), "alpha=0.5 exact midpoint")

    _raises(lambda: solve("scale_input", inputs=x, baselines=b,
                          alphas=[1.0, 0.5]), "0 <= a < 1")
    _raises(lambda: solve("scale_input", inputs=x, baselines=b,
                          alphas=[-0.1, 0.5]), "0 <= a < 1")
    _raises(lambda: solve("scale_input", inputs=x, baselines=b,
                          alphas=[0.5]), "alphas length")
    _raises(lambda: solve("scale_input", inputs=x,
                          baselines=[[0., 0.], [1., 1.]],
                          alphas=a), "baselines feature dim")
    _raises(lambda: solve("scale_input", inputs=x, baselines=b, alphas=a,
                          ndim=1), "ndim must be")
    _raises(lambda: solve("scale_input", inputs=x,
                          baselines=[[0., 0., 0.]],
                          alphas=a), "baselines rows must match inputs rows")
    _raises(lambda: solve("scale_input", inputs=[[1., 2.], [3.]],
                          baselines=[[0., 0.], [1., 1.]],
                          alphas=[0.5, 0.5]),
            "inputs rows must have equal length")


# ================================================================ check 3
def check_noise_layout(solve):
    """repeat_interleave(k)+normal 布局 / stdevs 必须是 float 或等长 tuple。"""
    inp = [[1., 2., 3.], [4., 5., 6.]]
    noi = [0.5, -0.25, 1.0, -1.0, 0.25, -0.5,
           2.0, -2.0, 0.75, -0.75, 0.125, -0.125,
           0.375, -0.375, 0.625, -0.625, 0.875, -0.875]
    r = solve("noise_layout", inputs=inp, k=3, noises=noi, stdevs=0.5)
    _must(r["perturbed_shape"] == [6, 3], "perturbed shape (bsz*k,d)")
    _must(r["normal_shape"] == [6, 3], "normal shape")
    _must(r["expected_noises"] == 18, "expected noises = bsz*k*d")
    _must(r["layout"] == "repeat_interleave(k) + normal", "layout tag")
    _must(r["requires_grad"] is True, "gradient method requires_grad_()")
    _must(r["stdevs"] == 0.5, "float stdevs")
    exp = []
    for row in inp:
        for j in range(3):
            base = len(exp) * 3
            exp.append([row[t] + noi[base + t] for t in range(3)])
    _must(_eq(r["perturbed"], exp), "perturbed == interleave + noise (C order)")
    # 与 repeat() 布局区分：第 1 行仍来自 inp[0]
    _must(_eq(r["perturbed"][1],
              [inp[0][t] + noi[3 + t] for t in range(3)]),
          "row 1 comes from inp[0] (interleave)")
    _must(not _eq(r["perturbed"][1],
                  [inp[1][t] + noi[3 + t] for t in range(3)]),
          "row 1 is NOT inp[1] (repeat() layout)")

    r2 = solve("noise_layout", inputs=inp, k=3, noises=noi * 2,
               stdevs=(0.5, 0.25), n_in=2)
    _must(_eq(r2["stdevs"], [0.5, 0.25]), "tuple stdevs echoed")
    _must(r2["expected_noises"] == 36, "tuple stdevs expected noises")

    _raises(lambda: solve("noise_layout", inputs=inp, k=3, noises=noi,
                          stdevs=1), "stdevs must be type float")
    _raises(lambda: solve("noise_layout", inputs=inp, k=3, noises=noi,
                          stdevs=(0.5, 0.25)), "number of stdevs")
    _raises(lambda: solve("noise_layout", inputs=inp, k=3, noises=noi,
                          stdevs=(1,)), "each stdev")
    _raises(lambda: solve("noise_layout", inputs=inp, k=3, noises=noi[:5],
                          stdevs=0.5), "noises length")
    _raises(lambda: solve("noise_layout", inputs=inp, k=0,
                          noises=[], stdevs=0.5), "k must be positive")
    _raises(lambda: solve("noise_layout", inputs=[[1., 2.]], k=1,
                          noises=[0.0, True], stdevs=0.5),
            "noises must be finite numbers")


# ================================================================ check 4
def check_schedule(solve):
    """GradientShap 单 partition / numpy-torch 双流 / 各 RNG 调用次数。"""
    r = solve("schedule", n_samples=5, bsz=2, d=3, n_in=1, num_ref=4)
    _must(r["partitions"] == 1, "GradientShap never forwards batch_size -> 1 partition")
    _must(r["remaining"] == 0, "no remainder")
    _must(r["nt_samples_batch_size"] == 5, "effective batch == n_samples")
    _must(r["batch_size_forwarded"] is False, "batch_size not forwarded")
    _must(r["rng_order"] == ["choice", "normal", "uniform"], "RNG order")
    _must(r["choice_calls"] == [[4, 10, True]], "choice(num_ref, k*bsz, True)")
    _must(r["uniform_calls"] == [[0.0, 1.0, 10]], "uniform(0,1,k*bsz)")
    _must(r["normal_shapes"] == [[10, 3]], "normal shape (k*bsz,d)")
    _must(r["np_stream_shared"] is True, "choice & uniform share numpy RNG")
    _must(r["torch_stream_independent"] is True, "normal uses torch RNG")
    _must(r["expected_indices"] == 10 and r["expected_alphas"] == 10
          and r["expected_noises"] == 30, "expected sequence lengths")

    r2 = solve("schedule", n_samples=3, bsz=1, d=2, n_in=2, num_ref=5)
    _must(r2["rng_order"] == ["choice", "choice", "normal", "normal",
                              "uniform"], "tuple inputs RNG order")
    _must(r2["choice_calls"] == [[5, 3, True], [5, 3, True]],
          "choice once per input tensor")
    _must(r2["normal_shapes"] == [[3, 2], [3, 2]],
          "normal once per input tensor")
    _must(r2["uniform_calls"] == [[0.0, 1.0, 3]], "uniform only once")
    _must(r2["expected_indices"] == 6 and r2["expected_alphas"] == 3
          and r2["expected_noises"] == 12, "tuple sequence lengths")

    r3 = solve("schedule", n_samples=6, bsz=2, d=3, n_in=1, num_ref=4,
               compare_batch_size=2)
    d3 = r3["noise_tunnel_direct"]
    _must(d3["partitions"] == 3 and d3["remaining"] == 0, "direct NT partitions")
    _must(d3["choice_rounds"] == 1, "direct NT choice once (kwargs_copy)")
    _must(d3["normal_rounds"] == 3 and d3["uniform_rounds"] == 3,
          "direct NT normal/uniform per partition")
    r4 = solve("schedule", n_samples=7, bsz=1, d=2, n_in=1, num_ref=4,
               compare_batch_size=3)
    d4 = r4["noise_tunnel_direct"]
    _must(d4["partitions"] == 2 and d4["remaining"] == 1, "remainder partition")
    _must(d4["choice_rounds"] == 2 and d4["normal_rounds"] == 3
          and d4["uniform_rounds"] == 3, "remainder rounds")


# ================================================================ check 5
def check_gradient_shap(solve):
    """端到端数值：抽样→加噪→插值→梯度→乘子→smoothgrad 均值→delta。"""
    cases = [
        _case(1, 3, 2, "quadratic", True, 0.0, True, False, 4, 1),
        _case(1, 3, 3, "linear", True, 0.5, True, False, 4, 2),
        _case(2, 3, 2, "quadratic", False, 0.25, True, False, 4, 3),
        _case(2, 2, 4, "linear", False, 0.0, False, False, 5, 4),
        _case(1, 2, 5, "quadratic", True, 0.5, True, False, 3, 5),
        _case(2, 4, 3, "quadratic", True, 0.25, True, False, 6, 6),
        _case(1, 3, 2, "quadratic", True, 0.0, True, True, 4, 7),
        _case(2, 2, 3, "linear", True, 0.5, True, True, 4, 8),
        _case(3, 2, 2, "quadratic", True, 0.5, True, False, 4, 11),
    ]
    for ci, cfg in enumerate(cases):
        got = solve("gradient_shap", **cfg)
        exp = _exp_gs(cfg)
        d = (len(cfg["inputs"][0][0]) if cfg["tuple_inputs"]
             else len(cfg["inputs"][0]))
        _must(_eq(got["attributions"], exp["attributions"]),
              "case %d attributions" % ci)
        _must(got["partition_count"] == 1, "case %d single partition" % ci)
        _must(got["is_tuple"] == cfg["tuple_inputs"], "case %d is_tuple" % ci)
        _must(got["multiply_by_inputs"] == cfg["multiply_by_inputs"],
              "case %d multiply_by_inputs" % ci)
        _must(got["forward_kind"] == cfg["forward_kind"],
              "case %d forward_kind" % ci)
        _must(got["n_samples"] == cfg["n_samples"], "case %d n_samples" % ci)
        _must(got["bsz"] == (len(cfg["inputs"][0]) if cfg["tuple_inputs"]
                             else len(cfg["inputs"])), "case %d bsz" % ci)
        _must(got["feature_dim"] == d, "case %d feature_dim" % ci)
        _must(got["n_in"] == (2 if cfg["tuple_inputs"] else 1),
              "case %d n_in" % ci)
        if cfg["return_convergence_delta"]:
            _must("delta" in got and _eq(got["delta"], exp["delta"]),
                  "case %d delta" % ci)
        else:
            _must("delta" not in got, "case %d no delta when flag False" % ci)
        # tuple 场景：逐 tensor 核对各自 noise / indices 切片与 C-order 布局
        if cfg["tuple_inputs"]:
            ins = cfg["inputs"]
            bls = cfg["baselines"]
            kk = cfg["n_samples"]
            dd = len(ins[0][0])
            bb = len(ins[0])
            noi = cfg["noises"]
            idc = cfg["indices"]
            for ti in range(len(ins)):
                pt = got["per_tensor"][ti]
                nt = noi[ti * kk * bb * dd:(ti + 1) * kk * bb * dd]
                it = idc[ti * kk * bb:(ti + 1) * kk * bb]
                exp_pert = []
                for i in range(bb):
                    for j in range(kk):
                        r = i * kk + j
                        exp_pert.append([ins[ti][i][t] + nt[r * dd + t]
                                        for t in range(dd)])
                _must(_eq(pt["perturbed"], exp_pert),
                      "case %d tensor %d perturbed uses its OWN noise slice "
                      "and (i*k+j) C-order layout" % (ci, ti))
                _must(_eq(pt["drawn"], [list(bls[ti][i]) for i in it]),
                      "case %d tensor %d drawn uses its OWN indices slice"
                      % (ci, ti))
        # 中间量可对拍
        for pt in got["per_tensor"]:
            _must(len(pt["perturbed"]) == cfg["n_samples"]
                  * (len(cfg["inputs"][0]) if cfg["tuple_inputs"]
                     else len(cfg["inputs"])),
                  "case %d per_tensor perturbed rows" % ci)
            if cfg["multiply_by_inputs"]:
                _must(not _eq(pt["grads"], pt["attr"]),
                      "case %d attr != grads when mul=True" % ci)
            else:
                _must(_eq(pt["grads"], pt["attr"]),
                      "case %d attr == grads when mul=False" % ci)


# ================================================================ check 6
def check_convergence_delta(solve):
    """delta = sum(attrs) - (F(end)-F(start))，形状 (bsz*k,)，跨 tensor 累加。"""
    cfg = _case(2, 3, 3, "quadratic", True, 0.25, True, False, 5, 3)
    got = solve("gradient_shap", **cfg)
    k = cfg["n_samples"]
    bsz = len(cfg["inputs"])
    _must(len(got["delta"]) == k * bsz, "delta shape (bsz*k,)")
    exp = _exp_gs(cfg)
    _must(_eq(got["delta"], exp["delta"]), "delta values")

    # 手工核对 delta 公式：sum(attr_j) - (F(pert_j) - F(drawn_j))
    pt = got["per_tensor"][0]
    w = cfg["w"]
    for r in range(k * bsz):
        attr_sum = math.fsum(pt["attr"][r])
        f_end = math.fsum(w[t] * pt["perturbed"][r][t] ** 2
                          for t in range(len(w)))
        f_start = math.fsum(w[t] * pt["drawn"][r][t] ** 2
                            for t in range(len(w)))
        _must(_eq(got["delta"][r], attr_sum - (f_end - f_start)),
              "delta[%d] = sum(attr) - (F(end)-F(start))" % r)

    # tuple inputs：delta 跨 tensor 累加，只减一次 F 差
    cfg2 = _case(1, 3, 2, "linear", True, 0.0, True, True, 4, 9)
    got2 = solve("gradient_shap", **cfg2)
    exp2 = _exp_gs(cfg2)
    _must(_eq(got2["delta"], exp2["delta"]), "tuple inputs delta")
    _must(len(got2["delta"]) == 2, "tuple delta length")
    _must(len(got2["per_tensor"]) == 2, "two tensors recorded")

    # return_convergence_delta=False 时不返回 delta
    cfg3 = dict(cfg, return_convergence_delta=False)
    _must("delta" not in solve("gradient_shap", **cfg3),
          "delta omitted when flag False")


# ================================================================ check 7
def check_domain(solve):
    """各 mode 的域约束与错误消息。"""
    base = _case()
    _raises(lambda: solve("gradient_shap", **dict(base, stdevs=1)),
            "stdevs must be type float")
    _raises(lambda: solve("gradient_shap", **dict(base, forward_kind="cubic")),
            "unknown forward kind")
    _raises(lambda: solve("gradient_shap", **dict(base, w=[1.0])),
            "w must be")
    _raises(lambda: solve("gradient_shap", **dict(base, indices=[0])),
            "indices length")
    _raises(lambda: solve("gradient_shap", **dict(base, alphas=[0.5])),
            "alphas length")
    _raises(lambda: solve("gradient_shap", **dict(base, noises=[0.0])),
            "noises length")
    _raises(lambda: solve("gradient_shap", **dict(base, alphas=[1.0, 0.5])),
            "alphas must be in")
    _raises(lambda: solve("gradient_shap", **dict(base, multiply_by_inputs=1)),
            "multiply_by_inputs must be bool")
    cfg_t = _case(tup=True)
    _raises(lambda: solve("gradient_shap",
                          **dict(cfg_t, baselines=[cfg_t["baselines"][0]])),
            "tuple lengths must match")
    _raises(lambda: solve("baseline_draw", baselines=[[1., 2.]],
                          indices=[0, 0], n_samples=0, bsz=1),
            "n_samples must be positive")
    _raises(lambda: solve("noise_layout", inputs=[[1., 2.]], k=1,
                          noises=[0.0, 0.0], stdevs=0.5,
                          n_in=0), "n_in must be positive")
    _raises(lambda: solve("schedule", n_samples=0, bsz=1, d=2),
            "n_samples must be positive")
    cfg_sh = _case(bsz=2, d=3, k=2, tup=True)
    bad_sh = dict(cfg_sh, inputs=[cfg_sh["inputs"][0],
                                  [cfg_sh["inputs"][1][0]]])
    _raises(lambda: solve("gradient_shap", **bad_sh),
            "all input tensors must share bsz and feature dim")
    bad_bf = dict(base, baselines=[[0., 0.], [1., 1.],
                                   [2., 2.], [3., 3.]])
    _raises(lambda: solve("gradient_shap", **bad_bf),
            "baseline feature dim must match input feature dim (distrib mode)")
    bad_ir = dict(base, indices=[99] * len(base["indices"]))
    _raises(lambda: solve("gradient_shap", **bad_ir),
            "baseline index out of range")


# ================================================================ check 8
def check_unknown_mode(solve):
    """未知 mode 必须抛 ValueError 并回显 mode 名。"""
    _raises(lambda: solve("no_such_mode"), "unknown mode", "no_such_mode")
    _raises(lambda: solve("GRADIENT_SHAP"), "unknown mode", "GRADIENT_SHAP")
