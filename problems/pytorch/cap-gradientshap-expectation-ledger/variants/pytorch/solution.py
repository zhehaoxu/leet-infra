"""853 = Captum GradientShap 期望梯度账（参考解）

纯 Python 复刻官方 pytorch/captum main 的 GradientShap 机制，所有随机性由
显式序列（indices / alphas / noises）给出，不依赖 captum / torch / numpy，
也不依赖任何全局 RNG，便于数值判题。

官方回源：
  captum/attr/_core/gradient_shap.py::GradientShap.attribute
  captum/attr/_core/gradient_shap.py::InputBaselineXGradient.attribute
  captum/attr/_core/gradient_shap.py::_scale_input
  captum/attr/_core/noise_tunnel.py::NoiseTunnel.attribute
  captum/_utils/common.py::_expand_and_update_baselines
  captum/attr/_utils/attribution.py::GradientAttribution.compute_convergence_delta

5 个 mode：
  baseline_draw  基线分布抽样（np.random.choice 有放回 / 扁平 k*bsz 索引）
  scale_input    alpha 广播插值（(n,1,1,..) view / requires_grad_）
  noise_layout   加噪布局（repeat_interleave(k) + normal / stdevs 校验）
  schedule       RNG 调度账（单 partition / numpy-torch 双流 / 各调用次数）
  gradient_shap  端到端（抽样→加噪→插值→梯度→乘子→smoothgrad 均值→delta）
"""
import math


# ------------------------------------------------------------------ 公共工具
def _must(cond, msg):
    if not cond:
        raise ValueError(msg)


def _isnum(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def _rows(x, name):
    _must(isinstance(x, (list, tuple)) and len(x) > 0,
          "%s must be a non-empty list of rows" % name)
    _must(all(isinstance(r, (list, tuple)) for r in x),
          "%s must be a list of rows" % name)
    d = len(x[0])
    _must(d > 0, "%s rows must be non-empty" % name)
    _must(all(len(r) == d for r in x), "%s rows must have equal length" % name)
    _must(all(_isnum(v) for r in x for v in r),
          "%s must contain finite numbers" % name)
    return [list(r) for r in x], len(x), d


def _stdevs_check(stdevs, n_in):
    """官方 add_noise_to_inputs 的断言：float 标量，或长度等于 inputs 数的 tuple。"""
    if isinstance(stdevs, (tuple, list)):
        _must(len(stdevs) == n_in,
              "The number of input tensors must be equal to the number of stdevs values")
        _must(all(isinstance(s, float) for s in stdevs),
              "each stdev in a tuple must be type float")
    else:
        _must(isinstance(stdevs, float),
              "stdevs must be type float. Given: %s" % type(stdevs).__name__)


def _forward(forward_kind, wv, row):
    """单个 tensor 的 forward_func：对一行特征求和（linear / quadratic）。"""
    if forward_kind == "linear":
        return math.fsum(wv[t] * row[t] for t in range(len(row)))
    if forward_kind == "quadratic":
        return math.fsum(wv[t] * row[t] * row[t] for t in range(len(row)))
    raise ValueError("unknown forward kind: %r" % (forward_kind,))


def _grad(forward_kind, wv, scaled_row):
    """单个 tensor 的解析梯度。"""
    if forward_kind == "linear":
        return [float(wv[t]) for t in range(len(scaled_row))]
    if forward_kind == "quadratic":
        return [2.0 * wv[t] * scaled_row[t] for t in range(len(scaled_row))]
    raise ValueError("unknown forward kind: %r" % (forward_kind,))


# ------------------------------------------------------------ mode 1
def baseline_draw(baselines, indices, n_samples, bsz,
                  draw_baseline_from_distrib=True,
                  input_feature_dim=None, baseline_feature_dim=None):
    """官方 _expand_and_update_baselines 的 draw_baseline_from_distrib 分支。

    distrib=True : np.random.choice(num_ref, k*bsz, replace=True) -> baseline[indices]
                   索引是扁平的 k*bsz 个，不是按 example 分组。
    distrib=False: baseline.repeat_interleave(k, dim=0)
                   （仅当 baseline.shape[0] == input.shape[0] 且 > 1 时生效）
    """
    rows, num_ref, d = _rows(baselines, "baselines")
    k = int(n_samples)
    _must(k > 0, "n_samples must be positive")
    _must(bsz > 0, "bsz must be positive")
    expected = k * bsz
    _must(isinstance(indices, (list, tuple)) and len(indices) == expected,
          "indices length must be k*bsz (%d)" % expected)
    _must(all(isinstance(i, int) and not isinstance(i, bool) for i in indices),
          "indices must be ints")
    _must(all(0 <= i < num_ref for i in indices),
          "baseline index out of range [0, %d)" % num_ref)
    if input_feature_dim is not None:
        bf = d if baseline_feature_dim is None else baseline_feature_dim
        _must(bf == input_feature_dim,
              "The samples in input and baseline batches must have the same shape "
              "or the baseline corresponding to the input tensor must be a scalar")

    if draw_baseline_from_distrib:
        drawn = [list(rows[i]) for i in indices]
        layout = "flat_k_times_bsz"
        non_distrib = None
        drawn_shape = [expected, d]
        validate = "shape[1:] only (distrib)"
    else:
        applicable = (num_ref == bsz and num_ref > 1)
        if applicable:
            drawn = [list(rows[j]) for j in range(num_ref) for _ in range(k)]
            layout = "repeat_interleave"
            drawn_shape = [num_ref * k, d]
        else:
            drawn = [list(r) for r in rows]
            layout = "unchanged"
            drawn_shape = [num_ref, d]
        non_distrib = {"applicable": applicable, "layout": layout}
        validate = "full shape (non-distrib)"

    return {
        "num_ref": num_ref, "feature_dim": d, "k": k, "bsz": bsz,
        "expected_indices": expected,
        "indices": list(indices),
        "drawn": drawn, "drawn_shape": drawn_shape,
        "layout": layout, "replace": True,
        "rng": "np.random.choice(num_ref, k*bsz)",
        "validate": validate,
        "non_distrib": non_distrib,
    }


# ------------------------------------------------------------ mode 2
def scale_input(inputs, baselines, alphas, ndim=2):
    """官方 _scale_input：alpha.view((n,) + (1,)*(ndim-1)) 广播插值。"""
    x, n, d = _rows(inputs, "inputs")
    b, nb, db = _rows(baselines, "baselines")
    _must(nb == n, "baselines rows must match inputs rows")
    _must(db == d, "baselines feature dim must match inputs feature dim")
    _must(isinstance(alphas, (list, tuple)) and len(alphas) == n,
          "alphas length must equal batch size (%d)" % n)
    _must(all(_isnum(a) for a in alphas), "alphas must be finite numbers")
    _must(all(0.0 <= float(a) < 1.0 for a in alphas),
          "alpha must come from np.random.uniform(0.0, 1.0), i.e. 0 <= a < 1")
    _must(isinstance(ndim, int) and ndim >= 2, "ndim must be an int >= 2")

    coeff_shape = [n] + [1] * (ndim - 1)
    scaled = [
        [float(alphas[i]) * x[i][t] + (1.0 - float(alphas[i])) * b[i][t]
         for t in range(d)]
        for i in range(n)
    ]
    return {
        "batch": n, "feature_dim": d, "ndim": ndim,
        "coeff_shape": coeff_shape,
        "scaled": scaled,
        "interpolation": "alpha*input + (1-alpha)*baseline",
        "rng": "np.random.uniform(0.0, 1.0, batch)",
        "requires_grad": True,
    }


# ------------------------------------------------------------ mode 3
def noise_layout(inputs, k, noises, stdevs=0.0, n_in=1):
    """官方 add_noise_to_input：input.repeat_interleave(k, dim=0) + noise。

    noise = torch.normal(0, torch.tensor(stdev).repeat((bsz*k,) + feature_shape))
    stdevs 必须是 float 或长度等于 inputs 数的 tuple（int 会被 assert 拒绝）。
    """
    x, bsz, d = _rows(inputs, "inputs")
    k = int(k)
    _must(k > 0, "k must be positive")
    _must(isinstance(n_in, int) and n_in > 0, "n_in must be positive")
    _stdevs_check(stdevs, n_in)
    expected = bsz * k * d * n_in
    _must(isinstance(noises, (list, tuple)) and len(noises) == expected,
          "noises length must be bsz*k*d*n_in (%d)" % expected)
    _must(all(_isnum(v) for v in noises), "noises must be finite numbers")

    perturbed = []
    for i in range(bsz):
        for j in range(k):
            base = (i * k + j) * d
            perturbed.append([float(x[i][t]) + float(noises[base + t])
                              for t in range(d)])
    return {
        "bsz": bsz, "feature_dim": d, "k": k, "n_in": n_in,
        "perturbed": perturbed,
        "perturbed_shape": [bsz * k, d],
        "normal_shape": [bsz * k, d],
        "layout": "repeat_interleave(k) + normal",
        "stdevs": (list(stdevs) if isinstance(stdevs, (tuple, list))
                   else float(stdevs)),
        "stdevs_repeat": "torch.tensor(stdev).repeat((bsz*k,) + feature_shape)",
        "requires_grad": True,
        "expected_noises": expected,
    }


# ------------------------------------------------------------ mode 4
def schedule(n_samples, bsz, d, n_in=1, num_ref=4, compare_batch_size=None):
    """官方 GradientShap 的 RNG 调度账。

    GradientShap.attribute **不透出 nt_samples_batch_size**，所以
    NoiseTunnel 内 nt_samples_batch_size=None -> 单 partition。
    choice / uniform 走 numpy 全局流（共享）；normal 走独立的 torch 流。
    """
    k = int(n_samples)
    _must(k > 0, "n_samples must be positive")
    _must(bsz > 0 and d > 0 and n_in > 0,
          "bsz, d, n_in must be positive")
    _must(num_ref > 0, "num_ref must be positive")

    effective = k                      # GradientShap 恒为单 partition
    partitions = 1
    remaining = 0
    choice_calls = [[num_ref, effective * bsz, True] for _ in range(n_in)]
    normal_shapes = [[effective * bsz, d] for _ in range(n_in)]
    uniform_calls = [[0.0, 1.0, effective * bsz]]
    order = ["choice"] * n_in + ["normal"] * n_in + ["uniform"]

    cmp = None
    if compare_batch_size is not None:
        cbs = int(compare_batch_size)
        _must(0 < cbs <= k, "compare_batch_size must be in (0, n_samples]")
        b = min(k, cbs)
        p = k // b
        rem = k - p * b
        cmp = {
            "batch_size": b, "partitions": p, "remaining": rem,
            "choice_rounds": 1 + (1 if rem > 0 else 0),
            "normal_rounds": p + (1 if rem > 0 else 0),
            "uniform_rounds": p + (1 if rem > 0 else 0),
        }

    return {
        "n_samples": k, "bsz": bsz, "feature_dim": d, "n_in": n_in,
        "nt_samples_batch_size": effective,
        "batch_size_forwarded": False,
        "partitions": partitions, "remaining": remaining,
        "choice_calls": choice_calls,
        "normal_shapes": normal_shapes,
        "uniform_calls": uniform_calls,
        "rng_order": order,
        "np_stream_shared": True,
        "torch_stream_independent": True,
        "expected_indices": effective * bsz * n_in,
        "expected_alphas": effective * bsz,
        "expected_noises": effective * bsz * d * n_in,
        "noise_tunnel_direct": cmp,
    }


# ------------------------------------------------------------ mode 5
def gradient_shap(inputs, baselines, w, forward_kind="quadratic",
                  n_samples=5, stdevs=0.0, multiply_by_inputs=True,
                  return_convergence_delta=False, tuple_inputs=False,
                  indices=None, alphas=None, noises=None):
    """端到端复刻官方 GradientShap.attribute（单 partition）。

    流程：
      baselines 分布抽样 (np.random.choice, 每 tensor 独立)
        -> 加噪 (repeat_interleave(k) + torch.normal)
        -> alpha 插值 (np.random.uniform, 所有 tensor 共享一次)
        -> 解析梯度 (linear: w / quadratic: 2*w*scaled)
        -> multiply_by_inputs: (perturbed - drawn) * grad 或 grad
        -> smoothgrad 均值 (view(bsz,k,..).sum(dim=1) * 1/nt_samples)
        -> 可选 delta = sum(attrs) - (F(end) - F(start))，形状 (bsz*k,)
    """
    _must(forward_kind in ("linear", "quadratic"),
          "unknown forward kind: %r" % (forward_kind,))
    _must(isinstance(multiply_by_inputs, bool),
          "multiply_by_inputs must be bool")
    _must(isinstance(return_convergence_delta, bool),
          "return_convergence_delta must be bool")
    _must(isinstance(tuple_inputs, bool), "tuple_inputs must be bool")
    k = int(n_samples)
    _must(k > 0, "n_samples must be positive")

    if tuple_inputs:
        _must(isinstance(inputs, (list, tuple)) and len(inputs) > 0,
              "tuple inputs must be a non-empty list of tensors")
        tensors = [_rows(t, "inputs[%d]" % i)[0] for i, t in enumerate(inputs)]
        base_tensors = [_rows(t, "baselines[%d]" % i)[0]
                        for i, t in enumerate(baselines)]
        _must(len(tensors) == len(base_tensors),
              "inputs and baselines tuple lengths must match")
    else:
        tensors = [_rows(inputs, "inputs")[0]]
        base_tensors = [_rows(baselines, "baselines")[0]]
    n_in = len(tensors)
    bsz = len(tensors[0])
    d = len(tensors[0][0])
    for t in tensors:
        _must(len(t) == bsz and len(t[0]) == d,
              "all input tensors must share bsz and feature dim")
    for t in base_tensors:
        _must(len(t[0]) == d,
              "baseline feature dim must match input feature dim (distrib mode)")
    wv = list(w) if isinstance(w, (list, tuple)) else [w]
    _must(len(wv) >= d and all(_isnum(v) for v in wv[:d]),
          "w must be a numeric list covering feature dim")
    _stdevs_check(stdevs, n_in)

    idx = list(indices or [])
    alp = list(alphas or [])
    noi = list(noises or [])
    _must(len(idx) == k * bsz * n_in,
          "indices length must be k*bsz*n_in (%d)" % (k * bsz * n_in))
    _must(len(alp) == k * bsz,
          "alphas length must be k*bsz (%d)" % (k * bsz))
    _must(len(noi) == k * bsz * d * n_in,
          "noises length must be k*bsz*d*n_in (%d)" % (k * bsz * d * n_in))
    _must(all(isinstance(i, int) and not isinstance(i, bool) for i in idx),
          "indices must be ints")
    _must(all(_isnum(a) and 0.0 <= float(a) < 1.0 for a in alp),
          "alphas must be in [0,1)")
    _must(all(_isnum(v) for v in noi), "noises must be finite numbers")

    per_tensor = []
    all_perturbed = []
    all_drawn = []
    all_attr = []
    for ti in range(n_in):
        x = tensors[ti]
        bl = base_tensors[ti]
        num_ref = len(bl)
        idx_t = idx[ti * k * bsz:(ti + 1) * k * bsz]
        noi_t = noi[ti * k * bsz * d:(ti + 1) * k * bsz * d]
        _must(all(0 <= i < num_ref for i in idx_t),
              "baseline index out of range [0, %d)" % num_ref)

        drawn = [list(bl[i]) for i in idx_t]
        perturbed = []
        for i in range(bsz):
            for j in range(k):
                base = (i * k + j) * d
                perturbed.append([float(x[i][t]) + float(noi_t[base + t])
                                  for t in range(d)])
        scaled = [
            [float(alp[r]) * perturbed[r][t]
             + (1.0 - float(alp[r])) * drawn[r][t] for t in range(d)]
            for r in range(k * bsz)
        ]
        grads = [_grad(forward_kind, wv, scaled[r]) for r in range(k * bsz)]
        if multiply_by_inputs:
            attr = [
                [(perturbed[r][t] - drawn[r][t]) * grads[r][t]
                 for t in range(d)]
                for r in range(k * bsz)
            ]
        else:
            attr = [list(g) for g in grads]

        # smoothgrad：单 partition，按 (bsz,k) 分组求均值后 * 1/nt_samples
        out = [
            [math.fsum(attr[i * k + j][t] for j in range(k)) / k
             for t in range(d)]
            for i in range(bsz)
        ]
        per_tensor.append({
            "perturbed": perturbed, "drawn": drawn, "scaled": scaled,
            "grads": grads, "attr": attr, "attributions": out,
        })
        all_perturbed.append(perturbed)
        all_drawn.append(drawn)
        all_attr.append(attr)

    result = {
        "attributions": ([t["attributions"] for t in per_tensor]
                         if tuple_inputs else per_tensor[0]["attributions"]),
        "per_tensor": per_tensor,
        "is_tuple": tuple_inputs,
        "multiply_by_inputs": multiply_by_inputs,
        "forward_kind": forward_kind,
        "partition_count": 1,
        "n_samples": k,
        "bsz": bsz,
        "feature_dim": d,
        "n_in": n_in,
    }
    if return_convergence_delta:
        # delta per noisy sample (bsz*k,)：跨 tensor 累加 attr 行和，
        # 减一次 (F(end) - F(start))（F 接收全部 tensor）
        delta = []
        for r in range(k * bsz):
            attr_sum = math.fsum(
                math.fsum(all_attr[ti][r][t] for t in range(d))
                for ti in range(n_in)
            )
            f_end = math.fsum(
                _forward(forward_kind, wv, all_perturbed[ti][r])
                for ti in range(n_in)
            )
            f_start = math.fsum(
                _forward(forward_kind, wv, all_drawn[ti][r])
                for ti in range(n_in)
            )
            delta.append(attr_sum - (f_end - f_start))
        result["delta"] = delta
    return result


# ------------------------------------------------------------ 分派
def solve(mode, **kw):
    if mode == "baseline_draw":
        return baseline_draw(**kw)
    if mode == "scale_input":
        return scale_input(**kw)
    if mode == "noise_layout":
        return noise_layout(**kw)
    if mode == "schedule":
        return schedule(**kw)
    if mode == "gradient_shap":
        return gradient_shap(**kw)
    raise ValueError("unknown mode: %r" % (mode,))
