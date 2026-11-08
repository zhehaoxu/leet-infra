"""854 reference solution: Captum Lime interpretable-sampling ledger.

Pure Python (no torch / sklearn).  All randomness and upstream model outputs
are supplied explicitly.  Five modes share one deterministic accounting of
feature masks, binary perturbation, exponential kernel, weighted-ridge
surrogate and coefficient backfill.
"""
import math
from fractions import Fraction

_MODES = ("feature_mask", "interp_transform", "kernel", "surrogate", "lime")
_DISTANCES = ("cosine", "euclidean")
_FORWARDS = ("linear", "quadratic")
_EPS = 1e-8


def _must(cond, msg):
    if not cond:
        raise ValueError(msg)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _q(v):
    return Fraction(str(v))


def _rows(x, name, integers=False):
    """Normalize to a list of tensors, each a 2D list of rows."""
    _must(isinstance(x, (list, tuple)) and len(x) > 0,
          name + " must be a non-empty list")
    first = x[0]
    if isinstance(first, (list, tuple)) and len(first) > 0 and _is_num(first[0]):
        # single 2D tensor: [[...], [...]]
        tensors = [list(x)]
    else:
        # tuple of 2D tensors: [[[...]], [[...]]]
        tensors = list(x)
    out = []
    for ti, t in enumerate(tensors):
        _must(isinstance(t, (list, tuple)) and len(t) > 0,
              name + " tensor %d must be a non-empty 2D list" % ti)
        rows = []
        for row in t:
            _must(isinstance(row, (list, tuple)) and len(row) > 0,
                  name + " tensor %d rows must be non-empty" % ti)
            for v in row:
                if integers:
                    _must(type(v) is int, name + " indices must be int")
                else:
                    _must(_is_num(v), name + " values must be finite numbers")
                    _must(math.isfinite(float(v)), name + " values must be finite numbers")
            rows.append(list(row))
        out.append(rows)
    return out


def _one_row(t, name):
    _must(len(t) == 1, name + " expects a single example (bsz=1)")
    return list(t[0])


def _flat_any(x, name):
    """Flatten a 1D numeric list or a list of 2D tensors into one vector."""
    _must(isinstance(x, (list, tuple)) and len(x) > 0, name + " must be non-empty")
    if all(_is_num(v) for v in x):
        return [float(v) for v in x]
    out = []
    for t in _rows(x, name):
        out.extend(float(v) for v in _one_row(t, name))
    return out


def _solve_fraction(A, b):
    n = len(b)
    M = [[_q(A[i][j]) for j in range(n)] + [_q(b[i])] for i in range(n)]
    for c in range(n):
        p = None
        for r in range(c, n):
            if M[r][c] != 0:
                p = r
                break
        _must(p is not None, "surrogate Gram matrix is singular")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            if M[r][c] == 0:
                continue
            f = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    x = [Fraction(0)] * n
    for r in range(n - 1, -1, -1):
        acc = M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))
        x[r] = acc / M[r][r]
    return x


def feature_mask(inputs, feature_mask=None, tuple_inputs=False, **kw):
    ts = _rows(inputs, "inputs")
    if tuple_inputs:
        _must(len(ts) > 1, "tuple_inputs requires at least two input tensors")
    else:
        _must(len(ts) == 1, "single input requires tuple_inputs=False")
    for t in ts:
        _one_row(t, "input")

    if feature_mask is None:
        mask = []
        idx = 0
        for t in ts:
            d = len(t[0])
            mask.append([list(range(idx, idx + d))])
            idx += d
        return {
            "mask": mask if tuple_inputs else mask[0],
            "num_interp_features": idx,
            "shifted": False,
            "default": True,
            "min_index": 0,
            "shapes": [len(t[0]) for t in ts],
        }

    ms = _rows(feature_mask, "feature_mask", integers=True)
    _must(len(ms) == len(ts), "feature_mask tensor count must match inputs")
    for ti, m in enumerate(ms):
        mrow = _one_row(m, "feature_mask")
        _must(len(mrow) == len(ts[ti][0]),
              "feature_mask row length must match input row length")
        _must(all(v >= 0 for v in mrow), "feature_mask indices must be non-negative")
    flat = [v for m in ms for row in m for v in row]
    min_index = min(flat)
    shifted = min_index != 0
    if shifted:
        ms = [[[v - min_index for v in row] for row in m] for m in ms]
    F = max(v for m in ms for row in m for v in row) + 1
    return {
        "mask": ms if tuple_inputs else ms[0],
        "num_interp_features": F,
        "shifted": shifted,
        "default": False,
        "min_index": min_index,
        "shapes": [len(t[0]) for t in ts],
    }


def _mask_tensors(fm, tup):
    if tup:
        return fm["mask"]
    return [fm["mask"]]


def interp_transform(sample, inputs, baselines, feature_mask,
                     tuple_inputs=False, **kw):
    ts = _rows(inputs, "inputs")
    bs = _rows(baselines, "baselines")
    fm = feature_mask_info(inputs, feature_mask, tuple_inputs)
    mt = _mask_tensors(fm, tuple_inputs)
    F = fm["num_interp_features"]
    _must(isinstance(sample, (list, tuple)) and len(sample) == F,
          "sample length must equal num_interp_features")
    _must(all(type(v) is int and v in (0, 1) for v in sample),
          "sample entries must be int 0 or 1")
    _must(len(bs) == len(ts), "baselines tensor count must match inputs")
    _must(len(mt) == len(ts), "feature_mask tensor count must match inputs")
    perturbed = []
    binary = []
    for ti, t in enumerate(ts):
        row = _one_row(t, "input")
        brow = _one_row(bs[ti], "baseline")
        mrow = _one_row(mt[ti], "feature_mask")
        _must(len(row) == len(brow), "baselines feature dim must match inputs")
        b = [int(sample[m]) for m in mrow]
        p = [float(row[j]) if b[j] else float(brow[j]) for j in range(len(row))]
        binary.append(b)
        perturbed.append([p])
    return {
        "perturbed": perturbed if tuple_inputs else perturbed[0],
        "binary_mask": binary if tuple_inputs else binary[0],
        "num_interp_features": F,
    }


def feature_mask_info(inputs, feature_mask, tuple_inputs):
    """Internal: normalized feature-mask record (shared by modes)."""
    return feature_mask_fn(inputs, feature_mask, tuple_inputs)


# alias kept as a separate name so mutation anchors stay unambiguous
feature_mask_fn = feature_mask


def kernel(distance_mode, kernel_width, original, perturbed, **kw):
    _must(distance_mode in _DISTANCES,
          "distance_mode must be either cosine or euclidean.")
    _must(_is_num(kernel_width) and float(kernel_width) > 0.0,
          "kernel_width must be positive")
    a = _flat_any(original, "original")
    b = _flat_any(perturbed, "perturbed")
    _must(len(a) == len(b), "original and perturbed must have the same length")
    if distance_mode == "cosine":
        dot = math.fsum(float(a[i]) * float(b[i]) for i in range(len(a)))
        na = math.sqrt(math.fsum(float(v) ** 2 for v in a))
        nb = math.sqrt(math.fsum(float(v) ** 2 for v in b))
        denom = max(na * nb, _EPS)
        distance = 1.0 - dot / denom
    else:
        distance = math.sqrt(math.fsum(
            (float(a[i]) - float(b[i])) ** 2 for i in range(len(a))))
    weight = math.exp(-1.0 * (distance ** 2) / (2.0 * (float(kernel_width) ** 2)))
    return {"distance": distance, "weight": weight,
            "distance_mode": distance_mode, "kernel_width": float(kernel_width)}


def surrogate(interp_samples, outputs, weights, alpha=0.0, **kw):
    _must(isinstance(interp_samples, (list, tuple)) and len(interp_samples) > 0,
          "interp_samples must be a non-empty list")
    n = len(interp_samples)
    F = len(interp_samples[0])
    _must(F > 0, "interp_samples must have at least one feature")
    _must(len(outputs) == n, "outputs length must match interp_samples")
    _must(len(weights) == n, "weights length must match interp_samples")
    _must(_is_num(alpha) and float(alpha) >= 0.0, "alpha must be non-negative")
    X, Y, W = [], [], []
    for i, row in enumerate(interp_samples):
        _must(isinstance(row, (list, tuple)) and len(row) == F,
              "interp_samples rows must all have F features")
        _must(all(type(v) is int and v in (0, 1) for v in row),
              "interp_samples entries must be int 0 or 1")
        X.append([float(v) for v in row])
        _must(_is_num(outputs[i]), "outputs must be finite numbers")
        _must(_is_num(weights[i]) and float(weights[i]) >= 0.0,
              "weights must be non-negative finite numbers")
        Y.append(float(outputs[i]))
        W.append(float(weights[i]))
    gram = [[math.fsum(W[i] * X[i][a] * X[i][b] for i in range(n))
             for b in range(F)] for a in range(F)]
    for a in range(F):
        gram[a][a] += float(alpha)
    rhs = [math.fsum(W[i] * X[i][a] * Y[i] for i in range(n)) for a in range(F)]
    coefs = _solve_fraction(gram, rhs)
    return {"coefs": [float(c) for c in coefs], "gram": gram, "rhs": rhs,
            "n_samples": n, "num_interp_features": F, "alpha": float(alpha)}


def _forward_value(kind, wv, tensors, tup):
    _must(kind in _FORWARDS, "forward_kind must be linear or quadratic")
    _must(isinstance(wv, (list, tuple)) and len(wv) > 0,
          "forward_weights must be a non-empty list")
    total = 0.0
    for t in tensors:
        row = _one_row(t, "input")
        w = [float(v) for v in wv[:len(row)]]
        _must(len(w) == len(row), "forward_weights must cover every input feature")
        if kind == "linear":
            total += math.fsum(w[j] * float(row[j]) for j in range(len(row)))
        else:
            total += math.fsum(w[j] * float(row[j]) ** 2 for j in range(len(row)))
    return total


def _shape_tensors(x, tup):
    if tup:
        return x
    return [x]


def lime(inputs, baselines, samples, n_samples, forward_kind="linear",
         forward_weights=None, feature_mask=None, tuple_inputs=False,
         distance_mode="cosine", kernel_width=1.0, weights=None,
         perturbations_per_eval=1, return_input_shape=True, alpha=0.0, **kw):
    _must(type(n_samples) is int and n_samples > 0, "n_samples must be a positive int")
    _must(type(perturbations_per_eval) is int and perturbations_per_eval > 0,
          "perturbations_per_eval must be a positive int")
    _must(isinstance(return_input_shape, bool), "return_input_shape must be bool")
    fm = feature_mask_fn(inputs, feature_mask, tuple_inputs)
    F = fm["num_interp_features"]
    mt = _mask_tensors(fm, tuple_inputs)
    ts = _rows(inputs, "inputs")
    _must(isinstance(samples, (list, tuple)) and len(samples) == n_samples,
          "samples length must equal n_samples")
    _must(all(isinstance(s, (list, tuple)) and len(s) == F for s in samples),
          "each sample must have num_interp_features entries")
    _must(all(all(type(v) is int and v in (0, 1) for v in s) for s in samples),
          "sample entries must be int 0 or 1")
    if weights is not None:
        _must(isinstance(weights, (list, tuple)) and len(weights) == n_samples,
              "explicit weights length must equal n_samples")
    if distance_mode not in _DISTANCES:
        raise ValueError("distance_mode must be either cosine or euclidean.")
    _must(_is_num(kernel_width) and float(kernel_width) > 0.0,
          "kernel_width must be positive")
    _must(forward_weights is not None, "forward_weights must be provided")

    interp = [list(s) for s in samples]
    pert_all = []
    bin_all = []
    outputs = []
    wts = []
    orig_flat = _flat_any(ts, "inputs")
    for i, s in enumerate(interp):
        tr = interp_transform(s, inputs, baselines, feature_mask, tuple_inputs)
        pt = _shape_tensors(tr["perturbed"], tuple_inputs)
        bt = _shape_tensors(tr["binary_mask"], tuple_inputs)
        pert_all.append(pt)
        bin_all.append(bt)
        outputs.append(_forward_value(forward_kind, forward_weights, pt, tuple_inputs))
        if weights is None:
            kv = kernel(distance_mode, kernel_width, orig_flat,
                        _flat_any(pt, "perturbed"))
            wts.append(kv["weight"])
        else:
            _must(_is_num(weights[i]) and float(weights[i]) >= 0.0,
                  "explicit weights must be non-negative finite numbers")
            wts.append(float(weights[i]))

    batches = [perturbations_per_eval] * (n_samples // perturbations_per_eval)
    if n_samples % perturbations_per_eval:
        batches.append(n_samples % perturbations_per_eval)
    sg = surrogate(interp, outputs, wts, alpha)
    coefs = sg["coefs"]

    if return_input_shape:
        attr = []
        for ti, t in enumerate(ts):
            row = _one_row(t, "input")
            mrow = _one_row(mt[ti], "feature_mask")
            attr.append([[math.fsum(coefs[f] for f in range(F) if mrow[j] == f)
                          for j in range(len(row))]])
        attribution = attr if tuple_inputs else attr[0]
    else:
        attribution = list(coefs)

    return {
        "coefs": coefs,
        "attribution": attribution,
        "interp_samples": interp,
        "outputs": outputs,
        "weights": wts,
        "perturbed": [p if tuple_inputs else p[0] for p in pert_all],
        "binary_masks": [b if tuple_inputs else b[0] for b in bin_all],
        "batches": batches,
        "num_batches": len(batches),
        "batch_count": n_samples,
        "loader_batch_size": n_samples,
        "dataset_shapes": [[n_samples, F], [n_samples], [n_samples]],
        "num_interp_features": F,
        "mask": fm["mask"],
        "shifted": fm["shifted"],
        "default_mask": fm["default"],
        "gram": sg["gram"],
        "rhs": sg["rhs"],
        "return_input_shape": return_input_shape,
        "alpha": float(alpha),
        "distance_mode": distance_mode,
    }


def solve(mode, **kw):
    if mode == "feature_mask":
        return feature_mask(**kw)
    if mode == "interp_transform":
        return interp_transform(**kw)
    if mode == "kernel":
        return kernel(**kw)
    if mode == "surrogate":
        return surrogate(**kw)
    if mode == "lime":
        return lime(**kw)
    raise ValueError("unknown mode: %r" % (mode,))
