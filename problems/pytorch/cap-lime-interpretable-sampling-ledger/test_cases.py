"""854 judge test cases.  Every check receives `solve` from the judge.

Expected values are computed by an independent, self-contained implementation;
this file must not import from `solution`.
"""
import math
from fractions import Fraction

TOL = 1e-9
KERNEL_TOL = 1e-6
EPS = 1e-8


def _must(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _eq(a, b, tol=TOL):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b and type(a) is type(b)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return False
        return len(a) == len(b) and all(_eq(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def _check(got, exp, tag, tol=TOL):
    _must(_eq(got, exp, tol), "%s: got %r, expected %r" % (tag, got, exp))


def _raises(fn, keyword, tag):
    try:
        fn()
    except BaseException as e:
        _must(keyword in str(e),
              "%s: exception message %r missing keyword %r" % (tag, str(e), keyword))
        return
    raise AssertionError("%s: expected an exception" % tag)


def Q(v):
    return Fraction(str(v))


def _solve(A, b):
    n = len(b)
    M = [[Q(A[i][j]) for j in range(n)] + [Q(b[i])] for i in range(n)]
    for c in range(n):
        p = None
        for r in range(c, n):
            if M[r][c] != 0:
                p = r
                break
        if p is None:
            raise ValueError("surrogate Gram matrix is singular")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            if M[r][c] == 0:
                continue
            f = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    x = [Fraction(0)] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))) / M[r][r]
    return [float(v) for v in x]


def _kernel(mode, kw, a, b):
    if mode == "cosine":
        dot = math.fsum(a[i] * b[i] for i in range(len(a)))
        na = math.sqrt(math.fsum(v * v for v in a))
        nb = math.sqrt(math.fsum(v * v for v in b))
        d = 1.0 - dot / max(na * nb, EPS)
    else:
        d = math.sqrt(math.fsum((a[i] - b[i]) ** 2 for i in range(len(a))))
    return d, math.exp(-(d ** 2) / (2.0 * (kw ** 2)))


BASE = {
    "inputs": [[1.0, 2.0, 3.0, 4.0]],
    "baselines": [[0.0, 0.0, 0.0, 0.0]],
    "feature_mask": [[0, 0, 1, 1]],
    "samples": [[1, 0], [0, 1], [1, 1], [0, 0]],
    "n_samples": 4,
    "forward_kind": "linear",
    "forward_weights": [1.0, 2.0, 3.0, 4.0],
    "weights": [0.5, 0.25, 1.0, 0.75],
    "alpha": 0.0,
    "perturbations_per_eval": 1,
    "return_input_shape": True,
    "distance_mode": "cosine",
    "kernel_width": 1.0,
    "tuple_inputs": False,
}


def _case(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


def _exp_fm(inputs, feature_mask, tup):
    ins = inputs if tup else [inputs]
    if feature_mask is None:
        masks = []
        idx = 0
        for t in ins:
            d = len(t[0])
            masks.append([list(range(idx, idx + d))])
            idx += d
        return {"mask": masks if tup else masks[0], "num_interp_features": idx,
                "shifted": False, "default": True, "min_index": 0,
                "shapes": [len(t[0]) for t in ins]}
    ms = feature_mask if tup else [feature_mask]
    vals = [v for m in ms for row in m for v in row]
    mn = min(vals)
    shifted = mn != 0
    if shifted:
        ms = [[[v - mn for v in row] for row in m] for m in ms]
    F = max(v for m in ms for row in m for v in row) + 1
    return {"mask": ms if tup else ms[0], "num_interp_features": F,
            "shifted": shifted, "default": False, "min_index": mn,
            "shapes": [len(t[0]) for t in ins]}


def _exp_lime(cfg):
    tup = cfg.get("tuple_inputs", False)
    ins = cfg["inputs"] if tup else [cfg["inputs"]]
    bas = cfg["baselines"] if tup else [cfg["baselines"]]
    fm = _exp_fm(cfg["inputs"], cfg.get("feature_mask"), tup)
    masks = fm["mask"] if tup else [fm["mask"]]
    F = fm["num_interp_features"]
    orig = [float(v) for t in ins for v in t[0]]
    wv = [float(v) for v in cfg["forward_weights"]]
    kind = cfg.get("forward_kind", "linear")
    n = cfg["n_samples"]
    ppe = cfg.get("perturbations_per_eval", 1)
    alpha = float(cfg.get("alpha", 0.0))
    wexp = cfg.get("weights")
    pert, outs, wts, bins = [], [], [], []
    for i, s in enumerate(cfg["samples"]):
        rowp, rowb = [], []
        for ti, t in enumerate(ins):
            mrow = masks[ti][0]
            bm = [int(s[m]) for m in mrow]
            prow = [float(t[0][j]) if bm[j] else float(bas[ti][0][j])
                    for j in range(len(t[0]))]
            rowp.append([prow])
            rowb.append(bm)
        pert.append(rowp)
        bins.append(rowb)
        y = 0.0
        for ti, t in enumerate(ins):
            w = wv[:len(t[0])]
            if kind == "linear":
                y += math.fsum(w[j] * rowp[ti][0][j] for j in range(len(t[0])))
            else:
                y += math.fsum(w[j] * rowp[ti][0][j] ** 2 for j in range(len(t[0])))
        outs.append(y)
        if wexp is None:
            flatp = [v for p in rowp for v in p[0]]
            wts.append(_kernel(cfg.get("distance_mode", "cosine"),
                               float(cfg.get("kernel_width", 1.0)), orig, flatp)[1])
        else:
            wts.append(float(wexp[i]))
    X = [[float(v) for v in s] for s in cfg["samples"]]
    gram = [[math.fsum(wts[i] * X[i][a] * X[i][b] for i in range(n))
             for b in range(F)] for a in range(F)]
    for a in range(F):
        gram[a][a] += alpha
    rhs = [math.fsum(wts[i] * X[i][a] * outs[i] for i in range(n)) for a in range(F)]
    coefs = _solve(gram, rhs)
    batches = [ppe] * (n // ppe) + ([n % ppe] if n % ppe else [])
    if cfg.get("return_input_shape", True):
        attr = []
        for ti, t in enumerate(ins):
            attr.append([[math.fsum(coefs[f] for f in range(F)
                                   if masks[ti][0][j] == f)
                          for j in range(len(t[0]))]])
        attribution = attr if tup else attr[0]
    else:
        attribution = list(coefs)
    return {
        "coefs": coefs, "attribution": attribution,
        "interp_samples": [list(s) for s in cfg["samples"]],
        "outputs": outs, "weights": wts,
        "perturbed": [p if tup else p[0] for p in pert],
        "binary_masks": [b if tup else b[0] for b in bins],
        "batches": batches, "num_batches": len(batches),
        "batch_count": n, "loader_batch_size": n,
        "dataset_shapes": [[n, F], [n], [n]],
        "num_interp_features": F, "mask": fm["mask"],
        "shifted": fm["shifted"], "default_mask": fm["default"],
        "gram": gram, "rhs": rhs,
        "return_input_shape": cfg.get("return_input_shape", True),
        "alpha": alpha, "distance_mode": cfg.get("distance_mode", "cosine"),
    }


LIME_CASES = [
    ("L0", _case()),
    ("L1", _case(perturbations_per_eval=3)),
    ("L2", _case(alpha=0.5)),
    ("L3", _case(forward_kind="quadratic",
                 forward_weights=[1.0, 0.5, 2.0, 0.25], alpha=0.25)),
    ("L4", _case(baselines=[[-1.0, -1.0, -2.0, -2.0]], alpha=0.125)),
    ("L5", _case(feature_mask=None,
                 samples=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                 n_samples=6,
                 weights=[1.0, 0.5, 0.75, 0.25, 0.5, 0.25],
                 perturbations_per_eval=4)),
    ("L6", _case(feature_mask=[[5, 5, 6, 6]],
                 samples=[[1, 0], [0, 1], [1, 1], [0, 0]])),
    ("L7", _case(weights=None, kernel_width=2.0)),
    ("L8", _case(weights=None, distance_mode="euclidean", kernel_width=3.0)),
    ("L9", _case(return_input_shape=False)),
    ("L10", _case(n_samples=6,
                  samples=[[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]],
                  weights=[1.0, 0.5, 0.25, 0.75, 0.5, 0.25],
                  perturbations_per_eval=4)),
    ("L11", _case(tuple_inputs=True,
                  inputs=[[[1.0, 2.0]], [[3.0, 4.0]]],
                  baselines=[[[-0.5, -0.5]], [[-1.0, -1.0]]],
                  feature_mask=[[[0, 1]], [[1, 0]]],
                  forward_weights=[1.0, 2.0],
                  weights=[1.0, 0.5, 0.75, 0.25], alpha=0.0)),
    ("L12", _case(tuple_inputs=True,
                  inputs=[[[1.0, 2.0]], [[3.0, 4.0]]],
                  baselines=[[[0.0, 0.0]], [[0.0, 0.0]]],
                  feature_mask=None,
                  samples=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                  n_samples=6,
                  forward_weights=[1.0, 2.0],
                  weights=[1.0, 0.5, 0.75, 0.25, 0.5, 0.25])),
]


def check_feature_mask(solve):
    for tag, inputs, fm, tup in [
        ("fm-default", BASE["inputs"], None, False),
        ("fm-explicit", BASE["inputs"], [[0, 0, 1, 1]], False),
        ("fm-shift", BASE["inputs"], [[5, 5, 6, 6]], False),
        ("fm-gap", BASE["inputs"], [[0, 3, 3, 3]], False),
        ("fm-tuple-default", [[[1.0, 2.0]], [[3.0]]], None, True),
        ("fm-tuple-explicit", [[[1.0, 2.0]], [[3.0, 4.0]]],
         [[[2, 3]], [[3, 2]]], True),
    ]:
        got = solve("feature_mask", inputs=inputs, feature_mask=fm,
                    tuple_inputs=tup)
        exp = _exp_fm(inputs, fm, tup)
        for k in ("mask", "num_interp_features", "shifted", "default",
                  "min_index", "shapes"):
            _check(got[k], exp[k], tag + "." + k)
    _raises(lambda: solve("feature_mask", inputs=BASE["inputs"],
                          feature_mask=[[-1, 0, 1, 1]]),
            "non-negative", "fm-negative")
    _raises(lambda: solve("feature_mask", inputs=BASE["inputs"],
                          feature_mask=[[0.0, 0, 1, 1]]),
            "int", "fm-float-index")
    _raises(lambda: solve("feature_mask", inputs=BASE["inputs"],
                          feature_mask=[[0, 1]]),
            "row length", "fm-short-row")
    _raises(lambda: solve("feature_mask", inputs=BASE["inputs"],
                          tuple_inputs=True),
            "at least two", "fm-tuple-mismatch")
    _raises(lambda: solve("feature_mask", inputs=[]),
            "non-empty list", "fm-empty-inputs")
    _raises(lambda: solve("feature_mask", inputs=[[1.0, "x", 3.0, 4.0]]),
            "values must be finite numbers", "fm-non-numeric")
    _raises(lambda: solve("feature_mask", inputs=[[1.0, float("nan"), 3.0, 4.0]]),
            "values must be finite numbers", "fm-non-finite")
    _raises(lambda: solve("feature_mask", inputs=[[1.0, 2.0], [3.0, 4.0]]),
            "expects a single example", "fm-two-rows")
    _raises(lambda: solve("feature_mask", inputs=[[[1.0, 2.0]], [[3.0, 4.0]]]),
            "single input requires", "fm-tuple-shaped-single")
    _raises(lambda: solve("feature_mask", inputs=[[[1.0, 2.0]], [[3.0, 4.0]]],
                          tuple_inputs=True, feature_mask=[[[0, 1]]]),
            "tensor count must match inputs", "fm-mask-count")


def check_interp_transform(solve):
    cfg = BASE
    for s in ([1, 0], [0, 1], [1, 1], [0, 0]):
        got = solve("interp_transform", sample=s, inputs=cfg["inputs"],
                    baselines=cfg["baselines"], feature_mask=cfg["feature_mask"])
        bm = [s[m] for m in cfg["feature_mask"][0]]
        exp = [[cfg["inputs"][0][j] if bm[j] else cfg["baselines"][0][j]
                for j in range(4)]]
        _check(got["perturbed"], exp, "it-perturbed-%s" % s)
        _check(got["binary_mask"], bm, "it-binary-%s" % s)
        _check(got["num_interp_features"], 2, "it-F-%s" % s)
    nz = _case(baselines=[[-1.0, -2.0, -3.0, -4.0]])
    got = solve("interp_transform", sample=[0, 1], **{
        k: nz[k] for k in ("inputs", "baselines", "feature_mask", "tuple_inputs")})
    _check(got["perturbed"], [[-1.0, -2.0, 3.0, 4.0]], "it-nonzero-baseline")
    tup = _case(tuple_inputs=True, inputs=[[[1.0, 2.0]], [[3.0, 4.0]]],
                baselines=[[[-1.0, -2.0]], [[-3.0, -4.0]]],
                feature_mask=[[[0, 1]], [[1, 0]]])
    gt = solve("interp_transform", sample=[1, 0], **{
        k: tup[k] for k in ("inputs", "baselines", "feature_mask", "tuple_inputs")})
    _check(gt["perturbed"], [[[1.0, -2.0]], [[-3.0, 4.0]]], "it-tuple-perturbed")
    _check(gt["binary_mask"], [[1, 0], [0, 1]], "it-tuple-binary")
    _raises(lambda: solve("interp_transform", sample=[1], **{
        k: BASE[k] for k in ("inputs", "baselines", "feature_mask")}),
        "sample length", "it-sample-short")
    _raises(lambda: solve("interp_transform", sample=[1, 2], **{
        k: BASE[k] for k in ("inputs", "baselines", "feature_mask")}),
        "int 0 or 1", "it-sample-value")
    _raises(lambda: solve("interp_transform", sample=[True, False], **{
        k: BASE[k] for k in ("inputs", "baselines", "feature_mask")}),
        "int 0 or 1", "it-sample-bool")
    _raises(lambda: solve("interp_transform", sample=[1, 0],
                          inputs=BASE["inputs"],
                          baselines=[[0.0, 0.0]],
                          feature_mask=BASE["feature_mask"]),
            "baselines feature dim", "it-baseline-dim")
    _raises(lambda: solve("interp_transform", sample=[1, 0],
                          inputs=BASE["inputs"], baselines=BASE["baselines"],
                          feature_mask=[[0, 0, 1]]),
            "feature_mask row length", "it-mask-dim")
    _raises(lambda: solve("interp_transform", sample=[1, 0, 1], **{
        k: BASE[k] for k in ("inputs", "baselines", "feature_mask")}),
        "sample length must equal", "it-sample-long")
    _raises(lambda: solve("interp_transform", sample=[1, 0],
                          inputs=BASE["inputs"],
                          baselines=[[[0.0, 0.0, 0.0, 0.0]],
                                     [[0.0, 0.0, 0.0, 0.0]]],
                          feature_mask=BASE["feature_mask"]),
            "baselines tensor count", "it-bs-count")


def check_kernel(solve):
    pairs = [
        ("cos", "cosine", 1.0, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 0.0]),
        ("cos2", "cosine", 2.0, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 0.0]),
        ("same", "cosine", 1.0, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ("zero", "cosine", 1.0, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ("euc", "euclidean", 1.0, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 0.0]),
        ("euc3", "euclidean", 3.0, [1.0, 0.0, -1.0], [0.0, 1.0, 2.0]),
        ("2d", "cosine", 1.0, [[1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 0.0, 0.0]]),
    ]
    for tag, mode, kw, a, b in pairs:
        fa = a if not isinstance(a[0], list) else a[0]
        fb = b if not isinstance(b[0], list) else b[0]
        d, w = _kernel(mode, kw, fa, fb)
        got = solve("kernel", distance_mode=mode, kernel_width=kw,
                    original=a, perturbed=b)
        _check(got["distance"], d, tag + ".distance")
        _check(got["weight"], w, tag + ".weight", KERNEL_TOL)
        _check(got["distance_mode"], mode, tag + ".mode")
        _check(got["kernel_width"], kw, tag + ".kw")
    _raises(lambda: solve("kernel", distance_mode="manhattan", kernel_width=1.0,
                          original=[1.0, 2.0], perturbed=[1.0, 2.0]),
            "distance_mode", "kernel-bad-mode")
    _raises(lambda: solve("kernel", distance_mode="cosine", kernel_width=0.0,
                          original=[1.0, 2.0], perturbed=[1.0, 2.0]),
            "kernel_width", "kernel-bad-width")
    _raises(lambda: solve("kernel", distance_mode="cosine", kernel_width=True,
                          original=[1.0, 2.0], perturbed=[1.0, 2.0]),
            "kernel_width", "kernel-bool-width")
    _raises(lambda: solve("kernel", distance_mode="cosine", kernel_width=1.0,
                          original=[1.0, 2.0], perturbed=[1.0]),
            "same length", "kernel-length")


def check_surrogate(solve):
    cases = [
        ("sg0", [[1, 0], [0, 1], [1, 1], [0, 0]],
         [5.0, 25.0, 30.0, 0.0], [0.5, 0.25, 1.0, 0.75], 0.0),
        ("sg1", [[1, 0], [0, 1], [1, 1], [0, 0]],
         [5.0, 25.0, 30.0, 0.0], [0.5, 0.25, 1.0, 0.75], 0.5),
        ("sg2", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
         [1.0, 2.0, 4.0, 7.0], [1.0, 0.5, 0.25, 0.75], 0.125),
        ("sg3", [[1, 0], [0, 1]], [3.0, 7.0], [2.0, 3.0], 1.0),
    ]
    for tag, X, Y, W, alpha in cases:
        got = solve("surrogate", interp_samples=X, outputs=Y,
                    weights=W, alpha=alpha)
        n, F = len(X), len(X[0])
        gram = [[math.fsum(W[i] * X[i][a] * X[i][b] for i in range(n))
                 for b in range(F)] for a in range(F)]
        for a in range(F):
            gram[a][a] += alpha
        rhs = [math.fsum(W[i] * X[i][a] * Y[i] for i in range(n)) for a in range(F)]
        coefs = _solve(gram, rhs)
        _check(got["coefs"], coefs, tag + ".coefs")
        _check(got["gram"], gram, tag + ".gram")
        _check(got["rhs"], rhs, tag + ".rhs")
        _check(got["n_samples"], n, tag + ".n")
        _check(got["num_interp_features"], F, tag + ".F")
        _check(got["alpha"], alpha, tag + ".alpha")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0]],
                          outputs=[1.0, 2.0], weights=[1.0, 1.0]),
            "F features", "sg-ragged")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0, 2]],
                          outputs=[1.0, 2.0], weights=[1.0, 1.0]),
            "int 0 or 1", "sg-sample-value")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0, 1]],
                          outputs=[1.0], weights=[1.0, 1.0]),
            "outputs length", "sg-out-length")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0, 1]],
                          outputs=[1.0, 2.0], weights=[-1.0, 1.0]),
            "weights must be non-negative", "sg-neg-weight")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0, 1]],
                          outputs=[1.0, 2.0], weights=[1.0, 1.0], alpha=-1.0),
            "alpha", "sg-neg-alpha")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [1, 0]],
                          outputs=[1.0, 2.0], weights=[1.0, 1.0], alpha=0.0),
            "singular", "sg-singular")
    _raises(lambda: solve("surrogate", interp_samples=[], outputs=[], weights=[]),
            "non-empty list", "sg-empty")
    _raises(lambda: solve("surrogate", interp_samples=[[]], outputs=[1.0],
                          weights=[1.0]),
            "at least one feature", "sg-empty-row")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0, 1]],
                          outputs=[1.0, 2.0], weights=[1.0]),
            "weights length", "sg-w-short")
    _raises(lambda: solve("surrogate", interp_samples=[[True, False], [0, 1]],
                          outputs=[1.0, 2.0], weights=[1.0, 1.0]),
            "int 0 or 1", "sg-bool-sample")
    _raises(lambda: solve("surrogate", interp_samples=[[1, 0], [0, 1]],
                          outputs=["x", 2.0], weights=[1.0, 1.0]),
            "outputs must be finite numbers", "sg-out-string")


def check_lime(solve):
    for name, cfg in LIME_CASES:
        got = solve("lime", **cfg)
        exp = _exp_lime(cfg)
        tol = KERNEL_TOL if cfg.get("weights") is None else TOL
        for k in ("coefs", "attribution", "interp_samples", "outputs",
                  "perturbed", "binary_masks", "batches", "num_batches",
                  "batch_count", "loader_batch_size", "dataset_shapes",
                  "num_interp_features", "mask", "shifted", "default_mask",
                  "gram", "rhs", "return_input_shape", "alpha", "distance_mode"):
            _check(got[k], exp[k], name + "." + k, tol)
        _check(got["weights"], exp["weights"], name + ".weights", KERNEL_TOL)
        _must(set(got.keys()) == set(exp.keys()),
              name + ": key set mismatch: %r vs %r" % (sorted(got), sorted(exp)))
    # explicit-weight cases must be exactly reproducible
    r1 = solve("lime", **BASE)
    r2 = solve("lime", **BASE)
    _check(r1, r2, "L-deterministic")
    # alpha > 0 must shrink coefficients relative to alpha == 0
    r0 = solve("lime", **_case(alpha=0.0))
    rA = solve("lime", **_case(alpha=2.0))
    _must(any(abs(a) < abs(b) for a, b in zip(rA["coefs"], r0["coefs"])),
          "L-ridge: alpha must shrink coefficients")
    # return_input_shape=False must expose raw coefs
    rf = solve("lime", **_case(return_input_shape=False))
    _check(rf["attribution"], rf["coefs"], "L-raw-coefs")


def check_domain(solve):
    kw = dict(BASE)
    _raises(lambda: solve("lime", **dict(kw, n_samples=0)),
            "n_samples", "dom-n-samples-zero")
    _raises(lambda: solve("lime", **dict(kw, n_samples=True)),
            "positive int", "dom-n-samples-bool")
    _raises(lambda: solve("lime", **dict(kw, samples=[[1, 0]])),
            "samples length", "dom-samples-short")
    _raises(lambda: solve("lime", **dict(kw, samples=[[1], [0], [1], [0]])),
            "each sample must have", "dom-sample-cols")
    _raises(lambda: solve("lime", **dict(kw,
                                         samples=[[1, 0], [0, 1], [1, 2], [0, 0]])),
            "int 0 or 1", "dom-sample-value")
    _raises(lambda: solve("lime", **dict(kw, perturbations_per_eval=0)),
            "perturbations_per_eval", "dom-ppe-zero")
    _raises(lambda: solve("lime", **dict(kw, perturbations_per_eval=True)),
            "positive int", "dom-ppe-bool")
    _raises(lambda: solve("lime", **dict(kw, return_input_shape=1)),
            "return_input_shape", "dom-return-bool")
    _raises(lambda: solve("lime", **dict(kw, distance_mode="manhattan")),
            "distance_mode", "dom-distance")
    _raises(lambda: solve("lime", **dict(kw, kernel_width=0.0)),
            "kernel_width must be positive", "dom-kernel-width")
    _raises(lambda: solve("lime", **dict(kw, alpha=-1.0)),
            "alpha", "dom-alpha")
    _raises(lambda: solve("lime", **dict(kw, forward_weights=None)),
            "must be provided", "dom-forward-weights")
    _raises(lambda: solve("lime", **dict(kw, forward_weights=[])),
            "non-empty list", "dom-forward-empty")
    _raises(lambda: solve("lime", **dict(kw, forward_weights=[1.0])),
            "cover every input feature", "dom-forward-short")
    _raises(lambda: solve("lime", **dict(kw, forward_kind="cubic")),
            "forward_kind", "dom-forward-kind")
    _raises(lambda: solve("lime", **dict(kw, weights=[1.0, 0.5])),
            "weights length", "dom-weights-length")
    _raises(lambda: solve("lime", **dict(kw, weights=[1.0, -0.5, 1.0, 0.75])),
            "explicit weights", "dom-weights-negative")
    _raises(lambda: solve("lime", **dict(kw, baselines=[[0.0, 0.0]])),
            "baselines feature dim", "dom-baseline-dim")
    _raises(lambda: solve("lime", **dict(kw, feature_mask=[[0, 0, 1, 3]])),
            "num_interp_features", "dom-mask-gap")


def check_metadata(solve):
    cfg = BASE
    got = solve("lime", **cfg)
    _check(got["batches"], [1, 1, 1, 1], "meta-batches-ppe1")
    _check(got["num_batches"], 4, "meta-num-batches")
    _check(got["batch_count"], 4, "meta-batch-count")
    _check(got["loader_batch_size"], 4, "meta-loader")
    _check(got["dataset_shapes"], [[4, 2], [4], [4]], "meta-shapes")
    _check(got["num_interp_features"], 2, "meta-F")
    _check(got["shifted"], False, "meta-shifted")
    _check(got["default_mask"], False, "meta-default")
    _check(got["alpha"], 0.0, "meta-alpha")
    _check(got["distance_mode"], "cosine", "meta-distance")
    _check(got["return_input_shape"], True, "meta-return")
    _check(got["interp_samples"], cfg["samples"], "meta-samples")
    _check(got["outputs"], [5.0, 25.0, 30.0, 0.0], "meta-outputs")
    _check(got["weights"], cfg["weights"], "meta-weights")
    _check(got["binary_masks"],
           [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
           "meta-binary")
    _check(got["perturbed"][0], [[1.0, 2.0, 0.0, 0.0]], "meta-perturbed-0")
    _check(got["perturbed"][1], [[0.0, 0.0, 3.0, 4.0]], "meta-perturbed-1")
    for ppe, exp in ((2, [2, 2]), (3, [3, 1]), (5, [4])):
        g = solve("lime", **_case(perturbations_per_eval=ppe))
        _check(g["batches"], exp, "meta-ppe-%d" % ppe)
        _check(g["num_batches"], len(exp), "meta-nb-%d" % ppe)
        _check(g["batch_count"], 4, "meta-bc-%d" % ppe)
        _check(g["loader_batch_size"], 4, "meta-loader-%d" % ppe)
    g6 = solve("lime", **_case(feature_mask=None,
                               samples=[[1, 0, 0, 0], [0, 1, 0, 0],
                                        [0, 0, 1, 0], [0, 0, 0, 1],
                                        [1, 1, 1, 1], [0, 0, 0, 0]],
                               n_samples=6,
                               weights=[1.0, 0.5, 0.75, 0.25, 0.5, 0.25]))
    _check(g6["default_mask"], True, "meta-default-mask-true")
    _check(g6["num_interp_features"], 4, "meta-default-F")
    _check(g6["dataset_shapes"], [[6, 4], [6], [6]], "meta-default-shapes")
    gs = solve("lime", **_case(feature_mask=[[5, 5, 6, 6]]))
    _check(gs["shifted"], True, "meta-shift-true")
    _check(gs["mask"], [[0, 0, 1, 1]], "meta-shift-mask")


def check_unknown_mode(solve):
    _raises(lambda: solve("nope"), "unknown mode", "unknown-1")
    _raises(lambda: solve(""), "unknown mode", "unknown-2")
    _raises(lambda: solve("LIME", **BASE), "unknown mode", "unknown-3")
