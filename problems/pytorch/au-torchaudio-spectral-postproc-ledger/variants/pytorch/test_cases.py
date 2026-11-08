# -*- coding: utf-8 -*-
"""855 judge test cases. Every check receives `solve` from the judge."""
import cmath
import math


def _must(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _eq(a, b, tol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(_eq(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y, tol) for x, y in zip(a, b))
    if (isinstance(a, str) or isinstance(b, str) or a is None or b is None
            or isinstance(a, bool) or isinstance(b, bool)):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _check(got, exp, tag, tol=1e-9):
    _must(_eq(got, exp, tol), "%s: got %r expected %r" % (tag, got, exp))


def _raises(fn, keyword, tag):
    try:
        fn()
    except Exception as e:
        _must(keyword in str(e),
              "%s: message %r missing keyword %r" % (tag, str(e), keyword))
        return
    raise AssertionError("%s: expected an exception" % tag)


def _shape(v):
    if isinstance(v, (list, tuple)):
        return [len(v)] + (_shape(v[0]) if v else [])
    return []


def _flat(v):
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in _flat(item)]
    return [v]


def _prod(vals):
    out = 1
    for v in vals:
        out *= v
    return out


def _reshape(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat[:shape[0]])
    sub = _prod(shape[1:])
    return [_reshape(flat[i * sub:(i + 1) * sub], shape[1:])
            for i in range(shape[0])]


# ------------------------------------------------------- expectations
def exp_db(cfg):
    sh = _shape(cfg["x"])
    vals = [cfg["multiplier"] * math.log10(max(float(v), cfg["amin"]))
            - cfg["multiplier"] * cfg["db_multiplier"]
            for v in _flat(cfg["x"])]
    cuts = []
    if cfg.get("top_db") is not None:
        pc = sh[-3] if len(sh) > 2 else 1
        nb = _prod(sh[:-3]) if len(sh) > 3 else 1
        per = pc * sh[-2] * sh[-1]
        for b in range(nb):
            cut = max(vals[b * per:(b + 1) * per]) - cfg["top_db"]
            cuts.append(cut)
            vals[b * per:(b + 1) * per] = [max(v, cut) for v in vals[b * per:(b + 1) * per]]
    return _reshape(vals, sh), cuts


def exp_deltas(cfg):
    sh = _shape(cfg["specgram"])
    T = sh[-1]
    n = (cfg.get("win_length", 5) - 1) // 2
    den = 2.0 * sum(i * i for i in range(1, n + 1))
    flat = [float(v) for v in _flat(cfg["specgram"])]
    vals = []
    for c in range(len(flat) // T):
        row = flat[c * T:(c + 1) * T]
        if cfg.get("pad_mode", "replicate") == "replicate":
            padded = [row[0]] * n + row + [row[-1]] * n
        else:
            padded = [0.0] * n + row + [0.0] * n
        for t in range(T):
            acc = sum(i * (padded[t + n + i] - padded[t + n - i])
                      for i in range(1, n + 1))
            vals.append(acc / den)
    return _reshape(vals, sh)


def exp_windows(T, cw, mn, center):
    out = []
    for t in range(T):
        if center:
            ws, we = t - cw // 2, t - cw // 2 + cw
        else:
            ws, we = t - cw, t + 1
        if ws < 0:
            we -= ws
            ws = 0
        if not center and we > t:
            we = max(t + 1, mn)
        if we > T:
            ws -= we - T
            we = T
            ws = max(ws, 0)
        out.append([ws, we])
    return out


def exp_cmn(cfg):
    sh = _shape(cfg["specgram"])
    T, F = sh[-2], sh[-1]
    flat = [float(v) for v in _flat(cfg["specgram"])]
    nc = len(flat) // (T * F)
    wins = exp_windows(T, cfg.get("cmn_window", 600),
                       cfg.get("min_cmn_window", 100),
                       cfg.get("center", False))
    vals = []
    for c in range(nc):
        base = c * T * F
        for t, (ws, we) in enumerate(wins):
            count = we - ws
            for f in range(F):
                wv = [flat[base + j * F + f] for j in range(ws, we)]
                mean = sum(wv) / count
                v = flat[base + t * F + f] - mean
                if cfg.get("norm_vars", False):
                    if count == 1:
                        v = 0.0
                    else:
                        var = sum(q * q for q in wv) / count - mean * mean
                        v *= var ** -0.5
                vals.append(v)
    out = _reshape(vals, sh[:-2] + [T, F])
    if len(sh) == 2 and T == 1:
        out = out[0]
    return out


def exp_spec(cfg):
    sh = _shape(cfg["waveform"])
    wavs = cfg["waveform"] if len(sh) == 2 else [cfg["waveform"]]
    n_fft, hop = cfg["n_fft"], cfg["hop_length"]
    win = [float(v) for v in cfg["window"]]
    if cfg["win_length"] < n_fft:
        left = (n_fft - cfg["win_length"]) // 2
        win = [0.0] * left + win + [0.0] * (n_fft - cfg["win_length"] - left)
    norm = cfg.get("normalized", False)
    frame_norm = norm == "frame_length"
    window_norm = norm is True or norm == "window"
    fscale = math.sqrt(float(n_fft))
    wscale = math.sqrt(sum(v * v for v in cfg["window"]))
    bins = n_fft // 2 + 1 if cfg.get("onesided", True) else n_fft
    outs = []
    for wav in wavs:
        x = [float(v) for v in wav]
        pad = cfg.get("pad", 0)
        if pad > 0:
            x = [0.0] * pad + x + [0.0] * pad
        if cfg.get("center", True):
            p = n_fft // 2
            if cfg.get("pad_mode", "reflect") == "reflect":
                x = [x[p - i] for i in range(p)] + x + [x[-2 - i] for i in range(p)]
            else:
                x = [0.0] * p + x + [0.0] * p
        nf = 1 + (len(x) - n_fft) // hop
        zs = []
        for k in range(bins):
            line = []
            for t in range(nf):
                seg = [x[t * hop + j] * win[j] for j in range(n_fft)]
                z = sum(seg[j] * cmath.exp(-2j * math.pi * k * j / n_fft)
                        for j in range(n_fft))
                if frame_norm:
                    z /= fscale
                if window_norm:
                    z /= wscale
                line.append(z)
            zs.append(line)
        power = cfg.get("power", 2.0)
        if power is None:
            outs.append({"real": [[z.real for z in line] for line in zs],
                         "imag": [[z.imag for z in line] for line in zs]})
        else:
            outs.append([[abs(z) ** power for z in line] for line in zs])
    if len(sh) == 1:
        return outs[0]
    if cfg.get("power", 2.0) is None:
        return {"real": [o["real"] for o in outs],
                "imag": [o["imag"] for o in outs]}
    return outs


BASE_DB = {"x": [[1.0, 10.0, 100.0], [0.0, 0.5, 2.0]],
           "multiplier": 10.0, "amin": 1e-10, "db_multiplier": 0.0}
BASE_SPEC = {"waveform": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
             "window": [1.0, 1.0, 1.0, 1.0], "n_fft": 4,
             "hop_length": 2, "win_length": 4, "power": None,
             "normalized": False, "center": False}
X5 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
SEQ = [float(i + 1) for i in range(9)]


# ------------------------------------------------------------- checks
def check_db(solve):
    cases = [
        dict(BASE_DB, top_db=None),
        dict(BASE_DB, top_db=20.0),
        dict(BASE_DB, multiplier=20.0, top_db=20.0),
        dict(BASE_DB, db_multiplier=1.5, top_db=None),
        dict(BASE_DB, amin=1.0, top_db=15.0),
        {"x": [[[1.0, 100.0], [10.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
         "multiplier": 10.0, "amin": 1e-10, "db_multiplier": 0.0,
         "top_db": 20.0},
        {"x": [[[[1.0, 100.0], [10.0, 1.0]]],
               [[[0.01, 0.01], [0.01, 0.01]]]],
         "multiplier": 10.0, "amin": 1e-10, "db_multiplier": 0.0,
         "top_db": 20.0},
        {"x": [[[[[1.0, 100.0], [10.0, 1.0]]], [[[1.0, 1.0], [1.0, 1.0]]]],
               [[[[2.0, 4.0], [8.0, 16.0]]], [[[1.0, 3.0], [5.0, 7.0]]]]],
         "multiplier": 20.0, "amin": 1e-10, "db_multiplier": 0.0,
         "top_db": 10.0},
        # packed_channels > 1: chunk must include the channel dimension
        {"x": [[[[1.0, 100.0], [10.0, 1.0]], [[2.0, 4.0], [8.0, 16.0]]],
               [[[0.01, 0.01], [0.01, 0.01]], [[0.02, 0.02], [0.02, 0.02]]]],
         "multiplier": 10.0, "amin": 1e-10, "db_multiplier": 0.0,
         "top_db": 20.0},
    ]
    for ci, cfg in enumerate(cases):
        got = solve("db", **cfg)
        exp, cuts = exp_db(cfg)
        _check(got["out"], exp, "db%d.out" % ci)
        _check(got["shape"], _shape(cfg["x"]), "db%d.shape" % ci)
        sh = _shape(cfg["x"])
        pc = sh[-3] if len(sh) > 2 else 1
        nb = _prod(sh[:-3]) if len(sh) > 3 else 1
        _must(got["packed_channels"] == pc, "db%d.packed_channels" % ci)
        _must(got["num_batches"] == nb, "db%d.num_batches" % ci)
        _check(got["cutoffs"], cuts if cfg.get("top_db") is not None else None,
               "db%d.cutoffs" % ci)
    # per-batch independence of top_db cutoff
    four = cases[6]
    g = solve("db", **four)
    _must(g["cutoffs"][0] != g["cutoffs"][1],
          "db.4d cutoffs must differ per batch")
    _must(g["out"][0][0][0][0] > g["out"][1][0][0][0],
          "db.4d batch0 floor must be above batch1 floor")
    # 3D uses a single cutoff even though channels differ
    g3 = solve("db", **cases[5])
    _must(len(g3["cutoffs"]) == 1, "db.3d must have exactly one cutoff")
    # amin clamp (without top_db, so the clamp is directly observable)
    ga = solve("db", x=[[0.0]], multiplier=20.0, amin=1.0, db_multiplier=0.0)
    _check(ga["out"][0][0], 0.0, "db.amin clamp")
    ga2 = solve("db", x=[[0.0]], multiplier=20.0, amin=10.0, db_multiplier=0.0)
    _check(ga2["out"][0][0], 20.0, "db.amin clamp.10")


def check_db_inverse(solve):
    cases = [
        {"x": [0.0, 10.0, 20.0], "ref": 1.0, "power": 1.0},
        {"x": [0.0, 10.0, 20.0], "ref": 1.0, "power": 0.5},
        {"x": [[0.0, 10.0], [20.0, -10.0]], "ref": 3.0, "power": 0.5},
    ]
    for ci, cfg in enumerate(cases):
        got = solve("db_inverse", **cfg)
        exp = [cfg["ref"] * ((10.0 ** (0.1 * v)) ** cfg["power"])
               for v in _flat(cfg["x"])]
        _check(got["out"], _reshape(exp, _shape(cfg["x"])), "dbi%d.out" % ci)
        _check(got["shape"], _shape(cfg["x"]), "dbi%d.shape" % ci)
        _must(got["ref"] == cfg["ref"], "dbi%d.ref" % ci)
        _must(got["power"] == cfg["power"], "dbi%d.power" % ci)
    # round trip: power scale with multiplier=10
    x = [[1.0, 10.0], [100.0, 4.0]]
    d = solve("db", x=x, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    back = solve("db_inverse", x=d["out"], ref=1.0, power=1.0)
    _check(back["out"], x, "dbi.roundtrip.power")
    # amplitude scale uses multiplier=20 with power=0.5
    d20 = solve("db", x=x, multiplier=20.0, amin=1e-10, db_multiplier=0.0)
    back20 = solve("db_inverse", x=d20["out"], ref=1.0, power=0.5)
    _check(back20["out"], x, "dbi.roundtrip.amplitude")


def check_deltas(solve):
    cases = [
        {"specgram": [SEQ], "win_length": 3},
        {"specgram": [SEQ], "win_length": 5},
        {"specgram": [SEQ], "win_length": 7},
        {"specgram": [SEQ], "win_length": 5, "pad_mode": "constant"},
        {"specgram": [[SEQ, [v * 2.0 for v in SEQ]],
                      [[v * 0.5 for v in SEQ], [10.0 - v for v in SEQ]]],
         "win_length": 5},
        # even win_length: n=(W-1)//2 differs from W//2
        {"specgram": [SEQ], "win_length": 4},
    ]
    for ci, cfg in enumerate(cases):
        got = solve("deltas", **cfg)
        _check(got["out"], exp_deltas(cfg), "dl%d.out" % ci, 1e-9)
        _check(got["shape"], _shape(cfg["specgram"]), "dl%d.shape" % ci)
        W = cfg.get("win_length", 5)
        n = (W - 1) // 2
        _must(got["n"] == n, "dl%d.n" % ci)
        _check(got["denom"], n * (n + 1) * (2 * n + 1) / 3, "dl%d.denom" % ci)
        _check(got["kernel"], list(range(-n, n + 1)), "dl%d.kernel" % ci)
        _must(got["mode"] == cfg.get("pad_mode", "replicate"), "dl%d.mode" % ci)
        _must(got["num_channels"] == len(_flat(cfg["specgram"])) // len(SEQ),
              "dl%d.num_channels" % ci)
    # cross-correlation sign: increasing series gives positive delta
    g = solve("deltas", specgram=[SEQ], win_length=5)
    _must(g["out"][0][4] > 0.0, "deltas sign must be positive for rising input")
    dec = solve("deltas", specgram=[list(reversed(SEQ))], win_length=5)
    _must(dec["out"][0][4] < 0.0, "deltas sign must be negative for falling input")
    # replicate vs constant MUST differ at the boundary but agree in the interior
    a = solve("deltas", specgram=[[1.0, 2.0, 3.0, 4.0, 5.0]], win_length=5)
    b = solve("deltas", specgram=[[1.0, 2.0, 3.0, 4.0, 5.0]], win_length=5,
              pad_mode="constant")
    _must(not _eq(a["out"], b["out"]), "deltas pad modes must differ at boundary")
    _check(a["out"][0][2], b["out"][0][2], "deltas interior agrees")
    _must(a["out"][0][0] != b["out"][0][0], "deltas first frame must differ")


def check_cmn(solve):
    combos = ((2, 2, False, False), (3, 2, False, False),
              (3, 2, True, False), (5, 4, True, False),
              (2, 100, False, False), (3, 2, True, True),
              (2, 3, False, True))
    for ci, (cw, mn, ce, nv) in enumerate(combos):
        cfg = {"specgram": X5, "cmn_window": cw, "min_cmn_window": mn,
               "center": ce, "norm_vars": nv}
        got = solve("cmn", **cfg)
        _check(got["out"], exp_cmn(cfg), "cmn%d.out" % ci, 1e-8)
        _check(got["shape"], _shape(X5), "cmn%d.shape" % ci)
        _check(got["windows"], exp_windows(5, cw, mn, ce), "cmn%d.windows" % ci)
        _must(got["num_frames"] == 5, "cmn%d.num_frames" % ci)
        _must(got["num_feats"] == 2, "cmn%d.num_feats" % ci)
        _must(got["norm_vars"] == nv, "cmn%d.norm_vars" % ci)
        _must(all(w[0] >= 0 and w[1] <= 5 and w[1] > w[0] for w in got["windows"]),
              "cmn%d window bounds" % ci)
    # center vs non-center MUST produce different schedules and values
    c = solve("cmn", specgram=X5, cmn_window=3, min_cmn_window=2, center=True)
    f = solve("cmn", specgram=X5, cmn_window=3, min_cmn_window=2, center=False)
    _must(c["windows"] != f["windows"], "cmn center schedule must differ")
    _must(not _eq(c["out"], f["out"]), "cmn center output must differ")
    # 3D input keeps leading channel dim
    cfg3 = {"specgram": [X5, [[v * 2.0 for v in row] for row in X5]],
            "cmn_window": 3, "min_cmn_window": 2, "center": True}
    g3 = solve("cmn", **cfg3)
    _check(g3["out"], exp_cmn(cfg3), "cmn.3d", 1e-8)
    _check(g3["shape"], [2, 5, 2], "cmn.3d.shape")
    # official squeeze(0) quirk for 2D input with a single frame
    one = {"specgram": [[1.0, 2.0]], "cmn_window": 1,
           "min_cmn_window": 1, "center": False, "norm_vars": True}
    g1 = solve("cmn", **one)
    _check(g1["out"], [0.0, 0.0], "cmn.t1.zeros")
    _check(g1["shape"], [2], "cmn.t1.squeeze")
    # norm_vars actually changes the result when window_frames > 1
    a = solve("cmn", specgram=X5, cmn_window=3, min_cmn_window=2,
              center=True, norm_vars=False)
    b = solve("cmn", specgram=X5, cmn_window=3, min_cmn_window=2,
              center=True, norm_vars=True)
    _must(not _eq(a["out"], b["out"]), "cmn norm_vars must change output")


def check_spectrogram(solve):
    variants = (
        ("plain", {}),
        ("pad", {"pad": 2}),
        ("center", {"center": True}),
        ("center-const", {"center": True, "pad_mode": "constant"}),
        ("win2", {"window": [0.5, -0.5], "win_length": 2}),
        ("normw", {"normalized": "window"}),
        ("normf", {"normalized": "frame_length"}),
        ("normtrue", {"normalized": True}),
        ("power1", {"power": 1.0}),
        ("power2", {"power": 2.0}),
        ("power3", {"power": 3.0}),
        ("onesided", {"onesided": False}),
        ("deprecated", {"return_complex": True}),
    )
    for tag, extra in variants:
        cfg = dict(BASE_SPEC, **extra)
        got = solve("spectrogram", **cfg)
        exp = exp_spec(cfg)
        _check(got["out"], exp, "sp.%s.out" % tag, 1e-8)
        bins = cfg["n_fft"] // 2 + 1 if cfg.get("onesided", True) else cfg["n_fft"]
        frames = _shape(exp["real"] if isinstance(exp, dict) else exp)[-1]
        _check(got["shape"], [bins, frames], "sp.%s.shape" % tag)
        _must(got["bins"] == bins, "sp.%s.bins" % tag)
        _must(got["frames"] == frames, "sp.%s.frames" % tag)
        norm = cfg.get("normalized", False)
        _must(got["frame_norm"] == (norm == "frame_length"), "sp.%s.frame_norm" % tag)
        _must(got["window_norm"] == (norm is True or norm == "window"),
              "sp.%s.window_norm" % tag)
        _must(got["normalized"] == norm, "sp.%s.normalized" % tag)
        _must(got["power"] == cfg.get("power", 2.0), "sp.%s.power" % tag)
        _must(got["pad"] == cfg.get("pad", 0), "sp.%s.pad" % tag)
        _must(got["center"] == cfg.get("center", True), "sp.%s.center" % tag)
        _must(got["pad_mode"] == cfg.get("pad_mode", "reflect"), "sp.%s.pad_mode" % tag)
        _must(got["onesided"] == cfg.get("onesided", True), "sp.%s.onesided" % tag)
        w = cfg["window"]
        if cfg["win_length"] < cfg["n_fft"]:
            left = (cfg["n_fft"] - cfg["win_length"]) // 2
            w = [0.0] * left + list(w) + [0.0] * (cfg["n_fft"] - cfg["win_length"] - left)
        _check(got["window_padded"], list(w), "sp.%s.window_padded" % tag)
        _must((len(got["warnings"]) > 0) == (cfg.get("return_complex") is not None),
              "sp.%s.warnings" % tag)
    # 2D waveform: leading channel dimension is unfolded
    cfg2 = dict(BASE_SPEC, waveform=[BASE_SPEC["waveform"],
                                    [v * 0.5 for v in BASE_SPEC["waveform"]]])
    g2 = solve("spectrogram", **cfg2)
    _check(g2["out"], exp_spec(cfg2), "sp.2d.out", 1e-8)
    _check(g2["shape"], [2, 3, 3], "sp.2d.shape")
    cfg2p = dict(cfg2, power=2.0)
    g2p = solve("spectrogram", **cfg2p)
    _check(g2p["out"], exp_spec(cfg2p), "sp.2d.power2", 1e-8)
    # bins != frames so that shape [C, bins, frames] is unambiguous
    cfg2pad = dict(BASE_SPEC, waveform=[BASE_SPEC["waveform"],
                                        [v * 0.5 for v in BASE_SPEC["waveform"]]],
                   pad=2)
    g2pad = solve("spectrogram", **cfg2pad)
    _check(g2pad["out"], exp_spec(cfg2pad), "sp.2d.pad.out", 1e-8)
    _check(g2pad["shape"], [2, 3, 5], "sp.2d.pad.shape")
    # normalization branches MUST actually change the numeric output
    base = solve("spectrogram", **BASE_SPEC)["out"]
    nw = solve("spectrogram", **dict(BASE_SPEC, normalized="window"))["out"]
    nf = solve("spectrogram", **dict(BASE_SPEC, normalized="frame_length"))["out"]
    _must(not _eq(base, nw), "sp window_norm must change output")
    _must(not _eq(base, nf), "sp frame_norm must change output")
    _check(solve("spectrogram", **dict(BASE_SPEC, normalized=True))["out"],
           nw, "sp normalized True maps to window")
    # power branches
    p1 = solve("spectrogram", **dict(BASE_SPEC, power=1.0))["out"]
    p2 = solve("spectrogram", **dict(BASE_SPEC, power=2.0))["out"]
    _must(not _eq(p1, p2), "sp power=1 and power=2 must differ")
    _must(_eq(p1, [[math.hypot(base["real"][k][t], base["imag"][k][t])
                    for t in range(3)] for k in range(3)]),
          "sp power=1 equals abs")


def check_domain(solve):
    _raises(lambda: solve("db", x=[[1.0]], multiplier=10.0,
                          amin=0.0, db_multiplier=0.0),
            "amin must be a positive number", "dom.db.amin")
    _raises(lambda: solve("db", x=[[1.0]], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0, top_db="x"),
            "top_db must be a number or None", "dom.db.top_db")
    _raises(lambda: solve("db", x=[1.0], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0),
            "at least two dimensions", "dom.db.dims")
    _raises(lambda: solve("db_inverse", x=[[1.0]], ref="x", power=1.0),
            "ref must be a number", "dom.dbi.ref")
    _raises(lambda: solve("deltas", specgram=[[1.0, 2.0, 3.0]], win_length=2),
            "greater than or equal to 3", "dom.dl.win")
    _raises(lambda: solve("deltas", specgram=[[1.0, 2.0, 3.0]], win_length=3,
                          pad_mode="bogus"),
            "mode must be replicate or constant", "dom.dl.pad_mode")
    _raises(lambda: solve("cmn", specgram=[[1.0, 2.0]], cmn_window=0),
            "cmn_window must be a positive int", "dom.cmn.cw")
    _raises(lambda: solve("cmn", specgram=[[1.0, 2.0]], min_cmn_window=0),
            "min_cmn_window must be a positive int", "dom.cmn.mn")
    _raises(lambda: solve("cmn", specgram=[[1.0, 2.0]], center=1),
            "center must be a bool", "dom.cmn.center")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, normalized="bogus")),
            "Invalid normalized parameter", "dom.sp.normalized.str")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, normalized=7)),
            "Input type not supported", "dom.sp.normalized.type")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, win_length=8)),
            "win_length must be in", "dom.sp.win_length")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, window=[1.0, 1.0])),
            "window length must equal win_length", "dom.sp.window_len")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, pad=-1)),
            "pad must be a non-negative int", "dom.sp.pad")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, pad_mode="x")),
            "pad_mode must be reflect or constant", "dom.sp.pad_mode")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, power="x")),
            "power must be a number or None", "dom.sp.power")
    _raises(lambda: solve("spectrogram", waveform=[[[1.0, 2.0]]], window=[1.0],
                          n_fft=2, hop_length=1, win_length=1),
            "waveform must be 1D or 2D", "dom.sp.waveform.dims")
    # helper / domain guards that must be observable
    _raises(lambda: solve("db", x=[[True]], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0),
            "must contain only numbers", "dom.db.bool")
    _raises(lambda: solve("db", x=[], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0),
            "at least two dimensions", "dom.db.empty")
    _raises(lambda: solve("db", x=[[]], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0),
            "dimensions must be non-empty", "dom.db.emptydim")
    _raises(lambda: solve("db", x=[["a"]], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0),
            "must contain only numbers", "dom.db.str")
    _raises(lambda: solve("db", x=[[1.0, 2.0], [3.0]], multiplier=10.0,
                          amin=1.0, db_multiplier=0.0),
            "must be rectangular", "dom.db.ragged")
    _raises(lambda: solve("db", x=[[1.0]], multiplier="x",
                          amin=1.0, db_multiplier=0.0),
            "multiplier must be a number", "dom.db.mult")
    _raises(lambda: solve("db", x=[[1.0]], multiplier=10.0,
                          amin=1.0, db_multiplier="x"),
            "db_multiplier must be a number", "dom.db.dbm")
    _raises(lambda: solve("db_inverse", x=1.0, ref=1.0, power=1.0),
            "at least one dimension", "dom.dbi.scalar")
    _raises(lambda: solve("db_inverse", x=[1.0], ref=1.0, power="x"),
            "power must be a number", "dom.dbi.power")
    _raises(lambda: solve("deltas", specgram=[[1.0, 2.0, 3.0]],
                          win_length=3.5),
            "win_length must be an int", "dom.dl.win_float")
    _raises(lambda: solve("cmn", specgram=[[1.0, 2.0]], norm_vars=1),
            "norm_vars must be a bool", "dom.cmn.nv")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, window=[])),
            "window must be a non-empty 1D list", "dom.sp.window_empty")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, waveform=["x"] * 8)),
            "waveform must contain only numbers", "dom.sp.wave_str")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, window=["x"] * 4)),
            "window must contain only numbers", "dom.sp.win_str")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, n_fft=0)),
            "n_fft must be a positive int", "dom.sp.nfft")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, hop_length=0)),
            "hop_length must be a positive int", "dom.sp.hop")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, onesided=1)),
            "onesided must be a bool", "dom.sp.onesided")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, center=1)),
            "center must be a bool", "dom.sp.center")
    _raises(lambda: solve("spectrogram", **dict(BASE_SPEC, window=[0.0] * 4)),
            "window energy must be positive", "dom.sp.wscale")
    _raises(lambda: solve("spectrogram", waveform=[1.0, 2.0, 3.0],
                          window=[1.0] * 8, n_fft=8, hop_length=2,
                          win_length=8, center=True),
            "reflect pad requires", "dom.sp.reflect")
    _raises(lambda: solve("spectrogram", waveform=[1.0, 2.0, 3.0],
                          window=[1.0] * 8, n_fft=8, hop_length=2,
                          win_length=8, center=False),
            "waveform too short for n_fft", "dom.sp.nf")
    _raises(lambda: solve("unknown_mode"), "unknown mode", "dom.unknown")


def check_metadata(solve):
    # db metadata
    cfg = {"x": [[[[1.0, 100.0], [10.0, 1.0]]], [[[1.0, 1.0], [1.0, 1.0]]]],
           "multiplier": 10.0, "amin": 1e-10, "db_multiplier": 0.0,
           "top_db": 20.0}
    g = solve("db", **cfg)
    _must(g["packed_channels"] == 1, "meta.db.packed_channels")
    _must(g["num_batches"] == 2, "meta.db.num_batches")
    _check(g["cutoffs"], [0.0, -20.0], "meta.db.cutoffs")
    _must(solve("db", **dict(cfg, top_db=None))["cutoffs"] is None,
          "meta.db.cutoffs.None")
    # deltas metadata
    d = solve("deltas", specgram=[SEQ], win_length=7)
    _check(d["kernel"], [-3, -2, -1, 0, 1, 2, 3], "meta.dl.kernel")
    _check(d["denom"], 28.0, "meta.dl.denom")
    _must(d["n"] == 3, "meta.dl.n")
    _must(d["num_channels"] == 1, "meta.dl.channels")
    _must(d["mode"] == "replicate", "meta.dl.mode.default")
    _must(solve("deltas", specgram=[SEQ], win_length=5,
                pad_mode="constant")["mode"] == "constant",
          "meta.dl.mode.constant")
    # cmn metadata
    c = solve("cmn", specgram=X5, cmn_window=3, min_cmn_window=2, center=True)
    _check(c["windows"], [[0, 3], [0, 3], [1, 4], [2, 5], [2, 5]],
           "meta.cmn.windows.center")
    f = solve("cmn", specgram=X5, cmn_window=3, min_cmn_window=2, center=False)
    _check(f["windows"], [[0, 2], [0, 2], [0, 3], [0, 4], [1, 5]],
           "meta.cmn.windows.left")
    _must(c["num_frames"] == 5 and c["num_feats"] == 2, "meta.cmn.dims")
    _must(c["norm_vars"] is False, "meta.cmn.norm_vars.default")
    # spectrogram metadata
    s = solve("spectrogram", **BASE_SPEC)
    _must(s["bins"] == 3 and s["frames"] == 3, "meta.sp.bins.frames")
    _must(s["frame_norm"] is False and s["window_norm"] is False,
          "meta.sp.norm.default")
    _check(s["window_padded"], [1.0, 1.0, 1.0, 1.0], "meta.sp.window_padded")
    _must(s["warnings"] == [], "meta.sp.warnings.default")
    _must(s["center"] is False and s["pad_mode"] == "reflect",
          "meta.sp.center.pad_mode.default")
    _must(s["onesided"] is True, "meta.sp.onesided.default")
    sw = solve("spectrogram", **dict(BASE_SPEC, window=[0.5, -0.5], win_length=2))
    _check(sw["window_padded"], [0.0, 0.5, -0.5, 0.0],
           "meta.sp.window_padded.zero")
    _must(len(solve("spectrogram", **dict(BASE_SPEC, return_complex=False))
              ["warnings"]) == 1, "meta.sp.warnings.deprecated")


def check_unknown_mode(solve):
    _raises(lambda: solve("nope"), "unknown mode: 'nope'", "unknown.nope")
    _raises(lambda: solve("DB"), "unknown mode: 'DB'", "unknown.case")
    _raises(lambda: solve(""), "unknown mode: ''", "unknown.empty")
