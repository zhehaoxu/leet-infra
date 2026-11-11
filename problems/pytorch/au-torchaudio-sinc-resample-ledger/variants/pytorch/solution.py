#!/usr/bin/env python3
"""859 reference solution: torchaudio sinc-resampling ledger.

Pure Python, no torch, no numpy, no global RNG.  Five independent modes
reproduce the official ``torchaudio.functional`` mechanisms:

    meta     : gcd reduction, base_freq / width / shape, method + beta ledger
    i0       : modified Bessel I0 series (float64 and the float32 path)
    kernel   : t grid, clamp, Hann/Kaiser window, sinc, scale and kernels
    apply    : pad / stride-``o`` cross-correlation / interleave / truncate
    resample : identity short circuit and the full end-to-end pipeline

Judged domain: float64 throughout (the official ``resample`` passes
``waveform.dtype`` into the kernel builder).  Kaiser beta is the official
default, or a value whose float32 ``i0`` agrees bit-for-bit with the series;
see notes.md for the stripped conventions.
"""

import math
import struct

BETA_DEFAULT = 14.769656459379492
ALIASES = {
    "sinc_interpolation": "sinc_interp_hann",
    "kaiser_window": "sinc_interp_kaiser",
}
VALID_METHODS = ("sinc_interp_hann", "sinc_interp_kaiser")


# ---------------------------------------------------------------------------
# nested-list utilities
# ---------------------------------------------------------------------------
def shape_of(x):
    if not isinstance(x, list):
        return []
    if not x:
        return [0]
    return [len(x)] + shape_of(x[0])


def numel(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def flatten(x):
    if not isinstance(x, list):
        return [x]
    out = []
    for v in x:
        out.extend(flatten(v))
    return out


def unflatten(vals, shape):
    if not shape:
        return vals[0]
    step = numel(shape[1:])
    return [unflatten(vals[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def require_numbers(x, what):
    if not isinstance(x, list) or not x:
        raise ValueError("%s must be a non-empty numeric array" % what)
    vals = flatten(x)
    if not vals or not all(is_number(v) for v in vals):
        raise ValueError("%s must be a non-empty numeric array" % what)


def require_int(v, what, minimum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("%s must be an integer" % what)
    if minimum is not None and v < minimum:
        raise ValueError("%s must be at least %d" % (what, minimum))


def require_freq(v, what):
    if isinstance(v, bool) or not is_number(v) or int(v) != v:
        raise ValueError("Frequencies must be of integer type")
    require_int(int(v), what, 1)
    return int(v)


def require_positive(v, what):
    if not is_number(v) or v <= 0:
        raise ValueError("%s must be positive" % what)
    return float(v)


def f32(x):
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def i0_series(x):
    """Modified Bessel I0 by its power series: sum (x^2/4)^k / (k!)^2."""
    x = abs(float(x))
    a = x * x / 4.0
    term, total = 1.0, 1.0
    for k in range(1, 300):
        term *= a / (k * k)
        total += term
        if term < 1e-18 * total:
            break
    return total


def pack(x, what):
    shp = shape_of(x)
    if len(shp) < 1:
        raise ValueError("%s must have a time axis" % what)
    lead = shp[:-1]
    if not lead:
        rows = [list(x)]
    else:
        vals = flatten(x)
        step = shp[-1]
        rows = [vals[i * step:(i + 1) * step]
                for i in range(len(vals) // step)]
    return rows, lead


# ---------------------------------------------------------------------------
# shared parameter ledger
# ---------------------------------------------------------------------------
def params(kw):
    orig = require_freq(kw.get("orig_freq", 8), "orig_freq")
    new = require_freq(kw.get("new_freq", 4), "new_freq")
    W = kw.get("lowpass_filter_width", 6)
    require_int(W, "lowpass_filter_width", 1)
    rolloff = require_positive(kw.get("rolloff", 0.99), "rolloff")
    method = kw.get("resampling_method", "sinc_interp_hann")
    if not isinstance(method, str):
        raise ValueError("Invalid resampling method")
    deprecated = method in ALIASES
    replacement = ALIASES.get(method)
    if not deprecated and method not in VALID_METHODS:
        raise ValueError("Invalid resampling method: %s" % method)
    # QUIRK: the official alias branch only warns; it never rewrites the
    # local name, so every name != "sinc_interp_hann" takes the Kaiser branch.
    branch = "hann" if method == "sinc_interp_hann" else "kaiser"
    beta_in = kw.get("beta", None)
    if beta_in is not None and not is_number(beta_in):
        raise ValueError("beta must be a number")
    beta = f32(BETA_DEFAULT if beta_in is None else beta_in)

    g = math.gcd(orig, new)
    o, n = orig // g, new // g
    base = min(o, n) * rolloff
    width = int(math.ceil(W * o / base))
    K = 2 * width + o
    return {
        "orig": orig, "new": new, "gcd": g, "o": o, "n": n,
        "W": W, "rolloff": rolloff, "base": base, "width": width,
        "K": K, "method": method, "deprecated": deprecated,
        "replacement": replacement, "branch": branch,
        "beta": beta, "beta_default_used": beta_in is None,
        "scale": base / o,
        "warnings": (["deprecated: %s -> %s" % (method, replacement)]
                     if deprecated else []),
    }


def meta_view(p):
    return {
        "gcd": p["gcd"],
        "orig_freq_reduced": p["o"],
        "new_freq_reduced": p["n"],
        "base_freq": p["base"],
        "width": p["width"],
        "kernel_length": p["K"],
        "shape": [p["n"], 1, p["K"]],
        "lowpass_filter_width": p["W"],
        "rolloff": p["rolloff"],
        "method": p["method"],
        "deprecated": p["deprecated"],
        "replacement": p["replacement"],
        "window_branch": p["branch"],
        "warnings": list(p["warnings"]),
        "beta": p["beta"],
        "beta_default_used": p["beta_default_used"],
        "scale": p["scale"],
    }


# ---------------------------------------------------------------------------
# mode 1: meta
# ---------------------------------------------------------------------------
def _meta(kw):
    p = params(kw)
    return meta_view(p)


# ---------------------------------------------------------------------------
# mode 2: i0
# ---------------------------------------------------------------------------
def _i0(kw):
    values = kw.get("values")
    if is_number(values):
        values = [values]
    require_numbers(values, "values")
    vals = flatten(values)
    exact = [i0_series(v) for v in vals]
    return {
        "i0": unflatten(exact, shape_of(values)) if isinstance(
            kw.get("values"), list) else exact[0],
        "i0_float32": unflatten([f32(v) for v in exact], shape_of(values))
        if isinstance(kw.get("values"), list) else f32(exact[0]),
        "shape": shape_of(values) if isinstance(kw.get("values"), list) else [],
        "beta_default": BETA_DEFAULT,
        "beta_float32": f32(BETA_DEFAULT),
        "i0_beta_float32": f32(i0_series(f32(BETA_DEFAULT))),
    }


# ---------------------------------------------------------------------------
# mode 3: kernel
# ---------------------------------------------------------------------------
def _kernel(kw):
    p = params(kw)
    o, n, W, base, width, K = (p["o"], p["n"], p["W"], p["base"],
                               p["width"], p["K"])
    idx = [j / o for j in range(-width, width + o)]
    den = f32(i0_series(p["beta"]))
    kernels, t_rows, w_rows, s_rows = [], [], [], []
    for j in range(n):
        t = [max(-W, min(W, (-j / n + v) * base)) for v in idx]
        if p["branch"] == "hann":
            win = [math.cos(v * math.pi / W / 2) ** 2 for v in t]
        else:
            win = [i0_series(p["beta"] * math.sqrt(1 - (v / W) ** 2)) / den
                   for v in t]
        tp = [v * math.pi for v in t]
        sinc = [1.0 if v == 0 else math.sin(v) / v for v in tp]
        kernels.append([a * b * p["scale"] for a, b in zip(sinc, win)])
        t_rows.append(t)
        w_rows.append(win)
        s_rows.append(sinc)
    return {
        "kernels": kernels, "shape": [n, K],
        "t": t_rows, "window": w_rows, "sinc": s_rows,
        "width": width, "kernel_length": K, "scale": p["scale"],
        "beta_float32": p["beta"], "i0_beta_float32": den,
        "meta": meta_view(p),
    }


# ---------------------------------------------------------------------------
# mode 4: apply
# ---------------------------------------------------------------------------
def _apply(kw):
    wf = kw.get("waveform")
    require_numbers(wf, "waveform")
    orig = require_freq(kw.get("orig_freq", 8), "orig_freq")
    new = require_freq(kw.get("new_freq", 4), "new_freq")
    kernels = kw.get("kernels")
    require_numbers(kernels, "kernels")
    if shape_of(kernels) and len(shape_of(kernels)) != 2:
        raise ValueError("kernels must be a 2-D (new_freq, taps) array")
    width = kw.get("width")
    require_int(width, "width", 0)
    g = math.gcd(orig, new)
    o, n = orig // g, new // g
    K = 2 * width + o
    if shape_of(kernels) != [n, K]:
        raise ValueError("kernel shape must be [new_freq, 2*width+orig]")
    rows, lead = pack(wf, "waveform")
    out_rows = []
    num_frames = None
    for row in rows:
        L = len(row)
        padded = [0.0] * width + list(row) + [0.0] * (width + o)
        nf = L // o + 1
        num_frames = nf
        phases = []
        for j in range(n):
            kj = kernels[j]
            phases.append([sum(padded[s * o + t] * kj[t] for t in range(K))
                           for s in range(nf)])
        flat = [phases[j][s] for s in range(nf) for j in range(n)]
        target = int(math.ceil(n * L / o))
        out_rows.append(flat[:target])
    target = int(math.ceil(n * len(rows[0]) / o))
    shape = lead + [target]
    return {
        "out": unflatten(flatten(out_rows), shape),
        "shape": shape,
        "target_length": target,
        "num_frames": num_frames,
        "conv_length": num_frames * n,
        "padded_length": len(rows[0]) + 2 * width + o,
        "width": width,
        "stride": o,
    }


# ---------------------------------------------------------------------------
# mode 5: resample
# ---------------------------------------------------------------------------
def _resample(kw):
    wf = kw.get("waveform")
    require_numbers(wf, "waveform")
    # The <=0 guard is in ``resample`` only (the kernel builder has none).
    of, nf = kw.get("orig_freq", 8), kw.get("new_freq", 4)
    if is_number(of) and of <= 0 or is_number(nf) and nf <= 0:
        raise ValueError("Original frequency and desired frequency should be positive")
    p = params(kw)
    if p["orig"] == p["new"]:
        return {
            "out": unflatten(flatten(wf), shape_of(wf)),
            "shape": shape_of(wf),
            "target_length": shape_of(wf)[-1],
            "identity": True,
            "meta": meta_view(p),
        }
    k = _kernel(kw)
    a = _apply(dict(kw, kernels=k["kernels"], width=k["width"]))
    return {
        "out": a["out"], "shape": a["shape"],
        "target_length": a["target_length"], "identity": False,
        "meta": meta_view(p),
    }


MODES = ("meta", "i0", "kernel", "apply", "resample")


def solve(mode, **kw):
    if mode == "meta":
        return _meta(kw)
    if mode == "i0":
        return _i0(kw)
    if mode == "kernel":
        return _kernel(kw)
    if mode == "apply":
        return _apply(kw)
    if mode == "resample":
        return _resample(kw)
    raise ValueError("unknown mode: %r" % (mode,))
