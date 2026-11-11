#!/usr/bin/env python3
"""861 reference solution: torchaudio ITU-R BS.1770-4 loudness-gate ledger.

Pure Python (stdlib only).  Reimplements the official `loudness` chain from
`functional.py` together with the `filtering.py` primitives it depends on
(`treble_biquad` / `highpass_biquad` / `biquad` / `lfilter`) as five
independently judged modes.

Judged input contract
---------------------
All inputs are JSON-friendly numbers / lists.  `waveform` for `blocks` and
`loudness` is exactly two-dimensional `(channels, time)`; the official
`(..., channels, time)` batch axis is stripped from the judged domain (the
official 0-dim LKFS tensor is returned as a Python float).  `kweight` and
`lfilter` accept any number of leading dimensions and rebuild them.  All
arithmetic is float64; no torch / torchaudio / numpy, no global RNG.
"""
import math

MODES = ("coeffs", "lfilter", "kweight", "blocks", "loudness")

GATE_DURATION = 0.4
OVERLAP = 0.75
GAMMA_ABS = -70.0
KWEIGHT_BIAS = -0.691
CHANNEL_GAINS = [1.0, 1.0, 1.0, 1.41, 1.41]

KW_TREBLE_GAIN = 4.0
KW_TREBLE_FREQ = 1500.0
KW_TREBLE_Q = 1.0 / math.sqrt(2.0)
KW_HIGHPASS_FREQ = 38.0
KW_HIGHPASS_Q = 0.5


# ----------------------------------------------------------------- validation
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def require_numbers(v, name):
    if not isinstance(v, list) or not v:
        raise ValueError(name + " must be a non-empty list of numbers")
    if not all(_is_num(x) for x in v):
        raise ValueError(name + " must contain only numbers")
    return v


def require_matrix(v, name):
    if not isinstance(v, list) or not v or not isinstance(v[0], list):
        raise ValueError(name + " must be a non-empty 2-D list")
    width = len(v[0])
    if width == 0:
        raise ValueError(name + " must have at least one column")
    for row in v:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(name + " must be rectangular")
        if not all(_is_num(x) for x in row):
            raise ValueError(name + " must contain only numbers")
    return v


def require_int(v, name, low=None):
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(name + " must be an int")
    if low is not None and v < low:
        raise ValueError(name + " must be >= " + str(low))
    return v


def require_bool(v, name):
    if not isinstance(v, bool):
        raise ValueError(name + " must be a bool")
    return v


def _pack_rows(v, name):
    """Flatten any leading dimensions into a list of 1-D rows + the shape."""
    if not isinstance(v, list) or not v:
        raise ValueError(name + " must be a non-empty list")
    if all(_is_num(e) for e in v):
        return [list(v)], [len(v)]
    rows, shapes = [], []
    for e in v:
        r, s = _pack_rows(e, name)
        rows.extend(r)
        shapes.append(s)
    if any(s != shapes[0] for s in shapes[1:]):
        raise ValueError(name + " must be rectangular")
    return rows, [len(v)] + shapes[0]


def _rebuild(rows, shape):
    if len(shape) == 1:
        return list(rows[0])
    step = len(rows) // shape[0]
    return [_rebuild(rows[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def _log10(v):
    if math.isnan(v):
        return float("nan")
    if v == 0.0:
        return float("-inf")
    if v < 0.0:
        return float("nan")
    return math.log10(v)


def _div0(num, den):
    if den == 0:
        return float("nan")
    return num / den


# ----------------------------------------------------------------- biquad math
def _biquad_coeffs(sample_rate):
    w0t = 2.0 * math.pi * KW_TREBLE_FREQ / sample_rate
    alphat = math.sin(w0t) / 2 / KW_TREBLE_Q
    A = math.exp(KW_TREBLE_GAIN / 40 * math.log(10))
    temp1 = 2 * math.sqrt(A) * alphat
    temp2 = (A - 1) * math.cos(w0t)
    temp3 = (A + 1) * math.cos(w0t)
    treble_b = [A * ((A + 1) + temp2 + temp1),
                -2 * A * ((A - 1) + temp3),
                A * ((A + 1) + temp2 - temp1)]
    treble_a = [(A + 1) - temp2 + temp1,
                2 * ((A - 1) - temp3),
                (A + 1) - temp2 - temp1]
    w0h = 2.0 * math.pi * KW_HIGHPASS_FREQ / sample_rate
    alphah = math.sin(w0h) / 2.0 / KW_HIGHPASS_Q
    highpass_b = [(1 + math.cos(w0h)) / 2,
                  -1 - math.cos(w0h),
                  (1 + math.cos(w0h)) / 2]
    highpass_a = [1 + alphah, -2 * math.cos(w0h), 1 - alphah]
    return {
        "treble": _coeff_block(treble_a, treble_b, w0t, alphat,
                               A=A, t1=temp1, t2=temp2, t3=temp3),
        "highpass": _coeff_block(highpass_a, highpass_b, w0h, alphah),
    }


def _coeff_block(a, b, w0, alpha, A=None, t1=None, t2=None, t3=None):
    out = {
        "a": list(a), "b": list(b),
        "a_norm": [v / a[0] for v in a],
        "b_norm": [v / a[0] for v in b],
        "w0": w0, "alpha": alpha,
    }
    if A is not None:
        out.update({"A": A, "temp1": t1, "temp2": t2, "temp3": t3})
    return out


# ----------------------------------------------------------------- IIR filter
def _filter_row(row, a, b, clamp):
    """Official `_lfilter`: FIR on x, then IIR feedback on the *unclamped* y.

    `lfilter` clamps only the returned tensor, after `_lfilter` has finished,
    so the recursion state keeps the unclamped output values.
    """
    if a[0] == 0.0:
        raise ValueError("a_coeffs[0] must be nonzero")
    an = [v / a[0] for v in a]
    bn = [v / a[0] for v in b]
    out = []
    y1 = 0.0
    y2 = 0.0
    for n, x0 in enumerate(row):
        x1 = row[n - 1] if n >= 1 else 0.0
        x2 = row[n - 2] if n >= 2 else 0.0
        z = bn[0] * x0 + bn[1] * x1 + bn[2] * x2
        v = z - an[1] * y1 - an[2] * y2
        out.append(min(1.0, max(-1.0, v)) if clamp else v)
        y2 = y1
        y1 = v
    return out


def _kweight_rows(rows, coeffs):
    ta, tb = coeffs["treble"]["a"], coeffs["treble"]["b"]
    ha, hb = coeffs["highpass"]["a"], coeffs["highpass"]["b"]
    after_treble = [_filter_row(r, ta, tb, True) for r in rows]
    after_highpass = [_filter_row(r, ha, hb, True) for r in after_treble]
    return after_treble, after_highpass


# ----------------------------------------------------------------- gating ledger
def _gate_geometry(sample_rate, time_len):
    gate_samples = int(round(GATE_DURATION * sample_rate))
    step = int(round(gate_samples * (1 - OVERLAP)))
    if step <= 0:
        raise ValueError("step must be positive (sample_rate too small)")
    if time_len < gate_samples:
        raise ValueError("waveform shorter than gate_samples")
    n_blocks = (time_len - gate_samples) // step + 1
    return gate_samples, step, n_blocks


def _block_ledger(filtered, sample_rate):
    c = len(filtered)
    t = len(filtered[0])
    if c > 5:
        raise ValueError("Only up to 5 channels are supported.")
    if c < 1:
        raise ValueError("waveform must have at least one channel")
    gate_samples, step, n_blocks = _gate_geometry(sample_rate, t)
    g = list(CHANNEL_GAINS[:c])
    energy = []
    for row in filtered:
        blocks = []
        for b in range(n_blocks):
            start = b * step
            blocks.append(sum(row[start + k] * row[start + k]
                              for k in range(gate_samples)) / gate_samples)
        energy.append(blocks)
    energy_weighted = [sum(g[i] * energy[i][b] for i in range(c))
                       for b in range(n_blocks)]
    block_loudness = [KWEIGHT_BIAS + 10.0 * _log10(v) for v in energy_weighted]
    return {
        "gate_samples": gate_samples, "step": step, "n_blocks": n_blocks,
        "n_channels": c, "g": g, "energy": energy,
        "energy_weighted": energy_weighted, "block_loudness": block_loudness,
    }


def _dual_gate(led):
    c = led["n_channels"]
    nb = led["n_blocks"]
    energy = led["energy"]
    g = led["g"]
    bl = led["block_loudness"]
    gate_abs = [v > GAMMA_ABS for v in bl]
    sel_abs = [t for t in range(nb) if gate_abs[t]]
    energy_filtered_abs = [_div0(sum(energy[i][t] for t in sel_abs), len(sel_abs))
                           for i in range(c)]
    ew_abs = sum(g[i] * energy_filtered_abs[i] for i in range(c))
    gamma_rel = KWEIGHT_BIAS + 10.0 * _log10(ew_abs) - 10.0
    gate_rel = [gate_abs[t] and bl[t] > gamma_rel for t in range(nb)]
    sel_rel = [t for t in range(nb) if gate_rel[t]]
    energy_filtered_rel = [_div0(sum(energy[i][t] for t in sel_rel), len(sel_rel))
                           for i in range(c)]
    ew_rel = sum(g[i] * energy_filtered_rel[i] for i in range(c))
    lkfs = KWEIGHT_BIAS + 10.0 * _log10(ew_rel)
    led.update({
        "gate_abs": gate_abs, "n_gated_abs": len(sel_abs),
        "energy_filtered_abs": energy_filtered_abs, "ew_abs": ew_abs,
        "gamma_rel": gamma_rel, "gate_rel": gate_rel,
        "n_gated_rel": len(sel_rel),
        "energy_filtered_rel": energy_filtered_rel, "ew_rel": ew_rel,
        "lkfs": lkfs,
    })
    return led


# ----------------------------------------------------------------- modes
def _mode_coeffs(kw):
    sr = require_int(kw.get("sample_rate"), "sample_rate", 1)
    return {"sample_rate": sr, **_biquad_coeffs(sr)}


def _mode_lfilter(kw):
    a = require_numbers(kw.get("a_coeffs"), "a_coeffs")
    b = require_numbers(kw.get("b_coeffs"), "b_coeffs")
    if len(a) != 3:
        raise ValueError("a_coeffs must have exactly 3 numbers")
    if len(b) != 3:
        raise ValueError("b_coeffs must have exactly 3 numbers")
    clamp = require_bool(kw.get("clamp", True), "clamp")
    rows, shape = _pack_rows(kw.get("waveform"), "waveform")
    out = [_filter_row(r, a, b, clamp) for r in rows]
    return {"out": _rebuild(out, shape), "shape": shape,
            "clamp": clamp, "a_coeffs": list(a), "b_coeffs": list(b)}


def _mode_kweight(kw):
    sr = require_int(kw.get("sample_rate"), "sample_rate", 1)
    rows, shape = _pack_rows(kw.get("waveform"), "waveform")
    coeffs = _biquad_coeffs(sr)
    t, h = _kweight_rows(rows, coeffs)
    return {"after_treble": _rebuild(t, shape),
            "after_highpass": _rebuild(h, shape),
            "shape": shape, "sample_rate": sr, "coeffs": coeffs}


def _mode_blocks(kw):
    sr = require_int(kw.get("sample_rate"), "sample_rate", 1)
    filtered = require_matrix(kw.get("waveform"), "waveform")
    led = _block_ledger(filtered, sr)
    led["sample_rate"] = sr
    return _dual_gate(led)


def _mode_loudness(kw):
    sr = require_int(kw.get("sample_rate"), "sample_rate", 1)
    waveform = require_matrix(kw.get("waveform"), "waveform")
    coeffs = _biquad_coeffs(sr)
    t, h = _kweight_rows([list(r) for r in waveform], coeffs)
    led = _dual_gate(_block_ledger(h, sr))
    led.update({
        "sample_rate": sr,
        "after_treble": t, "filtered": h,
        "shape": [len(waveform), len(waveform[0])],
        "coeffs": coeffs,
    })
    return led


_DISPATCH = {
    "coeffs": _mode_coeffs,
    "lfilter": _mode_lfilter,
    "kweight": _mode_kweight,
    "blocks": _mode_blocks,
    "loudness": _mode_loudness,
}


def solve(mode, **kw):
    if mode not in _DISPATCH:
        raise ValueError("unknown mode: " + str(mode))
    return _DISPATCH[mode](kw)
