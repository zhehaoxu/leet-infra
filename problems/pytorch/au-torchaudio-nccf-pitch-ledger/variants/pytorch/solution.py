#!/usr/bin/env python3
"""858 reference solution: torchaudio NCCF pitch-detection ledger.

Pure Python, no torch, no numpy, no global RNG.  Five independent modes
reproduce the official ``torchaudio.functional`` mechanisms:

    nccf          : _compute_nccf (framing, padding, fourth-power normalizer)
    combine_max   : _combine_max (threshold mask selection)
    find_max      : _find_max_per_frame (half-window preference, two offsets)
    median_smooth : _median_smoothing (left replicate, unfold, lower median)
    pitch         : detect_pitch_frequency (full pipeline + frequency ledger)
"""

import math

EPSILON = 1e-9


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
    vals = flatten(x)
    if not vals or not all(is_number(v) for v in vals):
        raise ValueError("%s must be a non-empty numeric array" % what)
    if not isinstance(x, list):
        raise ValueError("%s must be an array" % what)


def require_int(v, what, minimum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("%s must be an integer" % what)
    if minimum is not None and v < minimum:
        raise ValueError("%s must be at least %d" % (what, minimum))


def require_positive(v, what):
    if not is_number(v) or v <= 0:
        raise ValueError("%s must be a positive number" % what)


def pack(x, what):
    """Flatten all leading dimensions into rows of equal last-dim length."""
    s = shape_of(x)
    if len(s) < 1 or s[-1] == 0:
        raise ValueError("%s must be a non-empty numeric array" % what)
    lead, last = s[:-1], s[-1]
    vals = flatten(x)
    n = max(1, numel(lead))
    rows = [vals[i * last:(i + 1) * last] for i in range(n)]
    return rows, lead


def lower_median(vals):
    """torch.median returns the lower median for an even-length window."""
    s = sorted(vals)
    return s[(len(s) - 1) // 2]


def argmax(vals):
    """First occurrence of the maximum, matching torch.max."""
    best_i, best_v = 0, vals[0]
    for i, v in enumerate(vals):
        if v > best_v:
            best_i, best_v = i, v
    return best_i, best_v


# ---------------------------------------------------------------------------
# mode 1: nccf
# ---------------------------------------------------------------------------
def nccf_meta(length, sample_rate, frame_time, freq_low):
    lags = int(math.ceil(sample_rate / freq_low))
    frame_size = int(math.ceil(sample_rate * frame_time))
    num_frames = int(math.ceil(length / frame_size))
    pad = lags + num_frames * frame_size - length
    return {"lags": lags, "frame_size": frame_size,
            "num_frames": num_frames, "pad": pad,
            "padded_length": length + pad}


def _nccf(kw):
    wf = kw["waveform"]
    require_numbers(wf, "waveform")
    sr = kw["sample_rate"]
    require_int(sr, "sample_rate", 1)
    ft = kw["frame_time"]
    require_positive(ft, "frame_time")
    fl = kw["freq_low"]
    require_int(fl, "freq_low", 1)
    rows, lead = pack(wf, "waveform")
    length = len(rows[0])
    meta = nccf_meta(length, sr, ft, fl)
    lags, F, n = meta["lags"], meta["frame_size"], meta["num_frames"]
    pad = meta["pad"]
    out_rows = []
    for row in rows:
        padded = list(row) + [0.0] * pad
        mat = []
        for i in range(n):
            line = []
            for lag in range(1, lags + 1):
                s1 = padded[i * F:i * F + F]
                s2 = padded[lag + i * F:lag + i * F + F]
                dot = sum(a * b for a, b in zip(s1, s2))
                nr1 = math.sqrt(sum(a * a for a in s1))
                nr2 = math.sqrt(sum(b * b for b in s2))
                line.append(dot / (EPSILON + nr1) ** 2 / (EPSILON + nr2) ** 2)
            mat.append(line)
        out_rows.append(mat)
    shape = lead + [n, lags]
    result = unflatten(flatten(out_rows), shape)
    return {"nccf": result, "shape": shape, "waveform_length": length, **meta}


# ---------------------------------------------------------------------------
# mode 2: combine_max
# ---------------------------------------------------------------------------
def _combine_max(kw):
    av, ai = kw["a_values"], kw["a_indices"]
    bv, bi = kw["b_values"], kw["b_indices"]
    for name, v in (("a_values", av), ("a_indices", ai),
                    ("b_values", bv), ("b_indices", bi)):
        require_numbers(v, name)
    thresh = kw.get("thresh", 0.99)
    require_positive(thresh, "thresh")
    va, ia, vb, ib = flatten(av), flatten(ai), flatten(bv), flatten(bi)
    if not (len(va) == len(ia) == len(vb) == len(ib)):
        raise ValueError("a and b must have the same length")
    mask = [a > thresh * b for a, b in zip(va, vb)]
    values = [a if m else b for m, a, b in zip(mask, va, vb)]
    indices = [a if m else b for m, a, b in zip(mask, ia, ib)]
    return {"values": values, "indices": indices, "mask": mask,
            "thresh": thresh}


# ---------------------------------------------------------------------------
# mode 3: find_max
# ---------------------------------------------------------------------------
def _find_max(kw):
    nc = kw["nccf"]
    require_numbers(nc, "nccf")
    if len(shape_of(nc)) < 2:
        raise ValueError("nccf must have at least 2 dimensions")
    sr = kw["sample_rate"]
    require_int(sr, "sample_rate", 1)
    fh = kw["freq_high"]
    require_int(fh, "freq_high", 1)
    rows, lead = pack(nc, "nccf")
    lag_min = int(math.ceil(sr / fh))
    half_size = len(rows[0]) // 2
    if half_size <= lag_min:
        raise ValueError("nccf last dimension is too short for freq_high")
    # _find_max_per_frame calls _combine_max(half, best) with its default
    # 0.99 threshold; the threshold is NOT a parameter of this function.
    idx_out, bv, bi, hv, hi = [], [], [], [], []
    for row in rows:
        bi_i, bv_v = argmax(row[lag_min:])
        hi_i, hv_v = argmax(row[lag_min:half_size])
        bv.append(bv_v)
        bi.append(bi_i)
        hv.append(hv_v)
        hi.append(hi_i)
        take_half = hv_v > 0.99 * bv_v
        idx_out.append((hi_i if take_half else bi_i) + lag_min + 1)
    shape = lead
    return {
        "indices": unflatten(idx_out, lead) if lead else idx_out,
        "lag_min": lag_min,
        "half_size": half_size,
        "best_values": unflatten(bv, lead) if lead else bv,
        "best_indices": unflatten(bi, lead) if lead else bi,
        "half_values": unflatten(hv, lead) if lead else hv,
        "half_indices": unflatten(hi, lead) if lead else hi,
        "shape": shape,
    }


# ---------------------------------------------------------------------------
# mode 4: median_smooth
# ---------------------------------------------------------------------------
def _median_smooth(kw):
    idx = kw["indices"]
    require_numbers(idx, "indices")
    w = kw["win_length"]
    require_int(w, "win_length", 3)
    rows, lead = pack(idx, "indices")
    p = (w - 1) // 2
    out_rows, padded_rows, roll_rows = [], [], []
    for row in rows:
        n = len(row)
        if n + p < w:
            raise ValueError("indices length is too short for win_length")
        padded = [row[0]] * p + list(row)
        roll = [padded[i:i + w] for i in range(n + p - w + 1)]
        out_rows.append([lower_median(win) for win in roll])
        padded_rows.append(padded)
        roll_rows.append(roll)
    tail = len(out_rows[0])
    shape = lead + [tail]
    return {
        "smoothed": unflatten(flatten(out_rows), shape),
        "padded": unflatten(flatten(padded_rows), lead + [len(padded_rows[0])]),
        "roll": unflatten(flatten(roll_rows), lead + [tail, w]),
        "pad_length": p,
        "shape": shape,
    }


# ---------------------------------------------------------------------------
# mode 5: pitch
# ---------------------------------------------------------------------------
def _pitch(kw):
    wf = kw["waveform"]
    require_numbers(wf, "waveform")
    sr = kw["sample_rate"]
    require_int(sr, "sample_rate", 1)
    ft = kw.get("frame_time", 0.01)
    require_positive(ft, "frame_time")
    w = kw.get("win_length", 30)
    require_int(w, "win_length", 3)
    fl = kw.get("freq_low", 85)
    require_int(fl, "freq_low", 1)
    fh = kw.get("freq_high", 3400)
    require_int(fh, "freq_high", 1)
    n = _nccf({"waveform": wf, "sample_rate": sr,
               "frame_time": ft, "freq_low": fl})
    f = _find_max({"nccf": n["nccf"], "sample_rate": sr, "freq_high": fh})
    ms = _median_smooth({"indices": f["indices"], "win_length": w})
    smoothed = ms["smoothed"]
    freq = [sr / (EPSILON + float(v)) for v in flatten(smoothed)]
    lead = shape_of(wf)[:-1]
    out_shape = lead + [ms["shape"][-1]]
    freq = unflatten(freq, out_shape)
    meta = {k: n[k] for k in ("lags", "frame_size", "num_frames", "pad",
                              "padded_length")}
    return {"freq": freq, "shape": out_shape, "nccf": n["nccf"],
            "indices": f["indices"], "smoothed": smoothed, "meta": meta}


MODES = ("nccf", "combine_max", "find_max", "median_smooth", "pitch")


def solve(mode, **kw):
    if mode == "nccf":
        return _nccf(kw)
    if mode == "combine_max":
        return _combine_max(kw)
    if mode == "find_max":
        return _find_max(kw)
    if mode == "median_smooth":
        return _median_smooth(kw)
    if mode == "pitch":
        return _pitch(kw)
    raise ValueError("unknown mode: %r" % (mode,))
