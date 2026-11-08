# -*- coding: utf-8 -*-
"""855 reference solution: torchaudio spectral post-processing ledger.

Pure Python (float64 by construction). Five modes:
  db / db_inverse / deltas / cmn / spectrogram
"""
import math


def _must(cond, msg):
    if not cond:
        raise ValueError(msg)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _shape(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return [0]
        return [len(v)] + _shape(v[0])
    return []


def _prod(vals):
    out = 1
    for v in vals:
        out *= v
    return out


def _flatten(v):
    if isinstance(v, (list, tuple)):
        out = []
        for item in v:
            out.extend(_flatten(item))
        return out
    return [v]


def _reshape(flat, shape):
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat[:shape[0]])
    sub = _prod(shape[1:])
    return [_reshape(flat[i * sub:(i + 1) * sub], shape[1:])
            for i in range(shape[0])]


def _rows(v, name):
    sh = _shape(v)
    _must(len(sh) >= 2, "%s must have at least two dimensions" % name)
    for s in sh:
        _must(s > 0, "%s dimensions must be non-empty" % name)
    flat = _flatten(v)
    _must(all(_is_num(x) for x in flat), "%s must contain only numbers" % name)
    _must(_prod(sh) == len(flat), "%s must be rectangular" % name)
    return sh, [float(x) for x in flat]


# ------------------------------------------------------------------ db
def db(x, multiplier, amin, db_multiplier, top_db=None):
    sh, flat = _rows(x, "x")
    _must(_is_num(multiplier), "multiplier must be a number")
    _must(_is_num(amin) and amin > 0.0, "amin must be a positive number")
    _must(_is_num(db_multiplier), "db_multiplier must be a number")
    if top_db is not None:
        _must(_is_num(top_db), "top_db must be a number or None")
    vals = [multiplier * math.log10(max(v, amin)) - multiplier * db_multiplier
            for v in flat]
    packed_channels = sh[-3] if len(sh) > 2 else 1
    num_batches = _prod(sh[:-3]) if len(sh) > 3 else 1
    freq, time = sh[-2], sh[-1]
    chunk = packed_channels * freq * time
    cutoffs = []
    if top_db is not None:
        for b in range(num_batches):
            seg = vals[b * chunk:(b + 1) * chunk]
            cutoffs.append(max(seg) - top_db)
        for b in range(num_batches):
            cutoff = cutoffs[b]
            for i in range(b * chunk, (b + 1) * chunk):
                vals[i] = max(vals[i], cutoff)
    return {
        "out": _reshape(vals, sh),
        "shape": sh,
        "packed_channels": packed_channels,
        "num_batches": num_batches,
        "cutoffs": cutoffs if top_db is not None else None,
    }


# ---------------------------------------------------------- db_inverse
def db_inverse(x, ref, power):
    sh, flat = _rows_any(x, "x")
    _must(_is_num(ref), "ref must be a number")
    _must(_is_num(power), "power must be a number")
    vals = [ref * ((10.0 ** (0.1 * v)) ** power) for v in flat]
    return {"out": _reshape(vals, sh), "shape": sh, "ref": ref, "power": power}


def _rows_any(v, name):
    sh = _shape(v)
    _must(len(sh) >= 1, "%s must have at least one dimension" % name)
    for s in sh:
        _must(s > 0, "%s dimensions must be non-empty" % name)
    flat = _flatten(v)
    _must(all(_is_num(x) for x in flat), "%s must contain only numbers" % name)
    _must(_prod(sh) == len(flat), "%s must be rectangular" % name)
    return sh, [float(x) for x in flat]


# --------------------------------------------------------------- deltas
def deltas(specgram, win_length=5, mode="replicate"):
    sh, flat = _rows(specgram, "specgram")
    time = sh[-1]
    _must(isinstance(win_length, int) and not isinstance(win_length, bool),
          "win_length must be an int")
    _must(win_length >= 3,
          "Window length should be greater than or equal to 3. Found win_length %d"
          % win_length)
    _must(mode in ("replicate", "constant"),
          "mode must be replicate or constant")
    n = (win_length - 1) // 2
    denom = n * (n + 1) * (2 * n + 1) / 3
    kernel = list(range(-n, n + 1))
    num_channels = len(flat) // time
    out_flat = []
    for c in range(num_channels):
        row = flat[c * time:(c + 1) * time]
        if mode == "replicate":
            padded = [row[0]] * n + row + [row[-1]] * n
        else:
            padded = [0.0] * n + row + [0.0] * n
        for t in range(time):
            acc = math.fsum(kernel[j] * padded[t + j]
                            for j in range(2 * n + 1))
            out_flat.append(acc / denom)
    return {
        "out": _reshape(out_flat, sh),
        "shape": sh,
        "n": n,
        "denom": denom,
        "kernel": kernel,
        "mode": mode,
        "num_channels": num_channels,
    }


# ------------------------------------------------------------------ cmn
def cmn_windows(num_frames, cmn_window, min_cmn_window, center):
    out = []
    for t in range(num_frames):
        if center:
            ws = t - cmn_window // 2
            we = ws + cmn_window
        else:
            ws = t - cmn_window
            we = t + 1
        if ws < 0:
            we -= ws
            ws = 0
        if not center and we > t:
            we = max(t + 1, min_cmn_window)
        if we > num_frames:
            ws -= we - num_frames
            we = num_frames
            if ws < 0:
                ws = 0
        out.append([ws, we])
    return out


def cmn(specgram, cmn_window=600, min_cmn_window=100,
        center=False, norm_vars=False):
    sh, flat = _rows(specgram, "specgram")
    num_frames, num_feats = sh[-2], sh[-1]
    _must(isinstance(cmn_window, int) and cmn_window > 0,
          "cmn_window must be a positive int")
    _must(isinstance(min_cmn_window, int) and min_cmn_window > 0,
          "min_cmn_window must be a positive int")
    _must(isinstance(center, bool), "center must be a bool")
    _must(isinstance(norm_vars, bool), "norm_vars must be a bool")
    num_channels = len(flat) // (num_frames * num_feats)
    wins = cmn_windows(num_frames, cmn_window, min_cmn_window, center)
    out_flat = []
    for c in range(num_channels):
        base = c * num_frames * num_feats
        cur_sum = [0.0] * num_feats
        cur_sumsq = [0.0] * num_feats
        last_ws = last_we = -1
        for t, (ws, we) in enumerate(wins):
            if last_ws == -1:
                # verbatim official slice upper bound: we - ws (ws is 0 here)
                for j in range(ws, we - ws):
                    for f in range(num_feats):
                        v = flat[base + j * num_feats + f]
                        cur_sum[f] += v
                        cur_sumsq[f] += v * v
            else:
                if ws > last_ws:
                    for f in range(num_feats):
                        v = flat[base + last_ws * num_feats + f]
                        cur_sum[f] -= v
                        cur_sumsq[f] -= v * v
                if we > last_we:
                    for f in range(num_feats):
                        v = flat[base + last_we * num_feats + f]
                        cur_sum[f] += v
                        cur_sumsq[f] += v * v
            window_frames = we - ws
            row = [flat[base + t * num_feats + f] - cur_sum[f] / window_frames
                   for f in range(num_feats)]
            if norm_vars:
                if window_frames == 1:
                    row = [0.0] * num_feats
                else:
                    for f in range(num_feats):
                        variance = cur_sumsq[f] / window_frames
                        variance -= (cur_sum[f] ** 2) / (window_frames ** 2)
                        row[f] *= variance ** -0.5
            out_flat.extend(row)
            last_ws, last_we = ws, we
    reshaped = _reshape(out_flat, sh[:-2] + [num_frames, num_feats])
    if len(sh) == 2 and num_frames == 1:
        reshaped = reshaped[0]
    return {
        "out": reshaped,
        "shape": _shape(reshaped),
        "windows": wins,
        "num_frames": num_frames,
        "num_feats": num_feats,
        "norm_vars": norm_vars,
    }


# ---------------------------------------------------------- spectrogram
def get_spec_norms(normalized):
    frame_length_norm = False
    window_norm = False
    if isinstance(normalized, str):
        if normalized not in ("frame_length", "window"):
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        elif normalized == "window":
            window_norm = True
    elif isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


def spectrogram(waveform, window, n_fft, hop_length, win_length,
                power=2.0, normalized=False, center=True,
                pad_mode="reflect", onesided=True, pad=0,
                return_complex=None):
    sh = _shape(waveform)
    _must(len(sh) in (1, 2), "waveform must be 1D or 2D")
    wsh = _shape(window)
    _must(len(wsh) == 1 and wsh[0] > 0, "window must be a non-empty 1D list")
    for s in sh:
        _must(s > 0, "waveform dimensions must be non-empty")
    _must(all(_is_num(x) for x in _flatten(waveform)),
          "waveform must contain only numbers")
    _must(all(_is_num(x) for x in _flatten(window)),
          "window must contain only numbers")
    _must(isinstance(n_fft, int) and n_fft > 0, "n_fft must be a positive int")
    _must(isinstance(hop_length, int) and hop_length > 0,
          "hop_length must be a positive int")
    _must(isinstance(win_length, int) and 0 < win_length <= n_fft,
          "win_length must be in (0, n_fft]")
    _must(win_length == wsh[0], "window length must equal win_length")
    _must(isinstance(pad, int) and pad >= 0, "pad must be a non-negative int")
    _must(isinstance(center, bool), "center must be a bool")
    _must(isinstance(onesided, bool), "onesided must be a bool")
    _must(pad_mode in ("reflect", "constant"),
          "pad_mode must be reflect or constant")
    _must(power is None or _is_num(power), "power must be a number or None")
    frame_norm, window_norm = get_spec_norms(normalized)

    warnings_out = []
    if return_complex is not None:
        warnings_out.append(
            "`return_complex` argument is now deprecated and is not effective.")

    win = [float(v) for v in window]
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        win = [0.0] * left + win + [0.0] * right

    window_scale = math.sqrt(sum(v * v for v in window))
    _must(window_scale > 0.0, "window energy must be positive")
    frame_scale = math.sqrt(float(n_fft))
    wavs = waveform if len(sh) == 2 else [waveform]
    outs = []
    bins = n_fft // 2 + 1 if onesided else n_fft
    num_frames = -1
    for wav in wavs:
        x = [float(v) for v in wav]
        if pad > 0:
            x = [0.0] * pad + x + [0.0] * pad
        if center:
            p = n_fft // 2
            if pad_mode == "reflect":
                _must(p > 0 and p < len(x),
                      "reflect pad requires n_fft//2 < waveform length")
                x = [x[p - i] for i in range(p)] + x + [x[-2 - i] for i in range(p)]
            else:
                x = [0.0] * p + x + [0.0] * p
        nf = 1 + (len(x) - n_fft) // hop_length if len(x) >= n_fft else 0
        _must(nf > 0, "waveform too short for n_fft")
        num_frames = nf
        re = [[0.0] * nf for _ in range(bins)]
        im = [[0.0] * nf for _ in range(bins)]
        for k in range(bins):
            for t in range(nf):
                seg = [x[t * hop_length + j] * win[j] for j in range(n_fft)]
                rv = math.fsum(seg[j] * math.cos(-2.0 * math.pi * k * j / n_fft)
                               for j in range(n_fft))
                iv = math.fsum(seg[j] * math.sin(-2.0 * math.pi * k * j / n_fft)
                               for j in range(n_fft))
                if frame_norm:
                    rv /= frame_scale
                    iv /= frame_scale
                if window_norm:
                    rv /= window_scale
                    iv /= window_scale
                re[k][t] = rv
                im[k][t] = iv
        if power is None:
            outs.append({"real": re, "imag": im})
        elif power == 1.0:
            outs.append([[math.hypot(re[k][t], im[k][t]) for t in range(nf)]
                         for k in range(bins)])
        else:
            outs.append([[(re[k][t] ** 2 + im[k][t] ** 2) ** (power / 2.0)
                          for t in range(nf)] for k in range(bins)])

    if len(sh) == 1:
        out = outs[0]
        shape = [bins, num_frames]
    else:
        if power is None:
            out = {"real": [o["real"] for o in outs],
                   "imag": [o["imag"] for o in outs]}
        else:
            out = outs
        shape = [sh[0], bins, num_frames]
    return {
        "out": out,
        "shape": shape,
        "bins": bins,
        "frames": num_frames,
        "frame_norm": frame_norm,
        "window_norm": window_norm,
        "normalized": normalized,
        "window_padded": win,
        "pad": pad,
        "center": center,
        "pad_mode": pad_mode,
        "onesided": onesided,
        "power": power,
        "warnings": warnings_out,
    }


def solve(mode, **kw):
    if mode == "db":
        return db(kw["x"], kw["multiplier"], kw["amin"],
                  kw["db_multiplier"], kw.get("top_db"))
    if mode == "db_inverse":
        return db_inverse(kw["x"], kw["ref"], kw["power"])
    if mode == "deltas":
        return deltas(kw["specgram"], kw.get("win_length", 5),
                      kw.get("pad_mode", "replicate"))
    if mode == "cmn":
        return cmn(kw["specgram"], kw.get("cmn_window", 600),
                   kw.get("min_cmn_window", 100), kw.get("center", False),
                   kw.get("norm_vars", False))
    if mode == "spectrogram":
        return spectrogram(kw["waveform"], kw["window"], kw["n_fft"],
                           kw["hop_length"], kw["win_length"],
                           kw.get("power", 2.0), kw.get("normalized", False),
                           kw.get("center", True), kw.get("pad_mode", "reflect"),
                           kw.get("onesided", True), kw.get("pad", 0),
                           kw.get("return_complex"))
    raise ValueError("unknown mode: %r" % (mode,))
