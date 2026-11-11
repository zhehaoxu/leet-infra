#!/usr/bin/env python3
"""864 reference solution: torchaudio speed / pitch-shift orchestration ledger.

Pure Python, no torch.  All DSP numerics (stft / istft / phase_vocoder /
sinc resample) are supplied as explicit inputs, as documented in notes.md;
this module reproduces only the official orchestration layer verbatim in
spirit and matches official864.comp() mode by mode.
"""
import math

MODES = (
    "rate", "pitch_rates", "stretch_plan", "phase_advance", "len_stretch",
    "pack_shape", "fix_shape", "pitch_shift", "speed_rates", "speed",
)


# --------------------------------------------------------------- validation
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _shape(x):
    sh = []
    while isinstance(x, (list, tuple)) and len(x) > 0:
        sh.append(len(x))
        x = x[0]
    return sh


def _pack(x, name):
    """Reject ragged / non-numeric nested lists with torch-style messages."""
    if not isinstance(x, (list, tuple)):
        raise ValueError("%s must be a nested list" % name)
    sh = _shape(x)
    if len(sh) < 1:
        raise ValueError("%s must be rectangular" % name)

    def walk(v, depth):
        if depth == len(sh) - 1:
            if not isinstance(v, (list, tuple)) or len(v) != sh[-1]:
                raise ValueError("%s must be rectangular" % name)
            for e in v:
                if not _is_num(e):
                    raise ValueError("%s must contain numbers" % name)
        else:
            if not isinstance(v, (list, tuple)) or len(v) != sh[depth]:
                raise ValueError("%s must be rectangular" % name)
            for e in v:
                walk(e, depth + 1)

    walk(x, 0)

    def conv(v):
        if isinstance(v, (list, tuple)):
            return [conv(e) for e in v]
        return float(v)

    return conv(x)


def _rows(x, rank):
    """Yield the last-axis rows in row-major order."""
    if rank == 1:
        yield list(x)
    elif rank == 2:
        for row in x:
            yield list(row)
    else:
        for part in x:
            for row in _rows(part, rank - 1):
                yield row


def _need_num(v, name):
    if not _is_num(v):
        raise TypeError("%s must be a number, got %s" % (name, type(v).__name__))
    return float(v)


def _need_int(v, name):
    if not isinstance(v, int) or isinstance(v, bool):
        raise TypeError("%s must be an int, got %s" % (name, type(v).__name__))
    return v


# --------------------------------------------------------------- small helpers
def _flatten(x):
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(_flatten(v))
        else:
            out.append(float(v))
    return out


def _reshape(flat, shape):
    total = 1
    for n in shape:
        total *= n
    if total != len(flat):
        raise ValueError("shape %s is invalid for %d elements"
                         % (_torchsize(shape), len(flat)))
    idx = 0

    def build(depth):
        nonlocal idx
        if depth == len(shape) - 1:
            row = flat[idx:idx + shape[-1]]
            idx += shape[-1]
            return row
        return [build(depth + 1) for _ in range(shape[depth])]

    return build(0)


def _torchsize(shape):
    return "[" + ", ".join(str(n) for n in shape) + "]"


def _linspace(start, end, n):
    """Mirror torch.linspace: uniform step, last element forced to `end`."""
    if n <= 0:
        return []
    if n == 1:
        return [float(start)]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n - 1)] + [float(end)]


def _fix(waveform, shape):
    """Official _fix_waveform_shape: trim/pad the last axis, then view()."""
    ori_len = shape[-1]
    rows = _rows(waveform, len(_shape(waveform)))
    fixed = []
    for row in rows:
        L = len(row)
        if L > ori_len:
            fixed.append(list(row[:ori_len]))
        elif L < ori_len:
            fixed.append(list(row) + [0.0] * (ori_len - L))
        else:
            fixed.append(list(row))
    flat = _flatten(fixed)
    target = list(shape[:-1]) + [ori_len]
    return _reshape(flat, target)


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _ceil_div_scaled(x, tgt, src, dtype):
    val = math.ceil((x * tgt) / src)
    if dtype in ("int64", "int32"):
        return int(val)
    if dtype == "float32":
        # ceil values are integral and exactly representable in float32 here
        return float(val)
    return float(val)


# --------------------------------------------------------------- config
def _cfg(kw):
    c = dict(
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=None,
        hop_length=None,
        window=None,
        sample_rate=16000,
        waveform=[[0.5 + 0.1 * i + 0.25 * j for i in range(10)] for j in range(2)],
        spec_f=None,
        stretched=[0.0] * 5,
        shifted=None,
        fix_waveform=None,
        fix_target_shape=None,
        speed_factor=1.5,
        speed_orig_freq=16000,
        speed_lengths=None,
        speed_lengths_dtype="float64",
        speed_waveform=None,
        speed_out=None,
    )
    c.update(kw)
    sh = _shape(_pack(c["waveform"], "waveform"))
    ori_len = sh[-1]
    if c["spec_f"] is None:
        c["spec_f"] = [[0.0] * 3 for _ in range(c["n_fft"] // 2 + 1)]
    if c["shifted"] is None:
        lead = sh[:-1]
        row = [0.75 + 0.05 * i for i in range(ori_len + 3)]
        c["shifted"] = row if not lead else _nest(lead, row)
    if c["fix_waveform"] is None:
        c["fix_waveform"] = c["shifted"]
    if c["fix_target_shape"] is None:
        c["fix_target_shape"] = sh
    if c["speed_waveform"] is None:
        c["speed_waveform"] = c["waveform"]
    if c["speed_out"] is None:
        c["speed_out"] = c["speed_waveform"]
    return c


def _nest(lead, row):
    out = row
    for n in reversed(lead):
        out = [list(out) for _ in range(n)]
    return out


# --------------------------------------------------------------- ledger
def _ledger(c):
    wf = _pack(c["waveform"], "waveform")
    shf = _shape(wf)
    ori_len = shf[-1]
    lead = shf[:-1]

    n_fft = _need_int(c["n_fft"], "n_fft")
    sample_rate = _need_num(c["sample_rate"], "sample_rate")
    n_steps = _need_num(c["n_steps"], "n_steps")
    bins = _need_num(c["bins_per_octave"], "bins_per_octave")

    hop = n_fft // 4 if c["hop_length"] is None else _need_int(c["hop_length"], "hop_length")
    win = n_fft if c["win_length"] is None else _need_int(c["win_length"], "win_length")
    if c["window"] is None:
        window_len = win
    else:
        window_len = len(_pack(c["window"], "window"))

    spec_f = _pack(c["spec_f"], "spec_f")
    sf = _shape(spec_f)
    if len(sf) < 2:
        raise ValueError("spec_f must have at least two dimensions")
    freq_bins = sf[-2]

    rate = 2.0 ** (-n_steps / bins)
    phase_advance = _linspace(0.0, math.pi * hop, freq_bins)
    len_stretch = int(round(ori_len / rate))
    pack_shape = [_prod(lead) if lead else 1, ori_len]
    pitch_rates = [int(sample_rate / rate), int(sample_rate)]

    shifted = _pack(c["shifted"], "shifted")
    pitch_output = _fix(shifted, shf)

    fix_waveform = _pack(c["fix_waveform"], "fix_waveform")
    fix_shape = c["fix_target_shape"]
    if not isinstance(fix_shape, (list, tuple)) or not fix_shape:
        raise ValueError("fix_target_shape must be a non-empty shape")
    fix_output = _fix(fix_waveform, list(fix_shape))

    factor = _need_num(c["speed_factor"], "speed_factor")
    orig_freq = _need_num(c["speed_orig_freq"], "speed_orig_freq")
    src_raw = int(factor * orig_freq)
    tgt_raw = int(orig_freq)
    g = math.gcd(src_raw, tgt_raw)
    src, tgt = src_raw // g, tgt_raw // g

    _pack(c["stretched"], "stretched")
    _pack(c["speed_waveform"], "speed_waveform")
    lengths = c["speed_lengths"]
    dtype = c["speed_lengths_dtype"]
    if lengths is None:
        out_lengths = None
    else:
        lens_shape = _shape(lengths)
        lens = _pack(lengths, "speed_lengths")
        vals = [_ceil_div_scaled(v, tgt, src, dtype) for v in _flatten(lens)]
        out_lengths = _reshape(vals, lens_shape)

    speed_output = _pack(c["speed_out"], "speed_out")

    return {
        "rate": rate,
        "phase_advance": phase_advance,
        "hop_length": hop,
        "win_length": win,
        "window_len": window_len,
        "freq_bins": freq_bins,
        "pack_shape": pack_shape,
        "len_stretch": len_stretch,
        "pitch_rates": pitch_rates,
        "pitch_output": pitch_output,
        "fix_output": fix_output,
        "speed_rates": [src_raw, tgt_raw, g, src, tgt],
        "speed_out_lengths": out_lengths,
        "speed_output": speed_output,
    }


# --------------------------------------------------------------- modes
def solve(mode, **kw):
    if mode not in MODES:
        raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                         + ", ".join(MODES) + ")")
    L = _ledger(_cfg(kw))
    if mode == "rate":
        return L["rate"]
    if mode == "pitch_rates":
        return L["pitch_rates"]
    if mode == "stretch_plan":
        return [L["hop_length"], L["win_length"], L["window_len"], L["freq_bins"]]
    if mode == "phase_advance":
        return L["phase_advance"]
    if mode == "len_stretch":
        return L["len_stretch"]
    if mode == "pack_shape":
        return L["pack_shape"]
    if mode == "fix_shape":
        return L["fix_output"]
    if mode == "pitch_shift":
        return L["pitch_output"]
    if mode == "speed_rates":
        return L["speed_rates"]
    if mode == "speed":
        return [L["speed_output"], L["speed_out_lengths"]]
    raise AssertionError("unreachable")
