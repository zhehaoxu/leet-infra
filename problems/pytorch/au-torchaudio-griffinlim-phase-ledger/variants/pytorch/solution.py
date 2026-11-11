#!/usr/bin/env python3
"""860 reference solution: torchaudio Griffin-Lim phase-recovery ledger.

Pure Python (stdlib only).  Reimplements the four official functions
`inverse_spectrogram`, `_get_spec_norms`, `_get_complex_dtype` and `griffinlim`
together with the two spectral primitives they depend on (torch.stft /
torch.istft) as an explicit DFT + overlap-add ledger.

Judged input contract
---------------------
All inputs are JSON-friendly numbers / lists.  Complex matrices may be given
either as two real matrices (`real` and `imag`, shape [freq, frames]) or as a
single matrix of [re, im] pairs (`angles_init`).  Only two-dimensional
(freq, frames) spectrograms are in the judged domain; batch handling is out of
scope.  `rand_init` is not reproduced by an RNG: the caller supplies the exact
initial phase matrix, which is what the official `torch.rand(complex128)`
produced for a fixed seed.
"""
import cmath
import math

MODES = ("norms", "complex_dtype", "stft", "istft", "inverse", "griffinlim")

NOLA_EPS = 1e-11

_DTYPE_MAP = {
    "float64": "torch.complex128",
    "double": "torch.complex128",
    "float32": "torch.complex64",
    "float": "torch.complex64",
    "float16": "torch.complex32",
    "half": "torch.complex32",
}


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


def require_float(v, name, low=None):
    if not _is_num(v):
        raise ValueError(name + " must be a number")
    if low is not None and v < low:
        raise ValueError(name + " must be >= " + str(low))
    return float(v)


def require_bool(v, name):
    if not isinstance(v, bool):
        raise ValueError(name + " must be a bool")
    return v


def _geom(kw):
    n_fft = require_int(kw.get("n_fft", 8), "n_fft", 2)
    win_length = require_int(kw.get("win_length", n_fft), "win_length", 1)
    hop_length = require_int(kw.get("hop_length", n_fft // 4),
                             "hop_length", 1)
    if win_length > n_fft:
        raise ValueError("win_length must be <= n_fft")
    if hop_length > win_length:
        raise ValueError("hop_length must be <= win_length")
    window = require_numbers(kw.get("window", [1.0] * win_length), "window")
    if len(window) != win_length:
        raise ValueError("window length must equal win_length")
    return n_fft, hop_length, win_length, window


def _spec_norms(normalized):
    frame_length_norm, window_norm = False, False
    if isinstance(normalized, str):
        if normalized not in ("frame_length", "window"):
            raise ValueError("Invalid normalized parameter: " + normalized)
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


def _complex_dtype(real_dtype):
    if real_dtype in _DTYPE_MAP:
        return _DTYPE_MAP[real_dtype]
    raise ValueError("Unexpected dtype " + str(real_dtype))


# ----------------------------------------------------------------- primitives
def _pad_window(window, n_fft, win_length):
    """torch pads a short window to n_fft with zeros, left-biased."""
    if win_length == n_fft:
        return list(window)
    left = (n_fft - win_length) // 2
    return [0.0] * left + list(window) + [0.0] * (n_fft - win_length - left)


def _mirror(col, n_fft):
    """Rebuild the full Hermitian spectrum used by torch.fft.irfft."""
    return list(col) + [col[n_fft - k].conjugate()
                        for k in range(len(col), n_fft)]


def _irfft(col, n_fft):
    """Real inverse DFT; imaginary DC / Nyquist parts cancel automatically."""
    full = _mirror(col, n_fft)
    out = []
    for n in range(n_fft):
        acc = 0j
        for k, v in enumerate(full):
            acc += v * cmath.exp(2j * math.pi * k * n / n_fft)
        out.append((acc / n_fft).real)
    return out


def _rfft(seg, window, n_fft):
    """Forward DFT of one real, already-windowed frame."""
    cols = []
    for k in range(n_fft // 2 + 1):
        acc = 0j
        for n in range(n_fft):
            acc += seg[n] * window[n] * cmath.exp(
                -2j * math.pi * k * n / n_fft)
        cols.append(acc)
    return cols


def _istft(columns, n_fft, hop_length, win_length, window,
           center=True, normalized=False, length=None):
    """Overlap-add plus window-envelope division (torch.istft semantics)."""
    w = _pad_window(window, n_fft, win_length)
    n_frames = len(columns)
    total = (n_frames - 1) * hop_length + n_fft
    buf = [0.0] * total
    env = [0.0] * total
    for t, col in enumerate(columns):
        row = _irfft(col, n_fft)
        for n in range(n_fft):
            buf[t * hop_length + n] += row[n] * w[n]
            env[t * hop_length + n] += w[n] * w[n]
    if normalized:
        scale = math.sqrt(n_fft)
        buf = [v * scale for v in buf]
    start = n_fft // 2 if center else 0
    end = total - (n_fft // 2 if center else 0)
    size = length if length is not None else end - start
    out, den = [], []
    for i in range(start, start + size):
        out.append(buf[i] if i < total else 0.0)
        den.append(env[i] if i < total else 1.0)
    env_min = min(den) if den else 0.0
    if env_min < NOLA_EPS:
        raise ValueError("window overlap add min: " + str(env_min))
    return [a / b for a, b in zip(out, den)], env_min


def _stft(waveform, n_fft, hop_length, win_length, window,
          center=True, normalized=False):
    """Reflect-padded framed DFT (torch.stft semantics)."""
    w = _pad_window(window, n_fft, win_length)
    x = list(waveform)
    padded_length = len(x)
    if center:
        pad = n_fft // 2
        if len(x) <= pad:
            raise ValueError("reflect pad requires len(waveform) > n_fft // 2")
        x = x[1:pad + 1][::-1] + x + x[-pad - 1:-1][::-1]
        padded_length = len(x)
    if len(x) < n_fft:
        raise ValueError("waveform shorter than n_fft")
    n_frames = 1 + (len(x) - n_fft) // hop_length
    columns = []
    for t in range(n_frames):
        seg = x[t * hop_length:t * hop_length + n_fft]
        col = _rfft(seg, w, n_fft)
        if normalized:
            col = [v / math.sqrt(n_fft) for v in col]
        columns.append(col)
    return columns, n_frames, padded_length


def _split(columns):
    """Column-major complex matrix -> two [freq, frames] real matrices."""
    n_freq = len(columns[0])
    n_frames = len(columns)
    real = [[columns[t][k].real for t in range(n_frames)]
            for k in range(n_freq)]
    imag = [[columns[t][k].imag for t in range(n_frames)]
            for k in range(n_freq)]
    return real, imag


def _columns(real, imag):
    """Two [freq, frames] real matrices -> column-major complex matrix."""
    n_freq = len(real)
    n_frames = len(real[0])
    return [[complex(real[k][t], imag[k][t]) for k in range(n_freq)]
            for t in range(n_frames)]


def _angles(angles_init, n_freq, n_frames):
    """Initial phase matrix, explicitly supplied as [re, im] pairs."""
    if angles_init is None:
        return [[1.0 + 0.0j] * n_frames for _ in range(n_freq)]
    if not isinstance(angles_init, list) or len(angles_init) != n_freq:
        raise ValueError("angles_init must have shape [freq, frames]")
    out = []
    for row in angles_init:
        if not isinstance(row, list) or len(row) != n_frames:
            raise ValueError("angles_init must have shape [freq, frames]")
        col = []
        for pair in row:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                    not all(_is_num(v) for v in pair)):
                raise ValueError("angles_init entries must be [re, im] pairs")
            col.append(complex(pair[0], pair[1]))
        out.append(col)
    return out


# ----------------------------------------------------------------- modes
def _mode_norms(kw):
    frame_length_norm, window_norm = _spec_norms(kw.get("normalized", False))
    return {"frame_length_norm": frame_length_norm,
            "window_norm": window_norm}


def _mode_complex_dtype(kw):
    real_dtype = kw.get("real_dtype", "float64")
    if not isinstance(real_dtype, str):
        raise ValueError("real_dtype must be a str")
    return {"real_dtype": real_dtype,
            "complex_dtype": _complex_dtype(real_dtype)}


def _mode_stft(kw):
    waveform = require_numbers(kw.get("waveform"), "waveform")
    n_fft, hop_length, win_length, window = _geom(kw)
    center = require_bool(kw.get("center", True), "center")
    normalized = require_bool(kw.get("normalized", False), "normalized")
    columns, n_frames, padded_length = _stft(
        waveform, n_fft, hop_length, win_length, window, center, normalized)
    real, imag = _split(columns)
    return {"shape": [n_fft // 2 + 1, n_frames], "real": real, "imag": imag,
            "n_frames": n_frames, "padded_length": padded_length,
            "n_fft": n_fft, "hop_length": hop_length,
            "win_length": win_length, "center": center,
            "normalized": normalized}


def _mode_istft(kw):
    real = require_matrix(kw.get("real"), "real")
    imag = require_matrix(kw.get("imag"), "imag")
    n_fft, hop_length, win_length, window = _geom(kw)
    if len(real) != n_fft // 2 + 1 or len(imag) != len(real):
        raise ValueError("freq dimension must be n_fft // 2 + 1")
    if len(imag[0]) != len(real[0]):
        raise ValueError("real and imag shapes must agree")
    center = require_bool(kw.get("center", True), "center")
    normalized = require_bool(kw.get("normalized", False), "normalized")
    length = kw.get("length", None)
    if length is not None:
        length = require_int(length, "length", 0)
    out, env_min = _istft(_columns(real, imag), n_fft, hop_length,
                          win_length, window, center, normalized, length)
    return {"out": out, "shape": [len(out)], "target_length": len(out),
            "envelope_min": env_min,
            "window_l2": math.sqrt(sum(v * v for v in window))}


def _mode_inverse(kw):
    """inverse_spectrogram: norms -> optional window scaling -> ISTFT -> pad crop."""
    real = require_matrix(kw.get("real"), "real")
    imag = require_matrix(kw.get("imag"), "imag")
    n_fft, hop_length, win_length, window = _geom(kw)
    if len(real) != n_fft // 2 + 1 or len(imag) != len(real):
        raise ValueError("freq dimension must be n_fft // 2 + 1")
    if len(imag[0]) != len(real[0]):
        raise ValueError("real and imag shapes must agree")
    frame_length_norm, window_norm = _spec_norms(kw.get("normalized", False))
    center = require_bool(kw.get("center", True), "center")
    length = kw.get("length", None)
    if length is not None:
        length = require_int(length, "length", 0)
    pad = require_int(kw.get("pad", 0), "pad", 0)
    window_scale = math.sqrt(sum(v * v for v in window)) if window_norm else 1.0
    re = [[v * window_scale for v in row] for row in real]
    im = [[v * window_scale for v in row] for row in imag]
    istft_length = length + 2 * pad if length is not None else None
    out, env_min = _istft(_columns(re, im), n_fft, hop_length, win_length,
                          window, center, frame_length_norm, istft_length)
    if length is not None and pad > 0:
        out = out[pad:-pad]
    return {"out": out, "shape": [len(out)], "target_length": len(out),
            "frame_length_norm": frame_length_norm,
            "window_norm": window_norm, "window_scale": window_scale,
            "istft_length": istft_length, "pad": pad,
            "envelope_min": env_min}


def _mode_griffinlim(kw):
    specgram = require_matrix(kw.get("specgram"), "specgram")
    n_fft, hop_length, win_length, window = _geom(kw)
    n_freq, n_frames = len(specgram), len(specgram[0])
    if n_freq != n_fft // 2 + 1:
        raise ValueError("freq dimension must be n_fft // 2 + 1")
    power = require_float(kw.get("power", 1.0), "power")
    if power <= 0:
        raise ValueError("power must be > 0")
    n_iter = require_int(kw.get("n_iter", 2), "n_iter", 0)
    momentum = require_float(kw.get("momentum", 0.0), "momentum")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in range [0, 1). Found: "
                         + str(momentum))
    effective = momentum / (1 + momentum)
    length = kw.get("length", None)
    if length is not None:
        length = require_int(length, "length", 0)
    angles_init = kw.get("angles_init", None)
    mag = [[v ** (1 / power) for v in row] for row in specgram]
    angles = _angles(angles_init, n_freq, n_frames)
    tprev = None
    for _ in range(n_iter):
        cc = [[mag[k][t] * angles[k][t] for k in range(n_freq)]
              for t in range(n_frames)]
        inverse, _ = _istft(cc, n_fft, hop_length, win_length, window,
                            length=length)
        rebuilt, _, _ = _stft(inverse, n_fft, hop_length, win_length, window)
        if len(rebuilt) != n_frames:
            raise ValueError("rebuilt frame count does not match specgram")
        angles = [[c[k] for c in rebuilt] for k in range(n_freq)]
        if effective and tprev is not None:
            angles = [[angles[k][t] - tprev[k][t] * effective
                       for t in range(n_frames)] for k in range(n_freq)]
        angles = [[angles[k][t] / (abs(angles[k][t]) + 1e-16)
                   for t in range(n_frames)] for k in range(n_freq)]
        tprev = [[c[k] for c in rebuilt] for k in range(n_freq)]
    cc = [[mag[k][t] * angles[k][t] for k in range(n_freq)]
          for t in range(n_frames)]
    out, env_min = _istft(cc, n_fft, hop_length, win_length, window,
                          length=length)
    return {"out": out, "shape": [len(out)], "target_length": len(out),
            "momentum_effective": effective, "n_iter": n_iter,
            "power": power, "n_frames": n_frames,
            "envelope_min": env_min}


_DISPATCH = {
    "norms": _mode_norms,
    "complex_dtype": _mode_complex_dtype,
    "stft": _mode_stft,
    "istft": _mode_istft,
    "inverse": _mode_inverse,
    "griffinlim": _mode_griffinlim,
}


def solve(mode, **kw):
    if mode not in _DISPATCH:
        raise ValueError("unknown mode: " + str(mode))
    return _DISPATCH[mode](kw)
