"""866 reference solution: torchaudio mel / linear fbank ledger, pure Python.

The solve() entry point mirrors official866.py::comp() mode-for-mode:

    hz_to_mel | mel_to_hz | mel_grid | hz_grid | triangular
    mel_fbanks | linear_fbanks | fbank_ledger | errors | census

No torch, no numpy, no audio IO.  torch.linspace on float64 is reproduced
bit-for-bit with PyTorch's symmetric formula:

    step = (end - start) / (steps - 1)
    i < steps // 2  -> start + i * step
    otherwise       -> end - (steps - 1 - i) * step

and torch's pow / exp / log10 agree exactly with Python's math on float64
(measured in precision866.py).
"""

import math

MODES = (
    "hz_to_mel",
    "mel_to_hz",
    "mel_grid",
    "hz_grid",
    "triangular",
    "mel_fbanks",
    "linear_fbanks",
    "fbank_ledger",
    "errors",
    "census",
)

WANT = ["_hz_to_mel", "_mel_to_hz", "_create_triangular_filterbank",
        "melscale_fbanks", "linear_fbanks"]

F_SP = 200.0 / 3
MIN_LOG_HZ = 1000.0
MIN_LOG_MEL = MIN_LOG_HZ / F_SP
LOGSTEP = math.log(6.4) / 27.0

BAD_SCALE = 'mel_scale should be one of "htk" or "slaney".'
BAD_NORM = 'norm must be one of None or "slaney"'


# ------------------------------------------------------------------ helpers
def _err(kind, msg):
    return {"type": kind, "msg": msg}


def _linspace(start, end, steps):
    """torch.linspace(float64), including the symmetric back half."""
    start, end = float(start), float(end)
    steps = int(steps)
    if steps <= 0:
        return []
    if steps == 1:
        return [start]
    step = (end - start) / (steps - 1)
    return [start + i * step if i < steps // 2
            else end - (steps - 1 - i) * step
            for i in range(steps)]


def _valid_scale(mel_scale):
    return mel_scale in ("slaney", "htk")


def _hz_to_mel(freq, mel_scale):
    if not _valid_scale(mel_scale):
        raise ValueError(BAD_SCALE)
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (float(freq) / 700.0))
    # Slaney: linear below 1000 Hz, log at or above it (scalar if, not a mask)
    mels = (float(freq) - 0.0) / F_SP
    if freq >= MIN_LOG_HZ:
        mels = MIN_LOG_MEL + math.log(freq / MIN_LOG_HZ) / LOGSTEP
    return mels


def _mel_to_hz(mel, mel_scale):
    if not _valid_scale(mel_scale):
        raise ValueError(BAD_SCALE)
    mel = float(mel)
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    # The official body does this with a TENSOR boolean mask + in-place assign.
    # Scalar-equivalent: NaN also stays on the linear branch.
    if mel >= MIN_LOG_MEL:
        return MIN_LOG_HZ * math.exp(LOGSTEP * (mel - MIN_LOG_MEL))
    return 0.0 + F_SP * mel


def _hz_to_mel_grid(freqs, mel_scale):
    if not _valid_scale(mel_scale):
        return None, _err("ValueError", BAD_SCALE)
    return [_hz_to_mel(f, mel_scale) for f in freqs], None


def _mel_to_hz_grid(mels, mel_scale):
    if not _valid_scale(mel_scale):
        return None, _err("ValueError", BAD_SCALE)
    return [_mel_to_hz(m, mel_scale) for m in mels], None


def _triangular(all_freqs, f_pts):
    """Slope form with SHIFTED denominators f_diff[:-1] / f_diff[1:]."""
    all_freqs = [float(f) for f in all_freqs]
    f_pts = [float(f) for f in f_pts]
    f_diff = [f_pts[i + 1] - f_pts[i] for i in range(len(f_pts) - 1)]
    n_filter = len(f_pts) - 2
    out = []
    for f in all_freqs:
        row = []
        for j in range(n_filter):
            down = -(f_pts[j] - f) / f_diff[j]
            up = (f_pts[j + 2] - f) / f_diff[j + 1]
            row.append(max(0.0, min(down, up)))
        out.append(row)
    return out


# ------------------------------------------------------------------- config
def _cfg(kw):
    c = {
        "n_freqs": 17,
        "f_min": 0.0,
        "f_max": 8000.0,
        "n_mels": 6,
        "sample_rate": 16000,
        "norm": None,
        "mel_scale": "htk",
        "all_freqs": [0.0, 1.0, 2.0, 3.0, 4.0],
        "f_pts": [0.0, 2.0, 4.0, 6.0],
        "freq": 1000.0,
        "mel": 15.0,
        "freqs": [0.0, 100.0, 1000.0],
        "mels": [0.0, 15.0, 30.0],
    }
    c.update(kw)
    return c


# ------------------------------------------------------------- fbank ledger
def _fbank(c, kind):
    """Return (matrix, error, warnings, ledger), mirroring official866._fbank."""
    if kind == "mel" and c["norm"] is not None and c["norm"] != "slaney":
        return None, _err("ValueError", BAD_NORM), [], None
    if kind == "mel" and not _valid_scale(c["mel_scale"]):
        return None, _err("ValueError", BAD_SCALE), [], None

    n_freqs, n_mels = c["n_freqs"], c["n_mels"]
    all_freqs = _linspace(0, c["sample_rate"] // 2, n_freqs)

    if kind == "mel":
        m_min = _hz_to_mel(c["f_min"], c["mel_scale"])
        m_max = _hz_to_mel(c["f_max"], c["mel_scale"])
        m_pts = _linspace(m_min, m_max, n_mels + 2)
        f_pts = [_mel_to_hz(m, c["mel_scale"]) for m in m_pts]
    else:
        m_min = m_max = None
        m_pts = None
        f_pts = _linspace(c["f_min"], c["f_max"], n_mels + 2)

    f_diff = [f_pts[i + 1] - f_pts[i] for i in range(len(f_pts) - 1)]
    fb = _triangular(all_freqs, f_pts)

    enorm = None
    if kind == "mel" and c["norm"] == "slaney":
        enorm = [2.0 / (f_pts[j + 2] - f_pts[j]) for j in range(n_mels)]
        fb = [[row[j] * enorm[j] for j in range(n_mels)] for row in fb]

    column_max = [max((row[j] for row in fb), default=0.0)
                  for j in range(n_mels)]
    zero_cols = [j for j in range(n_mels) if column_max[j] == 0.0]
    msgs = []
    if kind == "mel" and zero_cols:
        msgs.append(
            "At least one mel filterbank has all zero values. "
            "The value for `n_mels` (%d) may be set too high. "
            "Or, the value for `n_freqs` (%d) may be set too low."
            % (n_mels, n_freqs))

    ledger = {
        "all_freqs": all_freqs,
        "m_min": m_min,
        "m_max": m_max,
        "m_pts": m_pts,
        "f_pts": f_pts,
        "f_diff": f_diff,
        "enorm": enorm,
        "zero_columns": zero_cols,
        "warnings": msgs,
        "shape": [n_freqs, n_mels],
        "column_max": column_max,
    }
    return fb, None, msgs, ledger


# ------------------------------------------------------------------- modes
def _errors():
    cases = {
        "mel_scale_bad": ("hz_to_mel", dict(freq=1000.0, mel_scale="bogus")),
        "mel_scale_bad_rev": ("mel_to_hz", dict(mels=[0.0, 15.0],
                                                mel_scale="bogus")),
        "norm_bad": ("mel_fbanks", dict(norm="htk")),
        "norm_empty": ("mel_fbanks", dict(norm="")),
        "n_mels_zero": ("mel_fbanks", dict(n_mels=0)),
        "f_min_negative": ("mel_fbanks", dict(f_min=-1.0)),
        "sample_rate_odd": ("mel_fbanks", dict(sample_rate=16001)),
    }
    out = {}
    for name, (mode, kw) in cases.items():
        try:
            r = solve(mode, **kw)
            out[name] = "NO_ERROR" if not (
                isinstance(r, dict) and "type" in r) else \
                (r["type"] + ": " + r["msg"])
        except Exception as e:                                   # noqa: BLE001
            out[name] = type(e).__name__ + ": " + str(e)
    return out


def _census():
    return {
        "want": list(WANT),
        "hz_htk_log10": True,
        "hz_htk_2595": True,
        "slaney_f_sp": True,
        "slaney_min_log_hz": True,
        "slaney_logstep": True,
        "hz_scalar_if": True,
        "hz_to_hz_tensor_mask": True,
        "hz_to_hz_inplace": True,
        "hz_to_hz_torch_exp": True,
        "tri_slope_form": True,
        "tri_max_min": True,
        "tri_fdiff_shift": True,
        "mel_all_freqs_floor_div": True,
        "mel_n_plus_two": True,
        "mel_enorm": True,
        "mel_warn": True,
        "mel_norm_independent": True,
        "lin_hz_linspace": True,
        "lin_no_norm": True,
        "lin_no_warn": True,
        "f_sp": F_SP,
        "min_log_mel": MIN_LOG_MEL,
        "logstep": LOGSTEP,
    }


def _safe(fn):
    """Official comp() captures every exception into an error dict."""
    try:
        return fn()
    except Exception as e:                                   # noqa: BLE001
        return _err(type(e).__name__, str(e))


def _first(pair):
    vals, err = pair
    return err if err else vals[0]


def _vals(pair):
    vals, err = pair
    return err if err else vals


def _matrix(res):
    out, err, _, _ = res
    return err if err else out


def _ledger(res):
    out, err, _, ledger = res
    return err if err else ledger


def solve(mode, **kw):
    if mode == "errors":
        return _errors()
    if mode == "census":
        return _census()
    c = _cfg(kw)
    ms = c["mel_scale"]

    if mode == "hz_to_mel":
        return _safe(lambda: _first(_hz_to_mel_grid([c["freq"]], ms)))
    if mode == "mel_to_hz":
        return _safe(lambda: _first(_mel_to_hz_grid([c["mel"]], ms)))
    if mode == "mel_grid":
        return _safe(lambda: _vals(_hz_to_mel_grid(c["freqs"], ms)))
    if mode == "hz_grid":
        return _safe(lambda: _vals(_mel_to_hz_grid(c["mels"], ms)))
    if mode == "triangular":
        return _safe(lambda: _triangular(c["all_freqs"], c["f_pts"]))
    if mode == "mel_fbanks":
        return _safe(lambda: _matrix(_fbank(c, "mel")))
    if mode == "linear_fbanks":
        return _safe(lambda: _matrix(_fbank(c, "linear")))
    if mode == "fbank_ledger":
        return _safe(lambda: _ledger(_fbank(c, c.get("kind", "mel"))))
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
