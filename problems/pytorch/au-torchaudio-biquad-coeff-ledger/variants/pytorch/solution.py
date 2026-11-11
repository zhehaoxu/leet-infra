#!/usr/bin/env python3
"""869 pure-Python reference solution -- no torch, no numpy, no torchaudio.

Recreates the torchaudio.functional.filtering biquad coefficient family as
eight independently judged modes.  `lfilter` is a spy in the official
environment (order 861 already judges the IIR recursion), so this file judges
only the coefficient derivation: w0/alpha/A, the RBJ/SoX cookbook, the bass
pre-normalization, the band exp/sqrt/noise branch, the de-emphasis and RIAA
preset tables, the coefficient ordering handed to lfilter and the exact
delegal-sample-rate errors.

IEEE semantics of torch float64 tensors are emulated explicitly: division by
zero yields +/-inf or nan (never an exception), sqrt of a negative yields nan,
sin/cos of a non-finite yields nan.  Non-finite results are canonicalized to
stable strings so every returned value is a plain JSON/Python literal.
"""
import math

MODES = ("coeffs", "call", "db2linear", "presets", "errors", "census",
         "grid", "biquad_order")

KINDS = ("allpass", "band", "bandpass", "bandreject", "bass", "equalizer",
         "lowpass", "deemph", "riaa")

WANT = (
    "_dB2Linear", "allpass_biquad", "band_biquad", "bandpass_biquad",
    "bandreject_biquad", "bass_biquad", "biquad", "deemph_biquad",
    "equalizer_biquad", "lowpass_biquad", "riaa_biquad",
)

LINES = {
    "_dB2Linear": (12, 13),
    "allpass_biquad": (70, 105),
    "band_biquad": (108, 158),
    "bandpass_biquad": (161, 204),
    "bandreject_biquad": (207, 241),
    "bass_biquad": (244, 292),
    "biquad": (295, 333),
    "deemph_biquad": (417, 462),
    "equalizer_biquad": (630, 669),
    "lowpass_biquad": (1102, 1132),
    "riaa_biquad": (1299, 1360),
}

CENSUS_SOURCE = {
    "want": sorted(list(WANT)),
    "line_start": 12,
    "line_end": 1360,
    "lines": {k: list(v) for k, v in LINES.items()},
    "bass_normalized": True,
    "allpass_raw": True,
    "band_a2_exp": True,
    "band_noise_mult": True,
    "bandpass_skirt": True,
    "equalizer_alpha_a": True,
    "equalizer_alpha_div_a": True,
    "deemph_two_rates": True,
    "riaa_four_rates": True,
    "riaa_1khz_gain": True,
    "biquad_a_first": True,
    "biquad_b_second": True,
    "db2linear": True,
    "no_rng": True,
}

DEEMPH_MSG = "Sample rate must be 44100 (audio-CD) or 48000 (DAT)"
RIAA_MSG = "Sample rate must be 44.1k, 48k, 88.2k, or 96k"

DEEMPH_PRESETS = {
    44100: (5283, 0.4845, -9.477),
    48000: (5356, 0.479, -9.62),
}

RIAA_PRESETS = {
    44100: ([-0.2014898, 0.9233820], [0.7083149, 0.9924091]),
    48000: ([-0.1766069, 0.9321590], [0.7396325, 0.9931330]),
    88200: ([-0.1168735, 0.9648312], [0.8590646, 0.9964002]),
    96000: ([-0.1141486, 0.9676817], [0.8699137, 0.9966946]),
}

ARGS = {
    "allpass": ("sample_rate", "central_freq", "Q"),
    "band": ("sample_rate", "central_freq", "Q", "noise"),
    "bandpass": ("sample_rate", "central_freq", "Q", "const_skirt_gain"),
    "bandreject": ("sample_rate", "central_freq", "Q"),
    "bass": ("sample_rate", "gain", "central_freq", "Q"),
    "equalizer": ("sample_rate", "center_freq", "gain", "Q"),
    "lowpass": ("sample_rate", "cutoff_freq", "Q"),
    "deemph": ("sample_rate",),
    "riaa": ("sample_rate",),
    "biquad": ("b0", "b1", "b2", "a0", "a1", "a2"),
}


# ---------------------------------------------------------------- torch IEEE
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
    return x


def _div(n, d):
    """IEEE division: torch tensors return inf/nan instead of raising."""
    n, d = float(n), float(d)
    if d == 0.0:
        if n == 0.0 or math.isnan(n):
            return float("nan")
        return math.copysign(float("inf"), (1 if n > 0 else -1) * math.copysign(1.0, d))
    return n / d


def _sin(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return math.sin(x)


def _cos(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return math.cos(x)


def _sqrt(x):
    x = float(x)
    if math.isnan(x) or x < 0.0:
        return float("nan")
    return math.sqrt(x)


def _exp(x):
    x = float(x)
    if math.isnan(x):
        return float("nan")
    if x == float("inf"):
        return float("inf")
    if x == -float("inf"):
        return 0.0
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def _err(e):
    return {"type": type(e).__name__, "msg": str(e)}


def _cfg(kw):
    c = {
        "kind": "lowpass",
        "sample_rate": 16000,
        "central_freq": 1000.0,
        "center_freq": 1000.0,
        "cutoff_freq": 1000.0,
        "gain": 6.0,
        "Q": 0.707,
        "noise": False,
        "const_skirt_gain": False,
        "waveform": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        "b0": 1.0, "b1": 0.5, "b2": 0.25,
        "a0": 1.0, "a1": 0.3, "a2": 0.1,
    }
    c.update(kw)
    return c


def _shape_of(x):
    """Nested-list shape, mirroring torch.tensor's ragged-input detection."""
    if not isinstance(x, list):
        return []

    def walk(v, dim):
        if not isinstance(v, list):
            return []
        if not v:
            return [0]
        first = walk(v[0], dim + 1)
        for item in v[1:]:
            sub = walk(item, dim + 1)
            if sub != first:
                raise ValueError(
                    "expected sequence of length %d at dim %d (got %d)"
                    % (first[0] if first else 0, dim + 1,
                       sub[0] if sub else 0))
        return [len(v)] + first

    return walk(x, 0)


# ------------------------------------------------------------- coefficient core
def _w0(sample_rate, freq):
    return _div(2 * math.pi * freq, sample_rate)


def _alpha(sample_rate, freq, q):
    return _div(_div(_sin(_w0(sample_rate, freq)), 2.0), q)


def _A(gain):
    return _exp(_div(gain, 40.0) * math.log(10))


def _core(kind, c):
    """Return raw (a, b) coefficient lists, exactly as handed to lfilter."""
    sr = c["sample_rate"]

    if kind == "biquad":
        return ([c["a0"], c["a1"], c["a2"]],
                [c["b0"], c["b1"], c["b2"]])

    if kind == "deemph":
        if sr not in DEEMPH_PRESETS:
            raise ValueError(DEEMPH_MSG)
        cf, width_slope, gain = DEEMPH_PRESETS[sr]
        w0 = _div(2 * math.pi * cf, sr)
        A = _exp(_div(gain, 40.0) * math.log(10))
        alpha = _div(_sin(w0), 2.0) * _sqrt(
            (A + _div(1.0, A)) * (_div(1.0, width_slope) - 1) + 2)
        t1 = 2 * _sqrt(A) * alpha
        t2 = (A - 1) * _cos(w0)
        t3 = (A + 1) * _cos(w0)
        b = [A * ((A + 1) + t2 + t1),
             -2 * A * ((A - 1) + t3),
             A * ((A + 1) + t2 - t1)]
        a = [(A + 1) - t2 + t1,
             2 * ((A - 1) - t3),
             (A + 1) - t2 - t1]
        return a, b

    if kind == "riaa":
        if sr not in RIAA_PRESETS:
            raise ValueError(RIAA_MSG)
        zeros, poles = RIAA_PRESETS[sr]
        b0, b1, b2 = 1.0, -(zeros[0] + zeros[1]), zeros[0] * zeros[1]
        a0, a1, a2 = 1.0, -(poles[0] + poles[1]), poles[0] * poles[1]
        y = _div(2 * math.pi * 1000, sr)
        b_re = b0 + b1 * _cos(-y) + b2 * _cos(-2 * y)
        a_re = a0 + a1 * _cos(-y) + a2 * _cos(-2 * y)
        b_im = b1 * _sin(-y) + b2 * _sin(-2 * y)
        a_im = a1 * _sin(-y) + a2 * _sin(-2 * y)
        g = _div(1.0, _sqrt(_div(b_re ** 2 + b_im ** 2,
                                 a_re ** 2 + a_im ** 2)))
        return [a0, a1, a2], [b0 * g, b1 * g, b2 * g]

    freq = c["cutoff_freq"] if kind == "lowpass" else (
        c["center_freq"] if kind == "equalizer" else c["central_freq"])
    q = c["Q"]
    w0 = _w0(sr, freq)
    cos_w0 = _cos(w0)
    alpha = _alpha(sr, freq, q)

    if kind == "lowpass":
        b0 = _div(1 - cos_w0, 2.0)
        b = [b0, 1 - cos_w0, b0]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
        return a, b

    if kind == "allpass":
        b = [1 - alpha, -2 * cos_w0, 1 + alpha]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
        return a, b

    if kind == "bandreject":
        b = [1.0, -2 * cos_w0, 1.0]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
        return a, b

    if kind == "bandpass":
        temp = _div(_sin(w0), 2.0) if c["const_skirt_gain"] else alpha
        b = [temp, 0.0, -temp]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
        return a, b

    if kind == "bass":
        A = _A(c["gain"])
        t1 = 2 * _sqrt(A) * alpha
        t2 = (A - 1) * cos_w0
        t3 = (A + 1) * cos_w0
        b_raw = [A * ((A + 1) - t2 + t1),
                 2 * A * ((A - 1) - t3),
                 A * ((A + 1) - t2 - t1)]
        a_raw = [(A + 1) + t2 + t1,
                 -2 * ((A - 1) + t3),
                 (A + 1) + t2 - t1]
        a0 = a_raw[0]
        return ([_div(v, a0) for v in a_raw],
                [_div(v, a0) for v in b_raw])

    if kind == "equalizer":
        A = _A(c["gain"])
        b = [1 + alpha * A, -2 * cos_w0, 1 - alpha * A]
        a = [1 + _div(alpha, A), -2 * cos_w0, 1 - _div(alpha, A)]
        return a, b

    if kind == "band":
        bw_Hz = _div(freq, q)
        a2 = _exp(_div(-2 * math.pi * bw_Hz, sr))
        a1 = _div(-4 * a2, 1 + a2) * cos_w0
        b0 = _sqrt(1 - _div(a1 * a1, 4 * a2)) * (1 - a2)
        if c["noise"]:
            mult = _div(
                _sqrt(_div(((1 + a2) * (1 + a2) - a1 * a1) * (1 - a2),
                           1 + a2)), b0)
            b0 = mult * b0
        return [1.0, a1, a2], [b0, 0.0, 0.0]

    raise ValueError("unknown kind: " + str(kind))


def _call(kind, c):
    a, b = _core(kind, c)
    return {
        "waveform_shape": _shape_of(c["waveform"]),
        "a_coeffs": a,
        "b_coeffs": b,
        "clamp": True,
        "batching": True,
        "dtype": "torch.float64",
    }


def _coeffs(kind, c):
    a, b = _core(kind, c)
    a0 = a[0]
    return {
        "kind": kind,
        "a": _clean(a),
        "b": _clean(b),
        "a_norm": _clean([_div(v, a0) for v in a]),
        "b_norm": _clean([_div(v, a0) for v in b]),
        "a0_raw": _clean(a0),
        "waveform_shape": _shape_of(c["waveform"]),
        "output_shape": _shape_of(c["waveform"]),
        "params": {k: c[k] for k in ARGS[kind] if k in c},
    }


def _presets():
    out = {}
    for sr in (44100, 48000):
        out["deemph_%d" % sr] = _coeffs("deemph", _cfg(dict(kind="deemph", sample_rate=sr)))
    for sr in (44100, 48000, 88200, 96000):
        out["riaa_%d" % sr] = _coeffs("riaa", _cfg(dict(kind="riaa", sample_rate=sr)))
    return out


def _errors():
    cases = {
        "deemph_bad_sr": dict(kind="deemph", sample_rate=22050),
        "riaa_bad_sr": dict(kind="riaa", sample_rate=22050),
        "q_zero": dict(kind="lowpass", Q=0.0),
        "sr_zero": dict(kind="lowpass", sample_rate=0),
        "negative_gain": dict(kind="bass", gain=-12.0),
        "band_noise": dict(kind="band", noise=True),
        "bandpass_skirt": dict(kind="bandpass", const_skirt_gain=True),
    }
    out = {}
    for name, extra in cases.items():
        c = _cfg(dict(extra))
        try:
            out[name] = {"type": "NO_ERROR", "value": _coeffs(c["kind"], c)}
        except Exception as e:                                   # noqa: BLE001
            out[name] = _err(e)
    return out


def _grid():
    allowed = {"deemph": (44100, 48000),
               "riaa": (44100, 48000, 88200, 96000)}
    out = {}
    for kind in KINDS:
        rows = []
        for sr in allowed.get(kind, (8000, 16000, 44100)):
            rows.append(_coeffs(kind, _cfg(dict(kind=kind, sample_rate=sr))))
        out[kind] = rows
    return out


def solve(mode, **kw):
    if mode == "census":
        return {k: ({kk: list(vv) for kk, vv in v.items()} if k == "lines"
                    else list(v) if isinstance(v, list) else v)
                for k, v in CENSUS_SOURCE.items()}
    if mode == "errors":
        return _errors()
    if mode == "presets":
        return _presets()
    if mode == "grid":
        return _grid()
    if mode == "db2linear":
        x = kw.get("x", 6.0)
        return math.exp(x * math.log(10) / 20.0)
    c = _cfg(kw)
    kind = c["kind"]
    if mode == "coeffs":
        return _coeffs(kind, c)
    if mode == "call":
        return _clean(_call(kind, c))
    if mode == "biquad_order":
        return _coeffs("biquad", dict(c, kind="biquad"))
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")


if __name__ == "__main__":
    print("SOLUTION 869 modes", len(MODES), "kinds", len(KINDS))
    print("  census keys:", len(solve("census")))
    print("  lowpass:", solve("coeffs", kind="lowpass")["a"])
    print("  bass a0_raw:", solve("coeffs", kind="bass")["a0_raw"])
    print("  call:", solve("call", kind="equalizer"))
    print("  db2linear(6):", solve("db2linear"))
    print("  presets:", sorted(solve("presets")))
    print("  errors:", {k: v["type"] for k, v in solve("errors").items()})
    print("  grid kinds:", sorted(solve("grid")))
