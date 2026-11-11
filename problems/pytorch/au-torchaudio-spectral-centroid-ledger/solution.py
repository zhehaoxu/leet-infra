#!/usr/bin/env python3
"""868 pure-Python reference solution -- no torch, no numpy, no torchaudio.

Recreates torchaudio.functional.spectral_centroid (functional.py 1257-1299)
as eight independently judged modes.  The spectrogram is an explicit injected
input (the official environment uses a spy for it), so this file judges the
centroid's own semantics: the frequency grid, the column-vector broadcast,
the raw-specgram denominator, zero-denominator NaN and the exact kwargs.
"""
import math

MODES = ("centroid", "freqs", "call_kwargs", "ledger", "zero_frame",
         "shapes", "errors", "census")

CENSUS = {
    "want": "spectral_centroid",
    "line_start": 1257,
    "line_end": 1299,
    "power_1": True,
    "normalized_false": True,
    "sample_rate_floor": True,
    "n_fft_half_plus_one": True,
    "reshape_col": True,
    "freq_dim": True,
    "weighted_sum": True,
    "div_by_sum": True,
    "default_center": True,
    "default_pad_mode": True,
    "default_onesided": True,
}

RAGGED_MSG = "expected sequence of length 1 at dim 1 (got 0)"
ONE_D_MSG = ("Dimension out of range (expected to be in range of [-1, 0], "
             "but got -2)")


def _err(e):
    return {"type": type(e).__name__, "msg": str(e)}


def _cfg(kw):
    c = {
        "specgram": [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]],
        "sample_rate": 16000,
        "n_fft": 4,
        "pad": 0,
        "hop_length": 2,
        "win_length": 4,
        "waveform_shape": [1, 8],
        "window_shape": [4],
    }
    c.update(kw)
    return c


def _linspace(start, end, steps):
    """Torch's symmetric linspace: back half is computed from `end`."""
    if steps <= 0:
        raise RuntimeError("linspace: steps must be >= 1")
    if steps == 1:
        return [float(start)]
    step = (float(end) - float(start)) / (steps - 1)
    half = steps // 2
    out = []
    for i in range(steps):
        if i < half:
            out.append(float(start) + i * step)
        else:
            out.append(float(end) - (steps - 1 - i) * step)
    return out


def _freq_vec(c):
    return _linspace(0, c["sample_rate"] // 2, 1 + c["n_fft"] // 2)


def _shape_of(x):
    """Nested-list shape, detecting ragged input the way torch.tensor does."""
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


def _call(c):
    return {
        "waveform_shape": list(c["waveform_shape"]),
        "pad": c["pad"],
        "window_shape": list(c["window_shape"]),
        "n_fft": c["n_fft"],
        "hop_length": c["hop_length"],
        "win_length": c["win_length"],
        "power": 1.0,
        "normalized": False,
        "center": True,
        "pad_mode": "reflect",
        "onesided": True,
        "return_complex": None,
    }


def _div(n, d):
    if d == 0.0:
        if n == 0.0:
            return "nan"
        return math.inf if n > 0 else -math.inf
    return n / d


def _weighted(sg, fv):
    """Elementwise freqs(F,1) * specgram(...,F,T) broadcast."""
    ndim = len(_shape_of(sg))
    if ndim < 3:
        return [[fv[i] * sg[i][t] for t in range(len(sg[0]))]
                for i in range(len(fv))]
    return [_weighted(item, fv) for item in sg]


def _sum_freq_dim(x, ndim):
    """Equivalent of x.sum(dim=-2) for nested lists with >= 2 dims."""
    if ndim == 2:
        return [sum(row[t] for row in x) for t in range(len(x[0]))]
    return [_sum_freq_dim(item, ndim - 1) for item in x]


def _div_lists(a, b):
    if isinstance(a, list):
        return [_div_lists(x, y) for x, y in zip(a, b)]
    return _div(a, b)


def _core(c, sg=None):
    """Return (centroid, freqs_vec, weighted, num, den, shape, call)."""
    sg = c["specgram"] if sg is None else sg
    fv = _freq_vec(c)
    shape = _shape_of(sg)
    if len(shape) == 1:
        raise IndexError(ONE_D_MSG)
    if shape[-2] != len(fv):
        raise RuntimeError(
            "The size of tensor a (%d) must match the size of tensor b (%d) "
            "at non-singleton dimension 0" % (len(fv), shape[-2]))
    weighted = _weighted(sg, fv)
    num = _sum_freq_dim(weighted, len(shape))
    # NOTE: the official denominator sums the RAW specgram, not the broadcast
    # result -- this is why a 1-D specgram raises IndexError on dim=-2.
    den = _sum_freq_dim(sg, len(shape))
    out = _div_lists(num, den)
    out_shape = shape[:-2] + shape[-1:]
    return out, fv, weighted, num, den, out_shape, _call(c)


def _freqs(c):
    return [[v] for v in _freq_vec(c)]


def _centroid(c, sg=None):
    return _core(c, sg)[0]


def _ledger(c):
    out, fv, weighted, num, den, oshape, call = _core(c)
    return {
        "sample_rate_half": c["sample_rate"] // 2,
        "n_freqs": 1 + c["n_fft"] // 2,
        "freqs": [[v] for v in fv],
        "freqs_shape": [len(fv), 1],
        "specgram_shape": _shape_of(c["specgram"]),
        "weighted": weighted,
        "numerator": num,
        "denominator": den,
        "centroid": out,
        "output_shape": oshape,
        "call": call,
        "freq_dim": -2,
    }


def _zero_frame(c):
    z = dict(c)
    z["specgram"] = [[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [0.0, 3.0, 5.0]]
    out, _, _, _, _, oshape, _ = _core(z)
    return {"shape": oshape, "value": out}


def _shapes(c):
    layouts = {
        "2d": [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]],
        "3d": [[[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]],
               [[3.0, 1.0], [1.0, 4.0], [2.0, 1.0]]],
        "4d": [[[[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]]],
               [[[3.0, 1.0], [1.0, 4.0], [2.0, 1.0]]]],
    }
    out = {}
    for name, sg in layouts.items():
        sub = dict(c)
        sub["specgram"] = sg
        try:
            val, _, _, _, _, oshape, _ = _core(sub)
            out[name] = {"shape": oshape, "value": val}
        except Exception as e:                                   # noqa: BLE001
            out[name] = _err(e)
    return out


def _errors(kw):
    cases = {
        "freq_mismatch": dict(n_fft=4, specgram=[[1.0, 2.0], [3.0, 4.0]]),
        "odd_sample_rate": dict(sample_rate=16001),
        "negative_sample_rate": dict(sample_rate=-8000),
        "n_fft_zero": dict(n_fft=0, specgram=[[1.0, 2.0]]),
        "empty_time": dict(specgram=[[1.0], [2.0], [3.0], [], [], []]),
        "one_d_spec": dict(specgram=[1.0, 2.0, 3.0]),
    }
    out = {}
    for name, extra in cases.items():
        c = _cfg(dict(extra))
        try:
            val, _, _, _, _, oshape, _ = _core(c)
            out[name] = {"type": "NO_ERROR", "shape": oshape, "value": val}
        except Exception as e:                                   # noqa: BLE001
            out[name] = _err(e)
    return out


def solve(mode, **kw):
    if mode == "census":
        return dict(CENSUS)
    if mode == "errors":
        return _errors(kw)
    try:
        c = _cfg(kw)
        if mode == "freqs":
            return _freqs(c)
        if mode == "call_kwargs":
            return _call(c)
        if mode == "centroid":
            return _centroid(c)
        if mode == "ledger":
            return _ledger(c)
        if mode == "zero_frame":
            return _zero_frame(c)
        if mode == "shapes":
            return _shapes(c)
    except Exception as e:                                       # noqa: BLE001
        return _err(e)
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")


if __name__ == "__main__":
    print("SOLUTION 868", solve("census")["want"], "modes", len(MODES))
    print("  freqs:", solve("freqs"))
    print("  centroid:", solve("centroid"))
    print("  zero_frame:", solve("zero_frame"))
    print("  shapes:", {k: v["shape"] for k, v in solve("shapes").items()})
    print("  errors:", {k: v["type"] for k, v in solve("errors").items()})
