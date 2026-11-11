#!/usr/bin/env python3
"""870 pure-Python reference solution: torchaudio dither quantization ledger.

Only `math` is used. No torch / numpy / torchaudio, no RNG: the random draws
of the official `_apply_probability_distribution` are explicit injected inputs.
"""
import math

MODES = ("quantize", "bartlett", "round", "noise_shaping", "branches",
         "shapes", "errors", "census")
DENSITIES = ("TPDF", "RPDF", "GPDF")

UP = 2 ** 15 - 2          # number_of_bits = 16 -> 2**(16-1)-2
DOWN = 2 ** 15            # 2**(16-1)
N_GAUSS = 6               # num_rand_variables

EMPTY_MSG = ("cannot reshape tensor of 0 elements into shape [-1, 0] because "
             "the unspecified dimension size -1 can be any value and is ambiguous")

CENSUS_SOURCE = {
    "source": "pytorch/audio main::src/torchaudio/functional/filtering.py",
    "source_lines": 1702,
    "n_top_level": 30,
    "top_level": [
        [12, "_dB2Linear"], [16, "_generate_wave_table"], [70, "allpass_biquad"],
        [108, "band_biquad"], [161, "bandpass_biquad"], [207, "bandreject_biquad"],
        [244, "bass_biquad"], [295, "biquad"], [336, "contrast"], [370, "dcshift"],
        [417, "deemph_biquad"], [465, "_add_noise_shaping"],
        [489, "_apply_probability_distribution"], [598, "dither"],
        [630, "equalizer_biquad"], [672, "filtfilt"], [713, "flanger"],
        [871, "gain"], [893, "highpass_biquad"],
        [926, "_lfilter_core_generic_loop"], [1027, "_lfilter"], [1032, "lfilter"],
        [1102, "lowpass_biquad"], [1135, "_overdrive_core_loop_generic"],
        [1150, "overdrive"], [1206, "phaser"], [1299, "riaa_biquad"],
        [1363, "treble_biquad"], [1414, "_measure"], [1485, "vad"],
    ],
    "funcs": {
        "_add_noise_shaping": [465, 486],
        "_apply_probability_distribution": [489, 595],
        "dither": [598, 627],
    },
    "densities": ["TPDF", "RPDF", "GPDF"],
    "modes": list(MODES),
    "consts": {"number_of_bits": 16, "up_scaling": 32766,
               "down_scaling": 32768, "num_rand_variables": 6},
}


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


def _err(e):
    return {"type": type(e).__name__, "msg": str(e)}


def _cfg(kw):
    c = {
        "waveform": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "dithered": [[0.1, 0.25, 0.3], [0.4, 0.5, 0.6]],
        "density_function": "TPDF",
        "noise_shaping": False,
        "rng": [0] * 24,
        "n": 5,
        "values": [-1.5, -0.5, 0.5, 1.5, 2.5, -2.5, 0.0, 1.4999999999],
    }
    c.update(kw)
    return c


# ---- shape helpers: torch.tensor / reshape(-1, T) semantics -------------------
def _shape_of(data, dim=0):
    if not isinstance(data, list):
        return []
    if not data:
        return [0]
    ref = data[0]
    if isinstance(ref, list):
        inner = _shape_of(ref, dim + 1)
        for x in data:
            if not isinstance(x, list):
                raise ValueError("expected sequence of length %d at dim %d (got scalar)"
                                 % (len(ref), dim + 1))
            if len(x) != len(ref):
                raise ValueError("expected sequence of length %d at dim %d (got %d)"
                                 % (len(ref), dim + 1, len(x)))
            sx = _shape_of(x, dim + 1)
            if sx != inner:
                raise ValueError("expected sequence of length %d at dim %d (got %d)"
                                 % (inner[0], dim + 2, sx[0]))
        return [len(data)] + inner
    for x in data:
        if isinstance(x, list):
            raise ValueError("expected sequence of length %d at dim %d (got sequence)"
                             % (len(data), dim + 1))
    return [len(data)]


def _flatten(data):
    if isinstance(data, list):
        out = []
        for x in data:
            out.extend(_flatten(x))
        return out
    return [data]


def _nest(shape, flat):
    if len(shape) == 1:
        return list(flat[:shape[0]])
    step = 1
    for s in shape[1:]:
        step *= s
    return [_nest(shape[1:], flat[i * step:(i + 1) * step]) for i in range(shape[0])]


def _pack(data):
    if not isinstance(data, list) or not data:
        raise RuntimeError(EMPTY_MSG)
    sh = _shape_of(data)
    T = sh[-1]
    if T == 0:
        raise RuntimeError(EMPTY_MSG)
    flat = _flatten(data)
    C = len(flat) // T
    rows = [flat[i * T:(i + 1) * T] for i in range(C)]
    return sh, T, C, rows


# ---- primitives ----------------------------------------------------------------
def _bartlett(n):
    """PyTorch periodic Bartlett: split float expressions at n//2."""
    if n == 1:
        return [1.0]
    f = 2.0 / n
    half = n // 2
    return [(i * f if i <= half else 2.0 - i * f) for i in range(n)]


def _round(x):
    """torch.round: round-half-to-even, preserving signed zero."""
    if math.isnan(x) or math.isinf(x):
        return x
    n = math.floor(x)
    rem = x - n
    if rem > 0.5:
        n += 1
    elif rem == 0.5 and n % 2 != 0:
        n += 1
    r = float(n)
    if r == 0.0 and math.copysign(1.0, x) < 0:
        return -0.0
    return r


def _quantized(v):
    return _round(v) / DOWN


# ---- core: _apply_probability_distribution + dither ----------------------------
def _quant_core(data, density, rng, noise_shaping):
    sh, T, C, rows = _pack(data)
    signal_scaled = [[v * UP for v in row] for row in rows]
    win = _bartlett(T)
    cs, ts = C - 1, T - 1
    calls = []
    used = [0]  # RNG spy consumes the injected sequence in CALL ORDER

    def draw(high):
        if high <= 0:
            raise RuntimeError("random_ expects 'from' to be less than 'to', "
                               "but got from=0 >= to=%d" % high)
        k = used[0]
        used[0] += 1
        v = rng[k] if k < len(rng) else 0
        calls.append({"high": high, "size": [1], "value": v})
        return v

    # random_channel / random_time are computed BEFORE the branch, and the
    # TPDF branch never uses them (dead-code quirk).
    rc = draw(cs) if cs > 0 else 0
    rt = draw(ts) if ts > 0 else 0

    grid = [list(r) for r in rows]  # packed waveform: a VIEW, so mutable

    if density == "RPDF":
        add = grid[rc][rt] - 0.5
        dis = [[v + add for v in row] for row in signal_scaled]
    elif density == "GPDF":
        # `gaussian` is a 0-dim VIEW of grid[rc][rt]; each `+=` mutates that
        # cell, so later reads see the grown value (doubling when indices coincide).
        for _ in range(N_GAUSS):
            chan = draw(cs)
            t = draw(ts)
            grid[rc][rt] = grid[rc][rt] + grid[chan][t]
        g = grid[rc][rt]
        dis = [[v + g for v in row] for row in signal_scaled]
    else:
        # TPDF, and ANY unrecognised density string.
        dis = [[signal_scaled[c][t] + win[t] for t in range(T)]
               for c in range(C)]

    pre = [[_quantized(v) for v in row] for row in dis]
    if noise_shaping:
        # NOTE: `dither` passes its OWN waveform to _add_noise_shaping, and the
        # GPDF branch has already mutated that tensor in place through the
        # 0-dim view alias -> the "original" here is grid, NOT rows.
        out_rows = [[pre[c][t] + (0.0 if t == 0 else
                    pre[c][t - 1] - grid[c][t - 1]) for t in range(T)]
                    for c in range(C)]
    else:
        out_rows = pre

    return {
        "out": _clean(_nest(sh, [v for row in out_rows for v in row])),
        "shape": list(sh),
        "C": C,
        "T": T,
        "channel_size": cs,
        "time_size": ts,
        "rng_calls": [dict(x) for x in calls],
        "bartlett": _clean(list(win)),
        "signal_scaled": _clean(_nest(sh, [v for row in signal_scaled for v in row])),
    }


def _noise_shaping(wave, dith):
    wsh, T, C, rows = _pack(wave)
    dsh, dT, dC, drows = _pack(dith)
    err = [[drows[c][t] - rows[c][t] for t in range(dT)] for c in range(dC)]
    out = [[drows[c][t] + (0.0 if t == 0 else err[c][t - 1])
            for t in range(dT)] for c in range(dC)]
    return {
        "out": _clean(_nest(dsh, [v for row in out for v in row])),
        "shape": list(dsh),
        "error": _clean(_nest(dsh, [v for row in err for v in row])),
        "waveform_shape": list(wsh),
        "dithered_shape": list(dsh),
    }


def solve(mode, **kw):
    if mode == "census":
        return {k: ([list(x) for x in v] if k == "top_level" else
                    ({kk: (list(vv) if isinstance(vv, list) else
                           dict(vv) if isinstance(vv, dict) else vv)
                      for kk, vv in v.items()} if k == "funcs" else
                     dict(v) if k == "consts" else
                     list(v) if isinstance(v, list) else v))
                for k, v in CENSUS_SOURCE.items()}
    c = _cfg(kw)

    if mode == "bartlett":
        return {"n": c["n"], "window": _clean(_bartlett(c["n"]))}

    if mode == "round":
        vals = c["values"]
        return {"values": _clean(list(vals)),
                "rounded": _clean([_round(v) for v in vals]),
                "scaled": _clean([v * UP for v in vals])}

    if mode == "noise_shaping":
        return _noise_shaping(c["waveform"], c["dithered"])

    if mode == "branches":
        res = {}
        for d in DENSITIES:
            q = _quant_core(c["waveform"], d, c["rng"], False)
            res[d] = {"out": q["out"], "rng_calls": q["rng_calls"]}
        return res

    if mode == "shapes":
        cases = c.get("shapes") or [
            ("2x3", c["waveform"]),
            ("1x3", [c["waveform"][0]]),
            ("2x1", [[0.1], [0.4]]),
            ("1x1", [[0.1]]),
        ]
        res = {}
        for tag, data in cases:
            try:
                q = _quant_core(data, "TPDF", c["rng"], False)
                res[tag] = {
                    "shape": q["shape"],
                    "C": q["C"], "T": q["T"],
                    "channel_size": q["channel_size"],
                    "time_size": q["time_size"],
                    "rng_calls": q["rng_calls"],
                    "out": q["out"],
                }
            except Exception as e:
                res[tag] = _err(e)
        return res

    if mode == "errors":
        cases = [
            ("bogus_density", {"density_function": "BOGUS"}),
            ("empty_density", {"density_function": ""}),
            ("gpdf_single_channel", {"density_function": "GPDF",
                                     "waveform": [[0.1, 0.2, 0.3]]}),
            ("ragged", {"waveform": [[0.1, 0.2], [0.3]]}),
            ("empty", {"waveform": []}),
            ("t1", {"waveform": [[0.1], [0.4]]}),
            ("noise_shaping_on", {"noise_shaping": True}),
        ]
        res = {}
        for tag, extra in cases:
            cc = dict(c, **extra)
            try:
                q = _quant_core(cc["waveform"], cc["density_function"],
                                cc["rng"], cc["noise_shaping"])
                res[tag] = {"out": q["out"], "shape": q["shape"],
                            "rng_calls": q["rng_calls"]}
            except Exception as e:
                res[tag] = _err(e)
        return res

    if mode == "quantize":
        q = _quant_core(c["waveform"], c["density_function"], c["rng"],
                        c["noise_shaping"])
        return {"out": q["out"], "shape": q["shape"],
                "rng_calls": q["rng_calls"], "bartlett": q["bartlett"],
                "signal_scaled": q["signal_scaled"],
                "density": c["density_function"],
                "noise_shaping": bool(c["noise_shaping"])}

    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
