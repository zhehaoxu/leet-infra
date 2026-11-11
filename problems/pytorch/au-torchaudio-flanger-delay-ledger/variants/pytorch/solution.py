#!/usr/bin/env python3
"""871 reference solution: torchaudio flanger delay-line ledger (pure Python).

Only `math` and `struct` are used. The verbatim official code runs on torch;
this module reproduces it bit-for-bit, including the float32 wave-table
downcast, torch.frac semantics, the regen feedback recurrence, the quadratic
interpolation algebra, and the final clamp.
"""
import math
import struct

MODES = ("census", "wave_table", "flanger", "branches", "params", "prefix",
         "errors", "shapes")
PARAMS = ("sample_rate", "delay", "depth", "regen", "width", "speed", "phase",
          "modulation", "interpolation")
MODULATIONS = ("sinusoidal", "triangular")
INTERPOLATIONS = ("linear", "quadratic")
WAVE_TYPES = ("SINE", "TRIANGLE")
DATA_TYPES = ("INT", "FLOAT")

SOURCE = "pytorch/audio main::src/torchaudio/functional/filtering.py"

_TOP = """12 _dB2Linear
16 _generate_wave_table
70 allpass_biquad
108 band_biquad
161 bandpass_biquad
207 bandreject_biquad
244 bass_biquad
295 biquad
336 contrast
370 dcshift
417 deemph_biquad
465 _add_noise_shaping
489 _apply_probability_distribution
598 dither
630 equalizer_biquad
672 filtfilt
713 flanger
871 gain
893 highpass_biquad
926 _lfilter_core_generic_loop
1027 _lfilter
1032 lfilter
1102 lowpass_biquad
1135 _overdrive_core_loop_generic
1150 overdrive
1206 phaser
1299 riaa_biquad
1363 treble_biquad
1414 _measure
1485 vad"""

DEFAULTS = {
    "delay": 0.0, "depth": 2.0, "regen": 0.0, "width": 71.0,
    "speed": 0.5, "phase": 25.0, "modulation": "sinusoidal",
    "interpolation": "linear",
}

CENSUS_SOURCE = {
    "source": SOURCE,
    "source_lines": 1702,
    "n_top_level": 30,
    "top_level": [[int(a), b] for a, b in
                  (ln.split() for ln in _TOP.splitlines())],
    "funcs": {"_generate_wave_table": [16, 67], "flanger": [713, 868]},
    "modes": list(MODES),
    "params": list(PARAMS),
    "defaults": dict(DEFAULTS),
    "arg_names": ["waveform", "sample_rate", "delay", "depth", "regen",
                  "width", "speed", "phase", "modulation", "interpolation"],
    "modulations": list(MODULATIONS),
    "interpolations": list(INTERPOLATIONS),
    "wave_types": list(WAVE_TYPES),
    "data_types": list(DATA_TYPES),
}

MSG_MOD = 'Only "sinusoidal" or "triangular" modulation allowed'
MSG_INT = 'Only "linear" or "quadratic" interpolation allowed'
MSG_CH = "Max 4 channels allowed"


# ------------------------------------------------------------------ helpers
def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


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
        "waveform": [[0.1, 0.2, 0.3, 0.4], [-0.2, 0.3, -0.4, 0.5]],
        "sample_rate": 100,
        "delay": 5.0,
        "depth": 10.0,
        "regen": 20.0,
        "width": 70.0,
        "speed": 10.0,
        "phase": 25.0,
        "modulation": "sinusoidal",
        "interpolation": "linear",
        "wave_type": "SINE",
        "data_type": "FLOAT",
        "table_size": 10,
        "min": 1.0,
        "max": 2.0,
        "wt_phase": 3 * math.pi / 2,
    }
    c.update(kw)
    return c


def _shape_of(data):
    sh = []
    cur = data
    while isinstance(cur, list):
        sh.append(len(cur))
        cur = cur[0] if cur else None
    return sh


def _flat(data):
    if isinstance(data, list):
        out = []
        for x in data:
            out.extend(_flat(x))
        return out
    return [data]


def _nest(shape, flat):
    if len(shape) == 1:
        return list(flat[:shape[0]])
    step = 1
    for s in shape[1:]:
        step *= s
    return [_nest(shape[1:], flat[i * step:(i + 1) * step])
            for i in range(shape[0])]


def _validate_waveform(data):
    """Reproduce torch.tensor / waveform.shape failures for malformed input."""
    sh = _shape_of(data)
    if len(sh) < 2:
        raise IndexError("tuple index out of range")
    widths = []

    def walk(x, depth):
        if depth == len(sh) - 1:
            widths.append(len(x) if isinstance(x, list) else None)
            return
        if not isinstance(x, list):
            widths.append(None)
            return
        for y in x:
            walk(y, depth + 1)
    walk(data, 0)
    if any(w != sh[-1] for w in widths):
        raise ValueError("expected sequence of length %d at dim 1 (got 1)"
                         % sh[-1])
    return sh


# ------------------------------------------------------------- wave table
def _raw_d(wave_type, table_size, phase):
    offset = int(phase / math.pi / 2 * table_size + 0.5)
    points = [(i + offset) % table_size for i in range(table_size)]
    ds = []
    for p in points:
        if wave_type == "SINE":
            base = (math.sin(p / table_size * 2 * math.pi) + 1) / 2
        elif wave_type == "TRIANGLE":
            base = p * 2 / table_size
            quarter = (4 * p) // table_size
            if quarter == 0:
                base = base + 0.5
            elif quarter in (1, 2):
                base = 1.5 - base
            else:
                base = base - 1.5
        else:
            # Official: neither SINE nor TRIANGLE matches -> d stays all zeros.
            base = 0.0
        ds.append(base)
    return offset, points, ds


def _table(wave_type, data_type, table_size, mn, mx, phase):
    offset, points, raw = _raw_d(wave_type, table_size, phase)
    scaled = [d * (mx - mn) + mn for d in raw]
    if data_type == "FLOAT":
        values = [_f32(v) for v in scaled]
        dtype = "torch.float32"
    elif data_type == "INT":
        values = [math.trunc(v - 0.5 if v < 0 else v + 0.5) for v in scaled]
        dtype = "torch.int32"
    else:
        # Official: neither INT nor FLOAT matches -> d stays float64.
        values = list(scaled)
        dtype = "torch.float64"
    return {"values": values, "dtype": dtype, "offset": offset,
            "points": points, "length": len(values)}


def _wave_table_mode(c):
    t = _table(c["wave_type"], c["data_type"], c["table_size"],
               c["min"], c["max"], c["wt_phase"])
    # The official environment cleans the WHOLE returned dict, so non-finite
    # min/max values inside kwargs become stable strings too.
    return _clean({
        "kwargs": {
            "wave_type": c["wave_type"], "data_type": c["data_type"],
            "table_size": c["table_size"], "min": c["min"], "max": c["max"],
            "phase": c["wt_phase"], "device": "cpu",
        },
        "values": t["values"],
        "dtype": t["dtype"],
        "length": t["length"],
        "phase_offset": t["offset"],
        "points": t["points"],
    })


# ------------------------------------------------------------------ scaling
def _scaling(c):
    feedback_gain = c["regen"] / 100
    delay_gain = c["width"] / 100
    channel_phase = c["phase"] / 100
    delay_min = c["delay"] / 1000
    delay_depth = c["depth"] / 1000
    in_gain = 1.0 / (1 + delay_gain)
    delay_gain = delay_gain / (1 + delay_gain)
    delay_gain = delay_gain * (1 - abs(feedback_gain))
    length = int((delay_min + delay_depth) * c["sample_rate"] + 0.5) + 2
    lfo_length = int(c["sample_rate"] / c["speed"])
    table_min = math.floor(delay_min * c["sample_rate"] + 0.5)
    table_max = length - 2.0
    n_channels = _shape_of(c["waveform"])[-2]
    phases = [math.trunc(i * lfo_length * channel_phase + 0.5)
              for i in range(n_channels)]
    return {
        "feedback_gain": feedback_gain,
        "channel_phase": channel_phase,
        "delay_min": delay_min,
        "delay_depth": delay_depth,
        "in_gain": in_gain,
        "delay_gain": delay_gain,
        "delay_buf_length": length,
        "table_min": table_min,
        "table_max": table_max,
        "lfo_length": lfo_length,
        "channel_phases": phases,
        "wave_type": "SINE" if c["modulation"] == "sinusoidal" else "TRIANGLE",
    }


def _clamp(v):
    if v != v:
        return v
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


# ------------------------------------------------------------- core replay
def _flanger_core(c):
    sh = _validate_waveform(c["waveform"])
    if c["modulation"] not in MODULATIONS:
        raise ValueError(MSG_MOD)
    if c["interpolation"] not in INTERPOLATIONS:
        raise ValueError(MSG_INT)
    if sh[-2] > 4:
        raise ValueError(MSG_CH)

    s = _scaling(c)
    L, lfo_length = s["delay_buf_length"], s["lfo_length"]
    if lfo_length == 0:
        raise RuntimeError("ZeroDivisionError")
    if L <= 0:
        raise RuntimeError("zeros: Dimension size must be non-negative.")

    t = _table(s["wave_type"], "FLOAT", lfo_length,
               float(s["table_min"]), s["table_max"], 3 * math.pi / 2)
    lfo = t["values"]
    wt_record = {
        "kwargs": {
            "wave_type": s["wave_type"], "data_type": "FLOAT",
            "table_size": lfo_length, "min": float(s["table_min"]),
            "max": float(s["table_max"]), "phase": 3 * math.pi / 2,
            "device": "cpu",
        },
        "values": _clean(list(lfo)),
        "dtype": "torch.float32",
        "length": lfo_length,
    }

    C, T = sh[-2], sh[-1]
    flat = _flat(c["waveform"])
    B = len(flat) // (C * T)
    # batch-major state: (b, ch) -> ring buffer list / last delayed sample
    bufs = {(b, ch): [0.0] * L for b in range(B) for ch in range(C)}
    last = {(b, ch): 0.0 for b in range(B) for ch in range(C)}
    out = [0.0] * len(flat)
    fg, ing, dg = s["feedback_gain"], s["in_gain"], s["delay_gain"]
    pos, lfo_pos = 0, 0
    for ti in range(T):
        pos = (pos + L - 1) % L
        for ch in range(C):
            delay_value = lfo[(lfo_pos + s["channel_phases"][ch]) % lfo_length]
            # torch.floor rounds toward -inf, but torch.frac is x - trunc(x)
            # (sign-preserving). For negative LFO samples the two disagree, so
            # the interpolation deliberately extrapolates backwards.
            integer = math.floor(delay_value)
            frac = _f32(delay_value - math.trunc(delay_value))
            if frac == 0.0:
                frac = 0.0
            for b in range(B):
                idx = (b * C + ch) * T + ti
                temp = flat[idx]
                key = (b, ch)
                ring = bufs[key]
                ring[pos] = temp + last[key] * fg
                d0 = ring[(pos + integer) % L]
                d1 = ring[(pos + integer + 1) % L]
                if c["interpolation"] == "linear":
                    delayed = d0 + (d1 - d0) * frac
                else:
                    d2 = ring[(pos + integer + 2) % L] - d0
                    dd1 = d1 - d0
                    a = d2 * 0.5 - dd1
                    bb = dd1 * 2 - d2 * 0.5
                    delayed = d0 + (a * frac + bb) * frac
                last[key] = delayed
                out[idx] = _clamp(temp * ing + delayed * dg)
        lfo_pos = (lfo_pos + 1) % lfo_length
    return {
        "out": _clean(_nest(sh, out)),
        "shape": sh,
        "dtype": "torch.float64",
        "wave_table": wt_record,
        "derived": _clean(dict(s)),
    }


def _only_out(c):
    return _flanger_core(c)["out"]


def _err_mode(c):
    cases = {
        "bad_modulation": dict(c, modulation="sawtooth"),
        "bad_interpolation": dict(c, interpolation="cubic"),
        "too_many_channels": dict(c, waveform=[[0.1, 0.2]] * 5),
        "speed_zero": dict(c, speed=0.0),
        "speed_large": dict(c, speed=10000.0),
        "one_dim": dict(c, waveform=[0.1, 0.2, 0.3]),
        "ragged": dict(c, waveform=[[0.1, 0.2], [0.3]]),
    }
    out = {}
    for tag, cc in cases.items():
        try:
            _flanger_core(cc)
            out[tag] = {"type": "NO_ERROR"}
        except Exception as e:
            out[tag] = _err(e)
    return out


def solve(mode, **kw):
    c = _cfg(kw)
    if mode == "census":
        out = {}
        for k, v in CENSUS_SOURCE.items():
            if isinstance(v, dict):
                out[k] = {kk: (list(vv) if isinstance(vv, list) else vv)
                          for kk, vv in v.items()}
            elif isinstance(v, list):
                out[k] = [list(x) if isinstance(x, list) else x for x in v]
            else:
                out[k] = v
        return out
    if mode == "wave_table":
        return _wave_table_mode(c)
    if mode == "flanger":
        return _flanger_core(c)
    if mode == "branches":
        return {m + "_" + i: _only_out(dict(c, modulation=m, interpolation=i))
                for m in MODULATIONS for i in INTERPOLATIONS}
    if mode == "params":
        scans = {
            "delay": (0.0, 5.0, 20.0),
            "depth": (0.0, 10.0, 30.0),
            "regen": (0.0, 20.0, 90.0),
            "width": (0.0, 70.0, 100.0),
            "speed": (5.0, 10.0, 20.0),
            "phase": (0.0, 25.0, 100.0),
        }
        return {k: [("v%d" % j, _only_out(dict(c, **{k: v})))
                    for j, v in enumerate(vals)] for k, vals in scans.items()}
    if mode == "prefix":
        rows = c["waveform"]
        T = len(rows[0])
        return {"prefixes": [
            _only_out(dict(c, waveform=[[r[t] for t in range(n)] for r in rows]))
            for n in range(1, T + 1)]}
    if mode == "errors":
        return _err_mode(c)
    if mode == "shapes":
        base = c["waveform"]
        shapes = {
            "d2": base,
            "d3": [base, [[v * 0.5 for v in r] for r in base]],
            "d4": [[base, [[v * 0.5 for v in r] for r in base]],
                   [[[v * 0.25 for v in r] for r in base],
                    [[v * 0.75 for v in r] for r in base]]],
        }
        out = {}
        for tag, wave in shapes.items():
            try:
                r = _flanger_core(dict(c, waveform=wave))
                out[tag] = {"shape": r["shape"], "out": r["out"]}
            except Exception as e:
                out[tag] = _err(e)
        return out
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
