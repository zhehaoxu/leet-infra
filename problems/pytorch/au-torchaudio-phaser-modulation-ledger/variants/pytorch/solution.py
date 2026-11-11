#!/usr/bin/env python3
"""872 pure-Python reference solution: torchaudio phaser modulation ledger.

Only `math` and `struct` are used. Every mode mirrors the verbatim official
environment in official872.py, including int32 truncation of the INT wave
table, the two independent ring phases, the read-before-write order, the
in-place gain_out scaling, clamp(-1, 1), view() packing, and the exact
exception types/messages of the boundary cases.
"""
import math
import struct

MODES = ("census", "wave_table", "phaser", "branches", "params", "prefix",
         "errors", "shapes")
WAVE_TYPES = ("SINE", "TRIANGLE")
DATA_TYPES = ("INT", "FLOAT")

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
        [871, "gain"], [893, "highpass_biquad"], [926, "_lfilter_core_generic_loop"],
        [1027, "_lfilter"], [1032, "lfilter"], [1102, "lowpass_biquad"],
        [1135, "_overdrive_core_loop_generic"], [1150, "overdrive"],
        [1206, "phaser"], [1299, "riaa_biquad"], [1363, "treble_biquad"],
        [1414, "_measure"], [1485, "vad"],
    ],
    "funcs": {"_generate_wave_table": [16, 67], "phaser": [1206, 1296]},
    "modes": list(MODES),
    "params": ["sample_rate", "gain_in", "gain_out", "delay_ms", "decay",
               "mod_speed", "sinusoidal"],
    "defaults": {"gain_in": 0.4, "gain_out": 0.74, "delay_ms": 3.0,
                 "decay": 0.4, "mod_speed": 0.5, "sinusoidal": True},
    "arg_names": ["waveform", "sample_rate", "gain_in", "gain_out", "delay_ms",
                  "decay", "mod_speed", "sinusoidal"],
    "wave_types": list(WAVE_TYPES),
    "data_types": list(DATA_TYPES),
}

WAVE_LONG = [
    [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
     0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75, -0.85,
     0.05, -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75,
     0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9,
     0.12, -0.22, 0.32, -0.42, 0.52, -0.62, 0.72, -0.82],
    [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
     -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85,
     -0.05, 0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75,
     -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9,
     -0.12, 0.22, -0.32, 0.42, -0.52, 0.62, -0.72, 0.82],
]


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


def _shape_of(data):
    shape = []
    cur = data
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return shape


def _ragged(data):
    if not isinstance(data, list) or not data:
        return False
    if not isinstance(data[0], list):
        return False
    n = len(data[0])
    for row in data:
        if not isinstance(row, list) or len(row) != n:
            return True
    return any(_ragged(row[0]) if isinstance(row[0], list) else False for row in data)


def _cfg(kw):
    c = {
        "waveform": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                     [-0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]],
        "sample_rate": 1000,
        "gain_in": 0.4,
        "gain_out": 0.74,
        "delay_ms": 3.0,
        "decay": 0.4,
        "mod_speed": 200.0,
        "sinusoidal": True,
        "wave_type": "SINE",
        "data_type": "INT",
        "table_size": 5,
        "min": 1.0,
        "max": 3.0,
        "wt_phase": math.pi / 2,
    }
    c.update(kw)
    return c


# ------------------------------------------------------------- wave table
def _raw_d(wave_type, table_size, phase):
    """Return phase_offset, points, and the raw d values (float64)."""
    offset = int(phase / math.pi / 2 * table_size + 0.5)
    points = [(i + offset) % table_size for i in range(table_size)]
    ds = []
    for p in points:
        if wave_type == "SINE":
            d = (math.sin(p / table_size * 2 * math.pi) + 1) / 2
        elif wave_type == "TRIANGLE":
            d = p * 2 / table_size
            v = (4 * p) // table_size
            if v == 0:
                d = d + 0.5
            elif v == 1:
                d = 1.5 - d
            elif v == 2:
                d = 1.5 - d
            elif v == 3:
                d = d - 1.5
        else:
            # Official has no else: d stays all zeros.
            d = 0.0
        ds.append(d)
    return offset, points, ds


def _table(wave_type, data_type, table_size, mn, mx, phase):
    offset, points, ds = _raw_d(wave_type, table_size, phase)
    scaled = [d * (mx - mn) + mn for d in ds]
    if data_type == "INT":
        values = []
        for v in scaled:
            v = v - 0.5 if v < 0 else v + 0.5
            # .to(torch.int32) truncates toward zero
            values.append(int(v))
        dtype = "torch.int32"
    elif data_type == "FLOAT":
        values = [_f32(v) for v in scaled]
        dtype = "torch.float32"
    else:
        # Official has no else: d stays float64.
        values = list(scaled)
        dtype = "torch.float64"
    return values, dtype, offset, points


def _wave_table_mode(c):
    values, dtype, offset, points = _table(
        c["wave_type"], c["data_type"], c["table_size"],
        c["min"], c["max"], c["wt_phase"])
    return _clean({
        "kwargs": {
            "wave_type": c["wave_type"],
            "data_type": c["data_type"],
            "table_size": c["table_size"],
            "min": c["min"],
            "max": c["max"],
            "phase": c["wt_phase"],
            "device": "cpu",
        },
        "values": values,
        "dtype": dtype,
        "length": c["table_size"],
        "phase_offset": offset,
        "points": points,
    })


# ---------------------------------------------------------------- phaser
def _derived(c):
    sr = c["sample_rate"]
    dbl = int(c["delay_ms"] * 0.001 * sr + 0.5)
    mbl = int(sr / c["mod_speed"] + 0.5)
    shp = _shape_of(c["waveform"])
    return {
        "delay_buf_len": dbl,
        "mod_buf_len": mbl,
        "wave_type": "SINE" if c["sinusoidal"] else "TRIANGLE",
        "data_type": "INT",
        "wt_min": 1.0,
        "wt_max": float(dbl),
        "wt_phase": math.pi / 2,
        "phase_offset": int(math.pi / 2 / math.pi / 2 * mbl + 0.5),
        "n_channels": shp[-2] if len(shp) >= 2 else 1,
        "time": shp[-1],
    }


def _pack(wave):
    shape = _shape_of(wave)
    T = shape[-1]
    rows = []

    def collect(x):
        if isinstance(x[0], list):
            for r in x:
                collect(r)
        else:
            rows.append(x)

    collect(wave)
    return shape, T, rows


def _rebuild(shape, flat):
    if len(shape) == 1:
        return flat[:shape[0]]
    step = 1
    for d in shape[1:]:
        step *= d
    return [_rebuild(shape[1:], flat[k * step:(k + 1) * step])
            for k in range(shape[0])]


def _clamp(v):
    if math.isnan(v):
        return v
    return min(1.0, max(-1.0, v))


def _phaser_core(c):
    """Verbatim semantics of the official phaser loop, in pure Python."""
    wave = c["waveform"]
    if _ragged(wave):
        raise ValueError("expected sequence of length 2 at dim 1 (got 1)")
    shape, T, rows = _pack(wave)
    C = len(rows)
    sr = c["sample_rate"]
    delay_buf_len = int(c["delay_ms"] * 0.001 * sr + 0.5)
    if delay_buf_len < 0:
        # torch.zeros(C, negative)
        raise RuntimeError("zeros: Dimension size must be non-negative.")
    mod_buf_len = int(sr / c["mod_speed"] + 0.5)
    if mod_buf_len < 0:
        # torch.arange(negative)
        raise RuntimeError("upper bound and lower bound inconsistent with step sign")
    wt = "SINE" if c["sinusoidal"] else "TRIANGLE"
    mod_buf = _table(wt, "INT", mod_buf_len, 1.0, float(delay_buf_len),
                     math.pi / 2)[0]
    scaled = [[v * c["gain_in"] for v in row] for row in rows]
    # Official does `delay_buf = delay_buf * decay` on an all-zero tensor:
    # with a negative decay every initial cell becomes -0.0, not +0.0.
    buf = [[0.0 * c["decay"] for _ in range(C)]
           for _ in range(delay_buf_len)]
    out_cols = [[] for _ in range(C)]
    delay_pos = 0
    mod_pos = 0
    for i in range(T):
        # Official: int((delay_pos + mod_buf_list[mod_pos]) % delay_buf_len).
        # An empty mod_buf_list raises IndexError first; a zero delay_buf_len
        # makes the int32 tensor modulo raise RuntimeError("ZeroDivisionError").
        idx_base = mod_buf[mod_pos]
        if delay_buf_len == 0:
            raise RuntimeError("ZeroDivisionError")
        idx = (delay_pos + idx_base) % delay_buf_len
        mod_pos = (mod_pos + 1) % mod_buf_len
        delay_pos = (delay_pos + 1) % delay_buf_len
        temp = [scaled[ch][i] + buf[idx][ch] for ch in range(C)]
        buf[delay_pos] = [v * c["decay"] for v in temp]
        for ch in range(C):
            out_cols[ch].append(_clamp(temp[ch] * c["gain_out"]))
    flat = [v for col in out_cols for v in col]
    return _rebuild(shape, flat)


def _phaser_mode(c):
    derived = _clean(_derived(c))
    wt = "SINE" if c["sinusoidal"] else "TRIANGLE"
    values, dtype, _, _ = _table(wt, "INT", derived["mod_buf_len"], 1.0,
                                 float(derived["delay_buf_len"]), math.pi / 2)
    return {
        "out": _clean(_phaser_core(c)),
        "shape": _shape_of(c["waveform"]),
        "dtype": "torch.float64",
        "wave_table": _clean({
            "kwargs": {
                "wave_type": wt,
                "data_type": "INT",
                "table_size": derived["mod_buf_len"],
                "min": 1.0,
                "max": float(derived["delay_buf_len"]),
                "phase": math.pi / 2,
                "device": "cpu",
            },
            "values": values,
            "dtype": dtype,
            "length": derived["mod_buf_len"],
        }),
        "derived": derived,
    }


def _only_out(c):
    return _clean(_phaser_core(c))


def _errors_mode(c):
    cases = {
        "mod_speed_zero": lambda: _phaser_core(dict(c, mod_speed=0.0)),
        "sample_rate_zero": lambda: _phaser_core(dict(c, sample_rate=0)),
        "delay_ms_zero": lambda: _phaser_core(dict(c, delay_ms=0.0)),
        "delay_ms_negative": lambda: _phaser_core(dict(c, delay_ms=-3.0)),
        "mod_speed_negative": lambda: _phaser_core(dict(c, mod_speed=-200.0)),
        "one_dim": lambda: _phaser_core(dict(c, waveform=[0.1, 0.2, 0.3, 0.4])),
        "ragged": lambda: _phaser_core(dict(c, waveform=[[0.1, 0.2], [0.3]])),
        "five_channels": lambda: _phaser_core(
            dict(c, waveform=[[0.1, 0.2, 0.3, 0.4]] * 5)),
    }
    out = {}
    for tag, fn in cases.items():
        try:
            fn()
            out[tag] = {"type": "NO_ERROR"}
        except Exception as e:
            out[tag] = _err(e)
    return out


def _shapes_mode(c):
    base = c["waveform"]
    half = [[v * 0.5 for v in r] for r in base]
    quarter = [[v * 0.25 for v in r] for r in base]
    threeq = [[v * 0.75 for v in r] for r in base]
    shapes = {
        "d2": base,
        "d3": [base, half],
        "d4": [[base, half], [quarter, threeq]],
    }
    out = {}
    for tag, wave in shapes.items():
        try:
            cc = dict(c, waveform=wave)
            out[tag] = {"shape": _shape_of(wave), "out": _only_out(cc)}
        except Exception as e:
            out[tag] = _err(e)
    return out


def solve(mode, **kw):
    c = _cfg(kw)
    if mode == "census":
        return {k: (_clean(v) if isinstance(v, (dict, list)) else v)
                for k, v in CENSUS_SOURCE.items()}
    if mode == "wave_table":
        return _wave_table_mode(c)
    if mode == "phaser":
        return _phaser_mode(c)
    if mode == "branches":
        return {
            "sinusoidal_true": _only_out(dict(c, sinusoidal=True)),
            "sinusoidal_false": _only_out(dict(c, sinusoidal=False)),
            "decay_zero": _only_out(dict(c, decay=0.0)),
            "gain_in_zero": _only_out(dict(c, gain_in=0.0)),
            "gain_out_one": _only_out(dict(c, gain_out=1.0)),
        }
    if mode == "params":
        scans = {
            "gain_in": (0.0, 0.4, 1.0, 2.0),
            "gain_out": (0.0, 0.74, 1.0, 100.0),
            "delay_ms": (1.0, 3.0, 8.0, 20.0),
            "decay": (0.0, 0.4, 0.99, 1.5),
            "mod_speed": (100.0, 200.0, 500.0, 1000.0),
            "sample_rate": (500, 1000, 2000),
        }
        long = dict(c, waveform=WAVE_LONG)
        return {k: [("v%d" % j, _only_out(dict(long, **{k: v})))
                    for j, v in enumerate(vals)] for k, vals in scans.items()}
    if mode == "prefix":
        rows = c["waveform"]
        T = len(rows[0])
        return {"prefixes": [
            _only_out(dict(c, waveform=[[r[t] for t in range(n)] for r in rows]))
            for n in range(1, T + 1)]}
    if mode == "errors":
        return _errors_mode(c)
    if mode == "shapes":
        return _shapes_mode(c)
    raise ValueError("unknown mode: " + str(mode) + " (expected one of "
                     + ", ".join(MODES) + ")")
