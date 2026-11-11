#!/usr/bin/env python3
"""868 judge cases: torchaudio spectral_centroid ledger.

Constants below are exact snapshots produced by fixtures868.py from the
verbatim official torchaudio spectral_centroid function. Helpers use
internal assert, never a returned bool. No torch, no import of solution.
"""
import math

# ---- official numerical fixtures ----
FREQ_16000_4 = [[0.0], [4000.0], [8000.0]]
FREQ_16000_5 = [[0.0], [4000.0], [8000.0]]
FREQ_16000_8 = [[0.0], [2000.0], [4000.0], [6000.0], [8000.0]]
FREQ_16000_14 = [[0.0], [1142.857142857143], [2285.714285714286], [3428.5714285714284], [4571.428571428572], [5714.285714285714], [6857.142857142857], [8000.0]]
FREQ_16001_4 = [[0.0], [4000.0], [8000.0]]
FREQ_22050_4 = [[0.0], [5512.5], [11025.0]]
FREQ_44100_6 = [[0.0], [7350.0], [14700.0], [22050.0]]
FREQ_10000_6 = [[0.0], [1666.6666666666667], [3333.333333333333], [5000.0]]
FREQ_8000_14 = [[0.0], [571.4285714285714], [1142.857142857143], [1714.2857142857142], [2285.714285714286], [2857.142857142857], [3428.5714285714284], [4000.0]]
FREQ_NEG8000_4 = [[0.0], [-2000.0], [-4000.0]]
FREQ_0_4 = [[0.0], [0.0], [0.0]]
FREQ_1_4 = [[0.0], [0.0], [0.0]]
FREQ_16000_0 = [[0.0]]
CENT_A_16000_4 = [5333.333333333333, 5500.0]
CENT_B_22050_4 = [8268.75, 8492.22972972973]
CENT_C_8000_4 = [2750.0, 2333.3333333333335, 1000.0, 3368.4210526315787]
CENT_D_16000_4 = ['nan', 5333.333333333333, 5500.0]
CENT_E_44100_6 = [11025.0, 11025.0, 11025.0, 11025.0]
CENT_F_16000_8 = [4666.666666666667, 4000.0]
CENT_A_16001_4 = [5333.333333333333, 5500.0]
CENT_A_NEG8000_4 = [-2666.6666666666665, -2750.0]
LEDGER_DEFAULT = {'sample_rate_half': 8000, 'n_freqs': 3, 'freqs': [[0.0], [4000.0], [8000.0]], 'freqs_shape': [3, 1], 'specgram_shape': [3, 2], 'weighted': [[0.0, 0.0], [8000.0, 4000.0], [24000.0, 40000.0]], 'numerator': [32000.0, 44000.0], 'denominator': [6.0, 8.0], 'centroid': [5333.333333333333, 5500.0], 'output_shape': [2], 'call': {'waveform_shape': [1, 8], 'pad': 0, 'window_shape': [4], 'n_fft': 4, 'hop_length': 2, 'win_length': 4, 'power': 1.0, 'normalized': False, 'center': True, 'pad_mode': 'reflect', 'onesided': True, 'return_complex': None}, 'freq_dim': -2}
LEDGER_ODD_SR = {'sample_rate_half': 8000, 'n_freqs': 3, 'freqs': [[0.0], [4000.0], [8000.0]], 'freqs_shape': [3, 1], 'specgram_shape': [3, 2], 'weighted': [[0.0, 0.0], [8000.0, 4000.0], [24000.0, 40000.0]], 'numerator': [32000.0, 44000.0], 'denominator': [6.0, 8.0], 'centroid': [5333.333333333333, 5500.0], 'output_shape': [2], 'call': {'waveform_shape': [1, 8], 'pad': 0, 'window_shape': [4], 'n_fft': 4, 'hop_length': 2, 'win_length': 4, 'power': 1.0, 'normalized': False, 'center': True, 'pad_mode': 'reflect', 'onesided': True, 'return_complex': None}, 'freq_dim': -2}
LEDGER_WIDE = {'sample_rate_half': 5000, 'n_freqs': 4, 'freqs': [[0.0], [1666.6666666666667], [3333.333333333333], [5000.0]], 'freqs_shape': [4, 1], 'specgram_shape': [4, 4], 'weighted': [[0.0, 0.0, 0.0, 0.0], [1666.6666666666667, 3333.3333333333335, 5000.0, 1666.6666666666667], [3333.333333333333, 6666.666666666666, 10000.0, 3333.333333333333], [5000.0, 10000.0, 15000.0, 5000.0]], 'numerator': [10000.0, 20000.0, 30000.0, 10000.0], 'denominator': [4.0, 8.0, 12.0, 4.0], 'centroid': [2500.0, 2500.0, 2500.0, 2500.0], 'output_shape': [4], 'call': {'waveform_shape': [1, 8], 'pad': 0, 'window_shape': [4], 'n_fft': 6, 'hop_length': 2, 'win_length': 4, 'power': 1.0, 'normalized': False, 'center': True, 'pad_mode': 'reflect', 'onesided': True, 'return_complex': None}, 'freq_dim': -2}
CALL_DEFAULT = {'waveform_shape': [1, 8], 'pad': 0, 'window_shape': [4], 'n_fft': 4, 'hop_length': 2, 'win_length': 4, 'power': 1.0, 'normalized': False, 'center': True, 'pad_mode': 'reflect', 'onesided': True, 'return_complex': None}
CALL_CUSTOM = {'waveform_shape': [2, 3, 32], 'pad': 3, 'window_shape': [8], 'n_fft': 8, 'hop_length': 5, 'win_length': 8, 'power': 1.0, 'normalized': False, 'center': True, 'pad_mode': 'reflect', 'onesided': True, 'return_complex': None}
ZERO_FRAME = {'shape': [3], 'value': ['nan', 5333.333333333333, 5500.0]}
SHAPES = {'2d': {'shape': [2], 'value': [5333.333333333333, 5500.0]}, '3d': {'shape': [2, 2], 'value': [[5333.333333333333, 5500.0], [3333.3333333333335, 4000.0]]}, '4d': {'shape': [2, 1, 2], 'value': [[[5333.333333333333, 5500.0]], [[3333.3333333333335, 4000.0]]]}}
ERRORS = {'freq_mismatch': {'type': 'RuntimeError', 'msg': 'The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0'}, 'odd_sample_rate': {'type': 'NO_ERROR', 'shape': [2], 'value': [5333.333333333333, 5500.0]}, 'negative_sample_rate': {'type': 'NO_ERROR', 'shape': [2], 'value': [-2666.6666666666665, -2750.0]}, 'n_fft_zero': {'type': 'NO_ERROR', 'shape': [2], 'value': [0.0, 0.0]}, 'empty_time': {'type': 'ValueError', 'msg': 'expected sequence of length 1 at dim 1 (got 0)'}, 'one_d_spec': {'type': 'IndexError', 'msg': 'Dimension out of range (expected to be in range of [-1, 0], but got -2)'}}
CENSUS = {'want': 'spectral_centroid', 'line_start': 1257, 'line_end': 1299, 'power_1': True, 'normalized_false': True, 'sample_rate_floor': True, 'n_fft_half_plus_one': True, 'reshape_col': True, 'freq_dim': True, 'weighted_sum': True, 'div_by_sum': True, 'default_center': True, 'default_pad_mode': True, 'default_onesided': True}
CENT_MIXED_SIGN = [float('-inf'), 'nan', 5500.0]
CENT_MIXED_POS = [float('inf'), 5333.333333333333]
SHAPES_NFFT2 = {'2d': {'type': 'RuntimeError', 'msg': 'The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0'}, '3d': {'type': 'RuntimeError', 'msg': 'The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1'}, '4d': {'type': 'RuntimeError', 'msg': 'The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2'}}

def _near(got, want, tag, rtol=1e-8, atol=1e-9):
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)), tag + ": expected a list"
        assert len(got) == len(want), "%s: length %d != %d" % (tag, len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            _near(a, b, "%s[%d]" % (tag, i), rtol, atol)
        return
    if isinstance(want, dict):
        assert isinstance(got, dict), tag + ": expected a dict"
        assert sorted(got) == sorted(want), "%s: keys %r != %r" % (tag, sorted(got), sorted(want))
        for k in want:
            _near(got[k], want[k], "%s.%s" % (tag, k), rtol, atol)
        return
    if isinstance(want, bool) or isinstance(got, bool):
        assert got is want, "%s: %r != %r" % (tag, got, want)
        return
    if isinstance(want, str) or want is None:
        assert got == want, "%s: %r != %r" % (tag, got, want)
        return
    if isinstance(want, float) and math.isnan(want):
        assert isinstance(got, float) and math.isnan(got), "%s: %r != nan" % (tag, got)
        return
    assert abs(float(got) - float(want)) <= atol + rtol * abs(float(want)), \
        "%s: %r != %r" % (tag, got, want)


def C(solve, mode, **kw):
    return solve(mode, **kw)


def _eq(got, want, tag):
    assert got == want, "%s: %r != %r" % (tag, got, want)


def _leaves(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_leaves(v))
        return out
    return [x]


SG_A = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]]
SG_B = [[0.5, 2.0], [1.5, 0.25], [3.0, 7.0]]
SG_C = [[2.0, 1.0, 4.0, 0.5], [1.0, 3.0, 1.0, 2.0], [5.0, 2.0, 1.0, 7.0]]
SG_D = [[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [0.0, 3.0, 5.0]]
SG_E = [[1.0, 2.0, 3.0, 1.0]] * 4
SG_F = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 1.0], [2.0, 2.0]]

CENT_CFG = {
    "A_16000_4": (16000, 4, SG_A),
    "B_22050_4": (22050, 4, SG_B),
    "C_8000_4": (8000, 4, SG_C),
    "D_16000_4": (16000, 4, SG_D),
    "E_44100_6": (44100, 6, SG_E),
    "F_16000_8": (16000, 8, SG_F),
    "A_16001_4": (16001, 4, SG_A),
    "A_NEG8000_4": (-8000, 4, SG_A),
}


def check_census(solve):
    _eq(C(solve, "census"), CENSUS, "census")


def check_census_lines(solve):
    c = C(solve, "census")
    assert c["want"] == "spectral_centroid", c["want"]
    assert c["line_start"] == 1257, c["line_start"]
    assert c["line_end"] == 1299, c["line_end"]


def check_census_source_facts(solve):
    c = C(solve, "census")
    for k, v in c.items():
        if k not in ("want", "line_start", "line_end"):
            assert v is True, "census.%s must be True: %r" % (k, v)


def check_call_default(solve):
    _eq(C(solve, "call_kwargs"), CALL_DEFAULT, "call_kwargs default")


def check_call_power_is_magnitude(solve):
    call = C(solve, "call_kwargs")
    assert call["power"] == 1.0, "must be magnitude spectrum, not power=2: %r" % call
    _eq(call["power"], 1.0, "power literal")


def check_call_normalized_false_bool(solve):
    call = C(solve, "call_kwargs")
    assert call["normalized"] is False, \
        "normalized must be the bool False, not a string: %r" % call["normalized"]


def check_call_spectrogram_defaults(solve):
    call = C(solve, "call_kwargs")
    assert call["center"] is True, call
    assert call["pad_mode"] == "reflect", call
    assert call["onesided"] is True, call
    assert call["return_complex"] is None, call


def check_call_custom(solve):
    _eq(C(solve, "call_kwargs", pad=3, n_fft=8, hop_length=5, win_length=8,
          waveform_shape=[2, 3, 32], window_shape=[8],
          specgram=[[1.0, 1.0]] * 5), CALL_CUSTOM, "call_kwargs custom")


def check_call_custom_constants(solve):
    call = C(solve, "call_kwargs", pad=3, n_fft=8, hop_length=5,
             win_length=8, waveform_shape=[2, 3, 32], window_shape=[8],
             specgram=[[1.0, 1.0]] * 5)
    assert call["power"] == 1.0 and call["normalized"] is False, call
    for k, v in (("pad", 3), ("n_fft", 8), ("hop_length", 5),
                 ("win_length", 8)):
        assert call[k] == v, "%s forwarded wrong: %r" % (k, call)
    assert call["waveform_shape"] == [2, 3, 32], call
    assert call["window_shape"] == [8], call


def check_freq_default(solve):
    _eq(C(solve, "freqs"), FREQ_16000_4, "freqs default")
    _eq(C(solve, "freqs"), [[0.0], [4000.0], [8000.0]], "freqs default literal")


def check_freq_all_grids_exact(solve):
    for name in sorted(k for k in globals() if k.startswith("FREQ_")):
        sr_s, nf_s = name[5:].rsplit("_", 1)
        sr = -int(sr_s[3:]) if sr_s.startswith("NEG") else int(sr_s)
        nf = int(nf_s)
        _eq(C(solve, "freqs", sample_rate=sr, n_fft=nf),
            globals()[name], "freqs " + name)


def check_freq_odd_sr_floor(solve):
    _eq(C(solve, "freqs", sample_rate=16001),
        C(solve, "freqs", sample_rate=16000), "odd sample_rate // 2")


def check_freq_odd_nfft_floor(solve):
    _eq(C(solve, "freqs", n_fft=5), C(solve, "freqs", n_fft=4),
        "odd n_fft // 2")


def check_freq_symmetric_back_half(solve):
    # torch.linspace computes the back half from `end`, not start + i*step
    grid = [r[0] for r in C(solve, "freqs", sample_rate=10000, n_fft=6)]
    _eq(grid, [r[0] for r in FREQ_10000_6], "symmetric grid")
    assert grid[0] == 0.0 and grid[-1] == 5000.0, grid
    step = 5000.0 / 3.0
    _eq(grid[3], 5000.0, "last element from end")
    _eq(grid[2], 5000.0 - step, "back-half element from end")


def check_freq_endpoint_floor(solve):
    g = [r[0] for r in C(solve, "freqs", sample_rate=16001)]
    assert g[-1] == 8000.0, g


def check_freq_nfft_zero(solve):
    _eq(C(solve, "freqs", n_fft=0), [[0.0]], "n_fft=0 single bin")


def check_freq_negative_sr(solve):
    _eq(C(solve, "freqs", sample_rate=-8000),
        [[0.0], [-2000.0], [-4000.0]], "negative sample_rate")


def check_freq_zero_sr(solve):
    for row in C(solve, "freqs", sample_rate=0):
        assert row[0] == 0.0, row


def check_freq_shape_is_column(solve):
    for sr, nf in ((16000, 4), (10000, 6), (8000, 14)):
        g = C(solve, "freqs", sample_rate=sr, n_fft=nf)
        assert len(g) == 1 + nf // 2, (sr, nf, g)
        for row in g:
            assert isinstance(row, list) and len(row) == 1, \
                "freqs must be a (F,1) column: %r" % g


def check_centroid_default(solve):
    _near(C(solve, "centroid"), CENT_A_16000_4, "centroid default")


def check_centroid_all_snapshots(solve):
    for tag, (sr, nf, sg) in sorted(CENT_CFG.items()):
        _near(C(solve, "centroid", sample_rate=sr, n_fft=nf, specgram=sg),
              globals()["CENT_" + tag], "centroid " + tag)


def check_centroid_hand_formula(solve):
    got = C(solve, "centroid")
    f = [0.0, 4000.0, 8000.0]
    num = [sum(f[i] * SG_A[i][t] for i in range(3)) for t in range(2)]
    den = [sum(SG_A[i][t] for i in range(3)) for t in range(2)]
    _near(num, [32000.0, 44000.0], "numerator")
    _near(den, [6.0, 8.0], "denominator")
    _near(got, [n / d for n, d in zip(num, den)], "centroid hand formula")


def check_centroid_zero_column_nan(solve):
    got = C(solve, "centroid", specgram=SG_D)
    _eq(got[0], "nan", "zero-denominator column must be nan")
    for v in got[1:]:
        assert isinstance(v, float) and math.isfinite(v), v


def check_centroid_odd_sr_same(solve):
    _eq(C(solve, "centroid", sample_rate=16001),
        C(solve, "centroid", sample_rate=16000), "odd/even sr centroid")


def check_centroid_negative_sr_negative(solve):
    got = C(solve, "centroid", sample_rate=-8000)
    assert all(v < 0.0 for v in got), got


def check_centroid_column_scaling_invariance(solve):
    base = C(solve, "centroid", specgram=SG_A)
    scaled = C(solve, "centroid",
               specgram=[[3.0, 8.0], [6.0, 4.0], [9.0, 20.0]])
    _near(base, scaled, "column scaling invariance")


def check_centroid_single_frequency_impulse(solve):
    f = [r[0] for r in C(solve, "freqs")]
    for k in range(3):
        sg = [[0.0, 0.0, 0.0] for _ in range(3)]
        sg[k] = [2.0, 5.0, 7.0]
        got = C(solve, "centroid", specgram=sg)
        for v in got:
            _near(v, f[k], "impulse k=%d" % k)


def check_centroid_bounded(solve):
    sg = [[2.0, 1.0, 4.0], [1.0, 3.0, 1.0], [5.0, 2.0, 1.0]]
    got = C(solve, "centroid", specgram=sg)
    for v in got:
        assert 0.0 <= v <= 8000.0, "nonnegative spectrum centroid out of band: %r" % v


def check_centroid_float_leaves(solve):
    for v in _leaves(C(solve, "centroid", specgram=SG_A)):
        assert type(v) is float, "centroid leaves must be Python floats: %r" % type(v)


def check_ledger_default(solve):
    _near(C(solve, "ledger"), LEDGER_DEFAULT, "ledger default")


def check_ledger_odd_sr(solve):
    _near(C(solve, "ledger", sample_rate=16001), LEDGER_ODD_SR, "ledger odd sr")


def check_ledger_wide(solve):
    _near(C(solve, "ledger", sample_rate=10000, n_fft=6, specgram=SG_E),
          LEDGER_WIDE, "ledger wide")


def check_ledger_self_consistency(solve):
    led = C(solve, "ledger")
    _near(led["centroid"],
          [n / d for n, d in zip(led["numerator"], led["denominator"])],
          "ledger centroid = num/den")
    _near(led["numerator"],
          [sum(row[t] for row in led["weighted"])
           for t in range(len(led["numerator"]))], "ledger numerator")
    _near(led["denominator"], [sum(row[t] for row in SG_A)
                               for t in range(2)], "ledger denominator")


def check_ledger_shapes(solve):
    led = C(solve, "ledger")
    assert led["freqs_shape"] == [3, 1], led["freqs_shape"]
    assert led["specgram_shape"] == [3, 2], led["specgram_shape"]
    assert led["output_shape"] == [2], led["output_shape"]
    assert led["freq_dim"] == -2, led["freq_dim"]


def check_ledger_call(solve):
    led = C(solve, "ledger")
    _eq(led["call"], CALL_DEFAULT, "ledger call kwargs")


def check_ledger_sample_rate_half(solve):
    led = C(solve, "ledger", sample_rate=22050)
    assert led["sample_rate_half"] == 11025, led["sample_rate_half"]
    assert led["n_freqs"] == 3, led["n_freqs"]


def check_zero_frame(solve):
    _near(C(solve, "zero_frame"), ZERO_FRAME, "zero_frame")


def check_zero_frame_nan(solve):
    z = C(solve, "zero_frame")
    _eq(z["shape"], [3], "zero_frame shape")
    _eq(z["value"][0], "nan", "zero frame must be nan")
    for v in z["value"][1:]:
        assert isinstance(v, float) and math.isfinite(v), v


def check_shapes(solve):
    _near(C(solve, "shapes"), SHAPES, "shapes")


def check_shapes_dims(solve):
    sh = C(solve, "shapes")
    assert sh["2d"]["shape"] == [2], sh["2d"]
    assert sh["3d"]["shape"] == [2, 2], sh["3d"]
    assert sh["4d"]["shape"] == [2, 1, 2], sh["4d"]


def check_shapes_3d_first_row(solve):
    sh = C(solve, "shapes")
    _eq(sh["3d"]["value"][0], sh["2d"]["value"], "3d first row == 2d")


def check_errors(solve):
    _near(C(solve, "errors"), ERRORS, "errors")


def check_errors_types(solve):
    e = C(solve, "errors")
    _eq(e["freq_mismatch"]["type"], "RuntimeError", "freq_mismatch type")
    _eq(e["one_d_spec"]["type"], "IndexError", "one_d_spec type")
    _eq(e["empty_time"]["type"], "ValueError", "empty_time type")
    for k in ("odd_sample_rate", "negative_sample_rate", "n_fft_zero"):
        _eq(e[k]["type"], "NO_ERROR", k)


def check_errors_messages(solve):
    e = C(solve, "errors")
    assert "must match" in e["freq_mismatch"]["msg"], e["freq_mismatch"]
    assert "-2" in e["one_d_spec"]["msg"], e["one_d_spec"]
    assert "dim 1" in e["empty_time"]["msg"], e["empty_time"]


def check_errors_denominator_uses_raw_specgram(solve):
    # A 1-D specgram would broadcast fine for the numerator, but the official
    # denominator sums the RAW specgram on dim=-2 -> IndexError.
    e = C(solve, "errors")
    assert e["one_d_spec"]["type"] == "IndexError", e["one_d_spec"]
    got = C(solve, "centroid", specgram=[1.0, 2.0, 3.0])
    assert isinstance(got, dict) and got["type"] == "IndexError", got


def check_errors_freq_mismatch_direct(solve):
    got = C(solve, "centroid", n_fft=4,
            specgram=[[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(got, dict) and got["type"] == "RuntimeError", got


def check_unknown_mode(solve):
    try:
        C(solve, "bogus_mode_868")
    except ValueError as ex:
        msg = str(ex)
        assert "unknown mode" in msg, msg
        assert "expected one of" in msg, msg
        assert "bogus_mode_868" in msg, msg
        assert "centroid" in msg and "census" in msg, \
            "message must list the real mode names: %r" % msg
        assert msg == ("unknown mode: bogus_mode_868 (expected one of "
                       + ", ".join(
                           ("centroid", "freqs", "call_kwargs", "ledger",
                            "zero_frame", "shapes", "errors", "census"))
                       + ")"), msg
    else:
        raise AssertionError("no error raised for unknown mode")


def check_census_returns_fresh_dict(solve):
    # Official comp("census") builds a brand-new dict on every call, so a
    # returned dict must never alias the module-level CENSUS constant.
    expected = dict(CENSUS)
    a = C(solve, "census")
    _eq(a, expected, "census payload")
    b = C(solve, "census")
    assert a is not b, "census must return a fresh dict on each call"
    a["want"] = "mutated_by_judge"
    c = C(solve, "census")
    _eq(c, expected, "mutating a returned census must not poison later calls")


def check_zero_denominator_inf_branch(solve):
    # den == 0 with num != 0 -> IEEE inf / -inf; den == 0 with num == 0 -> nan.
    neg = C(solve, "centroid",
            specgram=[[1.0, -1.0, 2.0], [-1.0, 2.0, 1.0], [0.0, -1.0, 5.0]])
    assert isinstance(neg, list) and len(neg) == 3, neg
    _eq(neg, CENT_MIXED_SIGN, "mixed-sign columns")
    assert neg[0] == float("-inf"), "negative numerator / zero -> -inf: %r" % neg[0]
    pos = C(solve, "centroid",
            specgram=[[-1.0, 2.0], [1.0, 2.0], [0.0, 5.0]])
    _eq(pos, CENT_MIXED_POS, "positive numerator / zero -> +inf")


def check_shapes_mismatched_n_fft(solve):
    # With n_fft=2 the frequency grid has 2 bins but every layout has F=3,
    # so ALL three layouts must go through the except branch and report the
    # real error dict (not a swallowed / faked NO_ERROR).
    got = C(solve, "shapes", n_fft=2)
    assert isinstance(got, dict) and sorted(got) == ["2d", "3d", "4d"], got
    for name in ("2d", "3d", "4d"):
        assert isinstance(got[name], dict), (name, got[name])
        _eq(got[name].get("type"), "RuntimeError", name + " type")
        assert "must match" in got[name]["msg"], (name, got[name])
    # torch reports a different non-singleton dimension index for 3-D / 4-D
    # inputs; only the flat 2-D case is pinned to the official literal.
    _eq(got["2d"], SHAPES_NFFT2["2d"], "2d exact official error")
