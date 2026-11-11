#!/usr/bin/env python3
"""864 judge cases: torchaudio speed / pitch-shift orchestration ledger.

Constants below are exact snapshots produced by fixtures864.py from the
verbatim official implementation. Helpers use internal assert, never a
returned bool. No torch, no import of the solution module.
"""

# ---- official numerical fixtures ----
WF10 = [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]]
WF9 = [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55]]
WF15 = [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4, 1.5, 1.6, 1.7000000000000002, 1.8, 1.9000000000000001], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65, 1.75, 1.85, 1.9500000000000002, 2.05, 2.1500000000000004]]
WF3D = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]], [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]]
WF1D = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4, 1.5]
SH13 = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]]
SH4 = [[0.75, 0.8, 0.85, 0.9], [0.75, 0.8, 0.85, 0.9]]
SH10 = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
SH3 = [[0.75, 0.8, 0.85], [0.75, 0.8, 0.85]]
SH3D = [[[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]], [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]]
SH1D = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
SPEC9 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
SPEC5 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
SPEC8 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
SPEC4 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
SPEC3 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
RATE_N4 = 0.7937005259840998
RATE_N12 = 0.5
RATE_NNEG12 = 2.0
RATE_N0 = 1.0
RATE_N7B24 = 0.8169577266205499
RATE_N55 = 0.7278265914210937
RATE_B6N5 = 0.5612310241546865
RATE_B36N5 = 0.9082183626944031
PR_N4 = [20158, 16000]
PR_N12 = [32000, 16000]
PR_NNEG12 = [8000, 16000]
PR_N0 = [16000, 16000]
PR_N7B24 = [19584, 16000]
PR_SR44100_N7 = [66075, 44100]
PR_SR8000_N1 = [8475, 8000]
PR_SR12345_N5 = [16478, 12345]
PLAN_DEF = [4, 16, 16, 9]
PLAN_FFT8 = [2, 8, 8, 5]
PLAN_FFT15 = [3, 15, 15, 8]
PLAN_FFT5 = [1, 5, 5, 3]
PLAN_EXPLICIT = [7, 9, 9, 9]
PLAN_WINDOW3 = [4, 16, 3, 9]
PLAN_SPEC4 = [4, 16, 16, 4]
PA_DEF = [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586, 7.853981633974483, 9.42477796076938, 10.995574287564276, 12.566370614359172]
PA_FFT8 = [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586]
PA_FFT15 = [0.0, 1.3463968515384828, 2.6927937030769655, 4.039190554615448, 5.385587406153931, 6.731984257692414, 8.078381109230897, 9.42477796076938]
PA_FFT5 = [0.0, 1.5707963267948966, 3.141592653589793]
PA_HOP7 = [0.0, 2.748893571891069, 5.497787143782138, 8.246680715673207, 10.995574287564276, 13.744467859455344, 16.493361431346415, 19.24225500323748, 21.991148575128552]
PA_HOP0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PA_SPEC4 = [0.0, 4.1887902047863905, 8.377580409572783, 12.566370614359172]
LEN_DEF = 13
LEN_N12 = 20
LEN_N0 = 10
LEN_N7B24 = 12
LEN_BANKER9 = 4
LEN_BANKER15 = 8
LEN_BANKER5 = 2
LEN_BANKER7 = 4
PACK_DEF = [2, 10]
PACK_3D = [6, 7]
PACK_1D = [1, 11]
PACK_W1 = [2, 1]
FIX_TRIM = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
FIX_PAD = [[0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
FIX_EXACT = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
FIX_HEAVYPAD = [[0.75, 0.8, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.8, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
FIX_3D = [[[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]], [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]]]
FIX_1D = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
PITCH_DEF = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
PITCH_PAD = [[0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
PITCH_EXACT = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
PITCH_3D = [[[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]], [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]]]
FIX_CUSTOM = [[0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
SPEED_DEF = [24000, 16000, 8000, 3, 2]
SPEED_11 = [17600, 16000, 1600, 11, 10]
SPEED_08 = [35280, 44100, 8820, 4, 5]
SPEED_20 = [16000, 8000, 8000, 2, 1]
SPEED_10 = [16000, 16000, 16000, 1, 1]
SPEED_07 = [30869, 44100, 1, 30869, 44100]
SPEED_23_100 = [229, 100, 1, 229, 100]
SPEED_OUT_DEF = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], None]
LENS_F64 = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], [3.0, 4.0, 0.0, 6.0]]
LENS_I64 = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], [2, 4, 0, 5]]
LENS_F32 = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], [3.0, 4.0]]
LENS_I32 = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], [1, 2, 6]]
LENS_FACTOR2 = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], [5.0, 0.0, 3.0]]
SPEED_OUT_CUSTOM = [[[1.0, 2.0], [3.0, 4.0]], None]
SPEC1 = [[0.0, 0.0, 0.0]]
SPEC3D = [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
SH4_1D = [0.75, 0.8, 0.85, 0.9]
SHTWO = [[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], [9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0]]
FIX_1D_PAD = [0.75, 0.8, 0.85, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FIX_TWO_ROWS = [[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], [9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5]]
FIX_CUSTOM_LEN5 = [[0.75, 0.8, 0.85, 0.9, 0.0], [0.75, 0.8, 0.85, 0.9, 0.0]]
PITCH_VS_FIX = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
PA_SPEC1 = [0.0]
PLAN_SPEC3D = [4, 16, 16, 5]
PA_SPEC3D = [0.0, 3.141592653589793, 6.283185307179586, 9.42477796076938, 12.566370614359172]
PITCH_3D_DEFAULT = [[[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]], [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]]]
FIX_3D_DEFAULT = [[[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]], [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]]]
DEFAULT_RATE = 0.7937005259840998
DEFAULT_PITCH_RATES = [20158, 16000]
DEFAULT_STRETCH_PLAN = [4, 16, 16, 9]
DEFAULT_PHASE_ADVANCE = [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586, 7.853981633974483, 9.42477796076938, 10.995574287564276, 12.566370614359172]
DEFAULT_LEN_STRETCH = 13
DEFAULT_PACK_SHAPE = [2, 10]
DEFAULT_FIX_SHAPE = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
DEFAULT_PITCH_SHIFT = [[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
DEFAULT_SPEED_RATES = [24000, 16000, 8000, 3, 2]
DEFAULT_SPEED = [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.4500000000000002, 1.55, 1.65]], None]

import math


def _near(got, want, tag, rtol=1e-8, atol=1e-9):
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)), tag + ": expected a list"
        assert len(got) == len(want), "%s: length %d != %d" % (tag, len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            _near(a, b, "%s[%d]" % (tag, i), rtol, atol)
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


def _raises(fn, tag, needle, excs=(ValueError, TypeError)):
    try:
        fn()
    except excs as e:
        assert needle in str(e), "%s: wrong message %r (want %r)" % (tag, str(e), needle)
    except BaseException as e:
        raise AssertionError("%s: wrong exception %s %s" % (tag, type(e).__name__, e))
    else:
        raise AssertionError("%s: no exception" % tag)


def C(solve, mode, **kw):
    return solve(mode, **kw)


def WF(n, lead=(2,)):
    out = [[0.5 + 0.1 * i + 0.25 * j for i in range(n)] for j in range(lead[0])] \
        if lead else [0.5 + 0.1 * i for i in range(n)]
    return out


# ============================================================ A. rate
def check_rate_default(solve):
    _near(C(solve, "rate"), RATE_N4, "rate default")


def check_rate_default_constant(solve):
    _near(C(solve, "rate"), DEFAULT_RATE, "rate default constant")


def check_rate_n12(solve):
    _near(C(solve, "rate", n_steps=12), RATE_N12, "rate n=12")


def check_rate_nneg12(solve):
    _near(C(solve, "rate", n_steps=-12), RATE_NNEG12, "rate n=-12")


def check_rate_n0(solve):
    _near(C(solve, "rate", n_steps=0), RATE_N0, "rate n=0")


def check_rate_n7b24(solve):
    _near(C(solve, "rate", n_steps=7, bins_per_octave=24), RATE_N7B24, "rate 7/24")


def check_rate_fractional_steps(solve):
    _near(C(solve, "rate", n_steps=5.5), RATE_N55, "rate fractional steps")


def check_rate_b6(solve):
    _near(C(solve, "rate", n_steps=5, bins_per_octave=6), RATE_B6N5, "rate b=6")


def check_rate_b36(solve):
    _near(C(solve, "rate", n_steps=5, bins_per_octave=36), RATE_B36N5, "rate b=36")


def check_rate_identity(solve):
    _near(C(solve, "rate", n_steps=0), 1.0, "rate identity")


def check_rate_monotone(solve):
    assert C(solve, "rate", n_steps=12) < C(solve, "rate", n_steps=-12), "rate monotone"


def check_rate_independent_of_waveform(solve):
    _near(C(solve, "rate", waveform=WF(3)), RATE_N4, "rate ignores waveform")
    _near(C(solve, "rate", waveform=WF(23)), RATE_N4, "rate ignores waveform 2")


def check_rate_negative_exponent(solve):
    _near(C(solve, "rate", n_steps=-24), 4.0, "rate -24 steps", 1e-12)


# ============================================================ B. pitch rates
def check_pitch_rates_default(solve):
    _near(C(solve, "pitch_rates"), PR_N4, "pitch rates default")


def check_pitch_rates_n12(solve):
    _near(C(solve, "pitch_rates", n_steps=12), PR_N12, "pitch rates n=12")


def check_pitch_rates_nneg12(solve):
    _near(C(solve, "pitch_rates", n_steps=-12), PR_NNEG12, "pitch rates n=-12")


def check_pitch_rates_n0(solve):
    _near(C(solve, "pitch_rates", n_steps=0), PR_N0, "pitch rates n=0")


def check_pitch_rates_n7b24(solve):
    _near(C(solve, "pitch_rates", n_steps=7, bins_per_octave=24),
          PR_N7B24, "pitch rates 7/24")


def check_pitch_rates_sr44100(solve):
    _near(C(solve, "pitch_rates", n_steps=7, sample_rate=44100),
          PR_SR44100_N7, "pitch rates 44100")


def check_pitch_rates_sr8000(solve):
    _near(C(solve, "pitch_rates", n_steps=1, sample_rate=8000),
          PR_SR8000_N1, "pitch rates 8000")


def check_pitch_rates_sr12345(solve):
    _near(C(solve, "pitch_rates", n_steps=5, sample_rate=12345),
          PR_SR12345_N5, "pitch rates 12345")


def check_pitch_rates_int_not_round(solve):
    got = C(solve, "pitch_rates", n_steps=4, sample_rate=16000)
    assert got[0] == 20158, "int truncation of 20158.738"
    assert got[0] != 20159, "must not round or ceil"


def check_pitch_rates_target_passthrough(solve):
    for sr in (8000, 16000, 44100, 12345):
        assert C(solve, "pitch_rates", sample_rate=sr)[1] == sr, "target sr"


def check_pitch_rates_identity(solve):
    _near(C(solve, "pitch_rates", n_steps=0, sample_rate=44100),
          [44100, 44100], "identity rates")


def check_pitch_rates_float_sample_rate(solve):
    _near(C(solve, "pitch_rates", n_steps=0, sample_rate=8000.5),
          [8000, 8000], "float sample rate truncated")


# ============================================================ C. stretch plan
def check_plan_default(solve):
    _near(C(solve, "stretch_plan"), PLAN_DEF, "plan default")


def check_plan_fft8(solve):
    _near(C(solve, "stretch_plan", n_fft=8, spec_f=SPEC5), PLAN_FFT8, "plan fft8")


def check_plan_fft15(solve):
    _near(C(solve, "stretch_plan", n_fft=15, spec_f=SPEC8), PLAN_FFT15, "plan fft15")


def check_plan_fft5(solve):
    _near(C(solve, "stretch_plan", n_fft=5, spec_f=SPEC3), PLAN_FFT5, "plan fft5")


def check_plan_explicit(solve):
    _near(C(solve, "stretch_plan", hop_length=7, win_length=9, window=[0.2] * 9),
          PLAN_EXPLICIT, "plan explicit")


def check_plan_window_only(solve):
    _near(C(solve, "stretch_plan", window=[0.0] * 3), PLAN_WINDOW3, "plan window3")


def check_plan_spec4(solve):
    _near(C(solve, "stretch_plan", spec_f=SPEC4), PLAN_SPEC4, "plan spec4")


def check_plan_default_values(solve):
    _near(C(solve, "stretch_plan"), [4, 16, 16, 9], "plan hop/win/window/freq")


def check_plan_hop_is_floor_quarter(solve):
    _near(C(solve, "stretch_plan", n_fft=15, spec_f=SPEC8)[0], 3, "hop floor quarter")


def check_plan_freq_bins(solve):
    _near(C(solve, "stretch_plan", spec_f=SPEC4)[3], 4, "freq bins from spec")


def check_plan_window_len_override(solve):
    _near(C(solve, "stretch_plan", window=[0.1] * 2)[2], 2, "window len override")


# ============================================================ D. phase advance
def check_pa_default(solve):
    _near(C(solve, "phase_advance"), PA_DEF, "pa default")


def check_pa_fft8(solve):
    _near(C(solve, "phase_advance", n_fft=8, spec_f=SPEC5), PA_FFT8, "pa fft8")


def check_pa_fft15(solve):
    _near(C(solve, "phase_advance", n_fft=15, spec_f=SPEC8), PA_FFT15, "pa fft15")


def check_pa_fft5(solve):
    _near(C(solve, "phase_advance", n_fft=5, spec_f=SPEC3), PA_FFT5, "pa fft5")


def check_pa_hop7(solve):
    _near(C(solve, "phase_advance", hop_length=7), PA_HOP7, "pa hop7")


def check_pa_hop0(solve):
    _near(C(solve, "phase_advance", hop_length=0), PA_HOP0, "pa hop0")


def check_pa_spec4(solve):
    _near(C(solve, "phase_advance", spec_f=SPEC4), PA_SPEC4, "pa spec4")


def check_pa_length_is_freq_bins(solve):
    _near(len(C(solve, "phase_advance")), 9, "pa length = n_fft/2+1")
    _near(len(C(solve, "phase_advance", spec_f=SPEC4)), 4, "pa length = spec rows")


def check_pa_first_zero(solve):
    _near(C(solve, "phase_advance")[0], 0.0, "pa first zero")


def check_pa_last_pi_hop(solve):
    _near(C(solve, "phase_advance")[-1], 4.0 * math.pi, "pa last pi*hop")
    _near(C(solve, "phase_advance", hop_length=7)[-1], 7.0 * math.pi, "pa last pi*7")


def check_pa_uniform(solve):
    pa = C(solve, "phase_advance")
    step = (pa[-1] - pa[0]) / (len(pa) - 1)
    for i in range(1, len(pa) - 1):
        _near(pa[i], i * step, "pa uniform %d" % i)


def check_pa_not_frame_count(solve):
    _near(len(C(solve, "phase_advance", spec_f=SPEC4)), 4, "pa not frame count")


# ============================================================ E. len_stretch
def check_len_default(solve):
    _near(C(solve, "len_stretch"), LEN_DEF, "len default")


def check_len_n12(solve):
    _near(C(solve, "len_stretch", n_steps=12), LEN_N12, "len n=12")


def check_len_n0(solve):
    _near(C(solve, "len_stretch", n_steps=0), LEN_N0, "len n=0")


def check_len_n7b24(solve):
    _near(C(solve, "len_stretch", n_steps=7, bins_per_octave=24),
          LEN_N7B24, "len 7/24")


def check_len_banker9(solve):
    _near(C(solve, "len_stretch", n_steps=-12, waveform=WF9), LEN_BANKER9, "len 4.5 -> 4")
    assert LEN_BANKER9 == 4, "fixture tie to even"


def check_len_banker15(solve):
    _near(C(solve, "len_stretch", n_steps=-12, waveform=WF15),
          LEN_BANKER15, "len 7.5 -> 8")
    assert LEN_BANKER15 == 8, "fixture tie to even"


def check_len_banker5(solve):
    _near(C(solve, "len_stretch", n_steps=-12, waveform=WF(5)),
          LEN_BANKER5, "len 2.5 -> 2")
    assert LEN_BANKER5 == 2, "fixture tie to even"


def check_len_banker7(solve):
    _near(C(solve, "len_stretch", n_steps=-12, waveform=WF(7)),
          LEN_BANKER7, "len 3.5 -> 4")
    assert LEN_BANKER7 == 4, "fixture tie to even"


def check_len_not_floor_half(solve):
    got = C(solve, "len_stretch", n_steps=-12, waveform=WF(9))
    assert got != 5, "must not use floor(x+0.5)"


def check_len_identity(solve):
    _near(C(solve, "len_stretch", n_steps=0), 10, "len identity")


def check_len_scales_with_rate(solve):
    r = C(solve, "rate", n_steps=12)
    _near(C(solve, "len_stretch", n_steps=12), int(round(10.0 / r)), "len from rate")


# ============================================================ F. pack shape
def check_pack_default(solve):
    _near(C(solve, "pack_shape"), PACK_DEF, "pack default")


def check_pack_3d(solve):
    _near(C(solve, "pack_shape", waveform=WF3D, shifted=SH3D), PACK_3D, "pack 3d")


def check_pack_1d(solve):
    _near(C(solve, "pack_shape", waveform=WF1D, shifted=SH1D), PACK_1D, "pack 1d")


def check_pack_w1(solve):
    _near(C(solve, "pack_shape", waveform=WF(1)), PACK_W1, "pack single sample")


def check_pack_flattens_leading(solve):
    _near(C(solve, "pack_shape", waveform=WF3D, shifted=SH3D), [6, 7], "pack flatten")


def check_pack_keeps_ori_len(solve):
    assert C(solve, "pack_shape", waveform=WF(13))[1] == 13, "pack ori_len"


# ============================================================ G. fix shape
def check_fix_trim(solve):
    _near(C(solve, "fix_shape", shifted=SH13), FIX_TRIM, "fix trim")


def check_fix_pad(solve):
    _near(C(solve, "fix_shape", shifted=SH4), FIX_PAD, "fix pad")


def check_fix_exact(solve):
    _near(C(solve, "fix_shape", shifted=SH10), FIX_EXACT, "fix exact")


def check_fix_heavy_pad(solve):
    _near(C(solve, "fix_shape", shifted=SH3), FIX_HEAVYPAD, "fix heavy pad")


def check_fix_3d(solve):
    _near(C(solve, "fix_shape", waveform=WF3D, shifted=SH3D), FIX_3D, "fix 3d")


def check_fix_1d(solve):
    _near(C(solve, "fix_shape", waveform=WF1D, shifted=SH1D), FIX_1D, "fix 1d")


def check_fix_custom_target(solve):
    _near(C(solve, "fix_shape", fix_waveform=SH4, fix_target_shape=[2, 10]),
          FIX_CUSTOM, "fix custom target")


def check_fix_pad_is_zero(solve):
    out = C(solve, "fix_shape", shifted=SH4)
    _near(out[0][4:], [0.0] * 6, "fix pad zeros")
    _near(out[1][4:], [0.0] * 6, "fix pad zeros row 2")


def check_fix_trim_keeps_prefix(solve):
    out = C(solve, "fix_shape", shifted=SH13)
    _near(out[0], SH13[0][:10], "fix trim prefix")


def check_fix_shape_rank(solve):
    out = C(solve, "fix_shape", waveform=WF3D, shifted=SH3D)
    assert len(out) == 2 and len(out[0]) == 3 and len(out[0][0]) == 7, "fix view rank"


# ============================================================ H. pitch shift
def check_pitch_default(solve):
    _near(C(solve, "pitch_shift"), PITCH_DEF, "pitch default")


def check_pitch_pad(solve):
    _near(C(solve, "pitch_shift", shifted=SH4), PITCH_PAD, "pitch pad")


def check_pitch_exact(solve):
    _near(C(solve, "pitch_shift", shifted=SH10), PITCH_EXACT, "pitch exact")


def check_pitch_3d(solve):
    _near(C(solve, "pitch_shift", waveform=WF3D, shifted=SH3D),
          PITCH_3D, "pitch 3d")


def check_pitch_equals_fix(solve):
    _near(C(solve, "pitch_shift", shifted=SH4),
          C(solve, "fix_shape", shifted=SH4), "pitch == fix")


def check_pitch_default_constant(solve):
    _near(C(solve, "pitch_shift"), DEFAULT_PITCH_SHIFT, "pitch default const")


# ============================================================ I. speed rates
def check_speed_rates_default(solve):
    _near(C(solve, "speed_rates"), SPEED_DEF, "speed rates default")


def check_speed_rates_11(solve):
    _near(C(solve, "speed_rates", speed_factor=1.1), SPEED_11, "speed 1.1")


def check_speed_rates_08(solve):
    _near(C(solve, "speed_rates", speed_factor=0.8, speed_orig_freq=44100),
          SPEED_08, "speed 0.8")


def check_speed_rates_20(solve):
    _near(C(solve, "speed_rates", speed_factor=2.0, speed_orig_freq=8000),
          SPEED_20, "speed 2.0")


def check_speed_rates_10(solve):
    _near(C(solve, "speed_rates", speed_factor=1.0), SPEED_10, "speed 1.0")


def check_speed_rates_float_truncation(solve):
    _near(C(solve, "speed_rates", speed_factor=0.7, speed_orig_freq=44100),
          SPEED_07, "speed 0.7 truncation")
    assert SPEED_07[0] == 30869, "0.7*44100 = 30869.999... -> 30869"


def check_speed_rates_23_100(solve):
    _near(C(solve, "speed_rates", speed_factor=2.3, speed_orig_freq=100),
          SPEED_23_100, "speed 2.3/100")


def check_speed_rates_gcd_reduces(solve):
    r = C(solve, "speed_rates")
    assert r[2] == 8000 and r[3:] == [3, 2], "gcd reduction"
    assert r[0] // r[2] == r[3] and r[1] // r[2] == r[4], "reduced quotient"


def check_speed_rates_identity(solve):
    _near(C(solve, "speed_rates", speed_factor=1.0),
          [16000, 16000, 16000, 1, 1], "identity gcd")


def check_speed_rates_float_orig(solve):
    _near(C(solve, "speed_rates", speed_orig_freq=16000.9)[1], 16000,
          "int orig_freq")


def check_speed_rates_lengths_independent(solve):
    _near(C(solve, "speed_rates", speed_lengths=[1.0]), SPEED_DEF,
          "lengths do not affect rates")


# ============================================================ J. speed
def check_speed_default(solve):
    _near(C(solve, "speed"), SPEED_OUT_DEF, "speed default")


def check_speed_lens_f64(solve):
    _near(C(solve, "speed", speed_lengths=[3.2, 5.0, 0.0, 7.9]),
          LENS_F64, "speed lengths f64")


def check_speed_lens_i64(solve):
    _near(C(solve, "speed", speed_lengths=[3, 5, 0, 7],
            speed_lengths_dtype="int64"), LENS_I64, "speed lengths i64")


def check_speed_lens_f32(solve):
    _near(C(solve, "speed", speed_lengths=[3.2, 5.0],
            speed_lengths_dtype="float32"), LENS_F32, "speed lengths f32")


def check_speed_lens_i32(solve):
    _near(C(solve, "speed", speed_lengths=[1, 2, 9],
            speed_lengths_dtype="int32"), LENS_I32, "speed lengths i32")


def check_speed_lens_factor2(solve):
    _near(C(solve, "speed", speed_factor=2.0, speed_lengths=[9.0, 0.0, 4.5]),
          LENS_FACTOR2, "speed lengths factor 2")


def check_speed_custom_output(solve):
    _near(C(solve, "speed", speed_out=[[1.0, 2.0], [3.0, 4.0]],
            speed_waveform=[[0.0, 0.0], [0.0, 0.0]]),
          SPEED_OUT_CUSTOM, "speed custom output")


def check_speed_none_lengths(solve):
    out = C(solve, "speed")
    assert out[1] is None, "None lengths stays None"


def check_speed_int_dtype_is_int(solve):
    out = C(solve, "speed", speed_lengths=[3, 5, 0, 7],
            speed_lengths_dtype="int64")[1]
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in out), \
        "int dtype must cast back"


def check_speed_ceil_not_floor(solve):
    out = C(solve, "speed", speed_lengths=[3.2, 5.0, 0.0, 7.9])[1]
    _near(out, [3.0, 4.0, 0.0, 6.0], "ceil results")
    assert out[0] != 2 and out[1] != 3, "must ceil, not floor"


def check_speed_output_passthrough(solve):
    out = C(solve, "speed")[0]
    _near(out, WF10, "speed returns injected waveform")


def check_speed_default_constant(solve):
    _near(C(solve, "speed"), DEFAULT_SPEED, "speed default const")


# ============================================================ K. validation
def check_unknown_mode(solve):
    _raises(lambda: solve("bogus"), "unknown mode", "unknown mode")
    try:
        solve("bogus")
    except (ValueError, TypeError) as e:
        msg = str(e)
        for m in ("rate", "pitch_rates", "stretch_plan", "phase_advance",
                  "len_stretch", "pack_shape", "fix_shape", "pitch_shift",
                  "speed_rates", "speed"):
            assert m in msg, "unknown-mode message must list %s: %r" % (m, msg)
    else:
        raise AssertionError("unknown mode must raise")


def check_zero_bins_raises(solve):
    _raises(lambda: solve("rate", bins_per_octave=0), "zero bins", "division",
            (ZeroDivisionError,))


def check_waveform_ragged(solve):
    _raises(lambda: solve("rate", waveform=[[1.0, 2.0], [3.0]]),
            "ragged waveform", "rectangular")


def check_waveform_bool_leaf(solve):
    _raises(lambda: solve("rate", waveform=[[True, 2.0], [3.0, 4.0]]),
            "bool leaf", "contain numbers")


def check_waveform_string_leaf(solve):
    _raises(lambda: solve("rate", waveform=[["x", 2.0], [3.0, 4.0]]),
            "string leaf", "contain numbers")


def check_waveform_not_nested(solve):
    _raises(lambda: solve("rate", waveform=3.0), "scalar waveform", "nested list")


def check_nfft_type(solve):
    _raises(lambda: solve("rate", n_fft="x", spec_f=SPEC9),
            "n_fft str", "n_fft must be an int", (TypeError,))


def check_nfft_bool(solve):
    _raises(lambda: solve("rate", n_fft=True, spec_f=SPEC9),
            "n_fft bool", "n_fft must be an int", (TypeError,))


def check_sample_rate_type(solve):
    _raises(lambda: solve("rate", sample_rate="x"),
            "sample_rate str", "sample_rate must be a number", (TypeError,))


def check_nsteps_type(solve):
    _raises(lambda: solve("rate", n_steps="x"),
            "n_steps str", "n_steps must be a number", (TypeError,))


def check_bins_type(solve):
    _raises(lambda: solve("rate", bins_per_octave="x"),
            "bins str", "bins_per_octave must be a number", (TypeError,))


def check_hop_type(solve):
    _raises(lambda: solve("stretch_plan", hop_length=1.5),
            "hop float", "hop_length must be an int", (TypeError,))


def check_win_type(solve):
    _raises(lambda: solve("stretch_plan", win_length="x"),
            "win str", "win_length must be an int", (TypeError,))


def check_window_bad_leaf(solve):
    _raises(lambda: solve("stretch_plan", window=[True, False]),
            "window bool", "window must contain numbers")


def check_spec_f_1d(solve):
    _raises(lambda: solve("phase_advance", spec_f=[0.0, 1.0, 2.0]),
            "spec_f 1d", "at least two dimensions")


def check_spec_f_bad_leaf(solve):
    _raises(lambda: solve("phase_advance", spec_f=[[0.0, "x"], [1.0, 2.0]]),
            "spec_f str", "spec_f must contain numbers")


def check_shifted_ragged(solve):
    _raises(lambda: solve("pitch_shift", shifted=[[1.0, 2.0], [3.0]]),
            "shifted ragged", "shifted must be rectangular")


def check_fix_waveform_bad_leaf(solve):
    _raises(lambda: solve("fix_shape", fix_waveform=[[1.0], ["x"]]),
            "fix bad leaf", "fix_waveform must contain numbers")


def check_fix_target_empty(solve):
    _raises(lambda: solve("fix_shape", fix_target_shape=[]),
            "empty target", "non-empty shape")


def check_speed_lengths_bad_leaf(solve):
    _raises(lambda: solve("speed", speed_lengths=[1.0, False]),
            "lengths bool", "speed_lengths must contain numbers")


def check_speed_out_bad_leaf(solve):
    _raises(lambda: solve("speed", speed_out=[[1.0, "x"], [2.0, 3.0]]),
            "speed_out str", "speed_out must contain numbers")


def check_speed_waveform_bad_leaf(solve):
    _raises(lambda: solve("speed", speed_waveform=[[1.0], [True]]),
            "speed_waveform bool", "speed_waveform must contain numbers")


def check_stretched_bad_leaf(solve):
    _raises(lambda: solve("pitch_shift", stretched=[1.0, "x"]),
            "stretched str", "stretched must contain numbers")


def check_speed_factor_type(solve):
    _raises(lambda: solve("speed_rates", speed_factor="x"),
            "speed_factor str", "speed_factor must be a number", (TypeError,))


def check_orig_freq_type(solve):
    _raises(lambda: solve("speed_rates", speed_orig_freq="x"),
            "orig_freq str", "speed_orig_freq must be a number", (TypeError,))


# ============================================================ L. consistency
def check_rate_matches_pitch_rates(solve):
    r = C(solve, "rate")
    pr = C(solve, "pitch_rates")
    _near(pr[0], int(16000.0 / r), "pitch orig freq from rate")


def check_len_matches_rate(solve):
    for n in (0, 4, 12, -12, 7):
        r = C(solve, "rate", n_steps=n)
        _near(C(solve, "len_stretch", n_steps=n), int(round(10.0 / r)),
              "len from rate n=%d" % n)


def check_pa_length_matches_plan(solve):
    _near(len(C(solve, "phase_advance")), C(solve, "stretch_plan")[3],
          "pa length == freq bins")


def check_pa_last_matches_plan(solve):
    hop = C(solve, "stretch_plan")[0]
    _near(C(solve, "phase_advance")[-1], math.pi * hop, "pa last == pi*hop")


def check_pack_matches_waveform(solve):
    _near(C(solve, "pack_shape", waveform=WF(13))[1], 13, "pack ori_len")


def check_pitch_equals_fix_all_branches(solve):
    for cfg in (dict(shifted=SH13), dict(shifted=SH4), dict(shifted=SH10)):
        _near(C(solve, "pitch_shift", **cfg), C(solve, "fix_shape", **cfg),
              "pitch == fix branch")


def check_default_constants_all_modes(solve):
    for mode, val in [
        ("rate", DEFAULT_RATE), ("pitch_rates", DEFAULT_PITCH_RATES),
        ("stretch_plan", DEFAULT_STRETCH_PLAN),
        ("phase_advance", DEFAULT_PHASE_ADVANCE),
        ("len_stretch", DEFAULT_LEN_STRETCH), ("pack_shape", DEFAULT_PACK_SHAPE),
        ("fix_shape", DEFAULT_FIX_SHAPE), ("pitch_shift", DEFAULT_PITCH_SHIFT),
        ("speed_rates", DEFAULT_SPEED_RATES), ("speed", DEFAULT_SPEED),
    ]:
        _near(C(solve, mode), val, "default %s" % mode)


def check_speed_reduced_matches_raw(solve):
    r = C(solve, "speed_rates")
    assert math.gcd(r[0], r[1]) == r[2], "gcd consistency"


def check_identity_rate_chain(solve):
    _near(C(solve, "rate", n_steps=0), 1.0, "identity rate")
    _near(C(solve, "len_stretch", n_steps=0), 10, "identity len")
    _near(C(solve, "pitch_rates", n_steps=0), [16000, 16000], "identity rates")


def check_fix_output_shape(solve):
    for cfg in (dict(shifted=SH13), dict(shifted=SH4), dict(shifted=SH10)):
        out = C(solve, "fix_shape", **cfg)
        assert len(out) == 2 and all(len(row) == 10 for row in out), "fix shape"


# ============================================================ M. hardening
def check_unknown_mode_order(solve):
    try:
        solve("bogus")
    except (ValueError, TypeError) as e:
        msg = str(e)
        want = ("rate, pitch_rates, stretch_plan, phase_advance, len_stretch, "
                "pack_shape, fix_shape, pitch_shift, speed_rates, speed")
        assert want in msg, "unknown-mode message must list MODES in order: %r" % msg
    else:
        raise AssertionError("unknown mode must raise")


def check_empty_waveform(solve):
    _raises(lambda: solve("rate", waveform=[]), "empty waveform", "rectangular")


def check_speed_default_dtype_float(solve):
    out = C(solve, "speed", speed_lengths=[3.2, 5.0])[1]
    assert all(isinstance(v, float) for v in out), \
        "default float64 dtype must not cast lengths to int"


def check_fix_two_distinct_rows(solve):
    _near(C(solve, "fix_shape", shifted=SHTWO), FIX_TWO_ROWS, "fix distinct rows")
    out = C(solve, "fix_shape", shifted=SHTWO)
    assert out[0][0] != out[1][0], "rows must not collapse to the first row"


def check_fix_1d_pad(solve):
    _near(C(solve, "fix_shape", fix_waveform=SH4_1D,
            fix_target_shape=[10]), FIX_1D_PAD, "fix 1d pad")


def check_pa_single_freq_bin(solve):
    _near(C(solve, "phase_advance", spec_f=SPEC1), PA_SPEC1, "pa one bin")
    _near(C(solve, "phase_advance", spec_f=SPEC1), [0.0], "pa one bin value")


def check_pa_spec_3d(solve):
    _near(C(solve, "phase_advance", spec_f=SPEC3D), PA_SPEC3D, "pa 3d spec")


def check_plan_spec_3d(solve):
    _near(C(solve, "stretch_plan", spec_f=SPEC3D), PLAN_SPEC3D, "plan 3d spec")


def check_pitch_3d_default_shifted(solve):
    _near(C(solve, "pitch_shift", waveform=WF3D),
          PITCH_3D_DEFAULT, "pitch 3d default shifted")


def check_fix_3d_default_shifted(solve):
    _near(C(solve, "fix_shape", waveform=WF3D),
          FIX_3D_DEFAULT, "fix 3d default shifted")


def check_fix_custom_target_len5(solve):
    _near(C(solve, "fix_shape", fix_waveform=SH4,
            fix_target_shape=[2, 5]), FIX_CUSTOM_LEN5, "fix custom len5")


def check_pitch_ignores_fix_inputs(solve):
    _near(C(solve, "pitch_shift", shifted=SH13, fix_waveform=SH4,
            fix_target_shape=[2, 5]), PITCH_VS_FIX, "pitch ignores fix_*")
    _near(C(solve, "pitch_shift", shifted=SH13, fix_waveform=SH4,
            fix_target_shape=[2, 5]), FIX_TRIM, "pitch uses shifted only")


def check_stretched_value_unused(solve):
    _near(C(solve, "pitch_shift", stretched=[1.0] * 5),
          PITCH_DEF, "stretched is an injected-only input")
