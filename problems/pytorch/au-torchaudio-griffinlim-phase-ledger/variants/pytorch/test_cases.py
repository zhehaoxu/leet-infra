#!/usr/bin/env python3
"""860 judge cases: torchaudio Griffin-Lim phase-recovery ledger.

Every check takes `solve` and asserts on explicit, hand-verified values taken
from the verbatim official implementation (see notes.md).  No torch, no import
from the solution module, no global RNG.
"""
import math

SPEC = [[1.0, 0.5, 0.25], [0.5, 0.25, 0.125], [0.25, 0.125, 0.0625],
        [0.125, 0.0625, 0.03125], [0.0625, 0.03125, 0.015625]]
ZERO = [[0.0] * 3 for _ in SPEC]
RECT = [1.0] * 8
HANN = [0.5 - 0.5 * math.cos(2 * math.pi * i / 8) for i in range(8)]
HANN6 = [0.5 - 0.5 * math.cos(2 * math.pi * i / 6) for i in range(6)]
SYM_HANN = [0.5 - 0.5 * math.cos(2 * math.pi * i / 7) for i in range(8)]
WF = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
LIN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
GEOM = dict(n_fft=8, hop_length=4, win_length=8, window=RECT)


def _g(**kw):
    d = dict(GEOM)
    d.update(kw)
    return d

STFT_RECT_R = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [4.0, 4.0, 0.0],
               [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
STFT_RECT_I = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [-0.0, -0.0, -0.0],
               [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]]
STFT_NOC_R = [[36.0], [-3.9999999999999996], [-4.0], [-4.0], [-4.0]]
STFT_NOC_I = [[0.0], [9.65685424949238], [4.0], [1.6568542494923797], [0.0]]
STFT_HANN_R = [[8.585786437626906, 20.0, 26.414213562373096],
               [-2.585786437626905, -10.0, -14.414213562373096],
               [-2.0, -8.881784197001252e-16, 1.0],
               [0.5857864376269049, 0.0, 0.4142135623730949],
               [-0.585786437626906, 0.0, -0.4142135623730958]]
STFT_HANN_I = [[0.0, 0.0, 0.0],
               [1.3111411434437774e-15, 3.8284271247461925,
                -2.414213562373093],
               [4.440892098500626e-16, -0.828427124746189, 1.414213562373095],
               [-2.1126486106410307e-17, -0.17157287525380926,
                -0.4142135623730949], [0.0, 0.0, 0.0]]
STFT_W6_R = [[5.5, 15.0, 20.5],
             [-3.121320343559643, -10.303300858899107, -14.424621202458749],
             [-0.49999999999999956, 2.500000000000001, 4.000000000000001],
             [1.1213203435596428, 0.3033008588991075, 0.42462120245874946],
             [-0.5000000000000004, -2.6645352591003757e-15, -0.5]]
STFT_W6_I = [[0.0, 0.0, 0.0],
             [6.010984557184401e-16, 2.0606601717798227, -1.5606601717798188],
             [-6.661338147750939e-16, -1.5000000000000022, 1.4999999999999973],
             [6.010984557184401e-16, 0.06066017177982319, -0.5606601717798192],
             [0.0, 0.0, 0.0]]
STFT_NORM_R = [[8.48528137423857, 12.727922061357855, 16.97056274847714],
               [2.414213562373095, -1.4142135623730947, -1.7071067811865472],
               [0.0, -1.414213562373095, 0.0],
               [0.4142135623730951, -1.414213562373095, 0.29289321881345254],
               [0.0, -1.414213562373095, 0.0]]
STFT_NORM_I = [[0.0, 0.0, 0.0],
               [-7.850462293418875e-17, 3.4142135623730945,
                -1.7071067811865472],
               [-0.0, 1.414213562373095, -0.0],
               [-7.850462293418875e-17, 0.5857864376269047,
                -0.29289321881345237], [0.0, 0.0, 0.0]]

ISTFT_RECT = [0.107421875, 0.0713178098159403, 0.052734375,
              0.1044634401840597, 0.0537109375, 0.03565890490797015,
              0.0263671875, 0.05223172009202985]
ISTFT_NOC = [0.3515625, 0.1834787607362388, 0.0703125, 0.05089623926376116,
             0.107421875, 0.0713178098159403, 0.052734375,
             0.1044634401840597, 0.0537109375, 0.03565890490797015,
             0.0263671875, 0.05223172009202985, 0.009765625,
             0.012724059815940294, 0.017578125, 0.04586969018405971]
ISTFT_L5 = [0.107421875, 0.0713178098159403, 0.052734375, 0.1044634401840597,
            0.0537109375]
ISTFT_L12 = [0.107421875, 0.0713178098159403, 0.052734375, 0.1044634401840597,
             0.0537109375, 0.03565890490797015, 0.0263671875,
             0.05223172009202985, 0.009765625, 0.012724059815940294,
             0.017578125, 0.04586969018405971]
ISTFT_L20 = ISTFT_L12 + [0.0] * 8
ISTFT_HANN = [0.0390625, 0.07583677172801989, 0.10546875,
              0.06478822827198011, 0.01953125, 0.037918385864009946,
              0.052734375, 0.032394114135990054]
ISTFT_W6 = [0.0390625, 0.06786165235168154, 0.21093750000000003,
            0.03393082617584078, 0.01953125, 0.03393082617584077,
            0.10546875000000001, 0.01696541308792039]
ISTFT_NORM = [0.3038349450410946, 0.2017172277608956, 0.14915533665653735,
              0.2954672277608956, 0.1519174725205473, 0.1008586138804478,
              0.07457766832826868, 0.1477336138804478]
ISTFT_IMAG = [0.107421875, -0.0036821901840597, 0.052734375,
              0.1794634401840597, 0.0537109375, -0.039341095092029855,
              0.0263671875, 0.12723172009202988]

INV_P1 = [0.0713178098159403, 0.052734375, 0.1044634401840597,
          0.0537109375, 0.03565890490797015, 0.0263671875,
          0.05223172009202985, 0.009765625]
INV_P2 = [0.052734375, 0.1044634401840597, 0.0537109375,
          0.03565890490797015, 0.0263671875, 0.05223172009202985,
          0.009765625, 0.012724059815940294]

GL_N0 = ISTFT_RECT
GL_N1 = [0.23087676036166155, 0.1518955550209224, 0.06776380339174468,
         0.09255309552717414, 0.0453451506833091, 0.006654104867545875,
         -0.0021888394367162497, 0.028194119584357538]
GL_N2 = [0.24646313878715653, 0.16097387545195785, 0.06199009419024969,
         0.06572711957677199, 0.047129605117462864, 0.011480723200897951,
         0.0018390369051308232, 0.03330265677037203]
GL_M05 = [0.24970947585732106, 0.1629758985220881, 0.057270792135068,
          0.05519008710389206, 0.04185675008986525, 0.01624815580651027,
          0.00858485691774559, 0.03707023356750943]
GL_M09 = [0.260008759853775, 0.1518366119606848, 0.06361387657495704,
          0.04245935381527341, 0.03921251445032841, 0.019936725111379068,
          0.010211724120939398, 0.04162668411266284]
GL_P2 = [0.4588054176799241, 0.15606272365034304, 0.04439463847753853,
         0.05544818529841157, 0.09921125627111294, -0.018845381523194912,
         -0.014482921835301941, 0.058444907134838325]
GL_HANN = [0.14843750000000006, 0.2852074104023117, 0.32131884207996375,
           0.16123235793704738, 0.0608946485598011, 0.09093499927183059,
           0.06339617512056968, 0.03393862785511056]
GL_W6 = [0.1640625, 0.30356391274719735, 0.6769607065584581,
         0.15993227767040896, 0.0812665181293516, 0.11181679460295331,
         0.1476480479407766, 0.046787999461105664]
GL_RAND = [0.16818969227961006, 0.14388270629721334, 0.1559266822535506,
           0.04730242593994229, -0.018910446779905082, 0.01474029563955172,
           0.06784094724674421, 0.07337144712329262]
ANGLES7 = [[[0.279380429906404, 0.27369371365176565],
            [0.8620929614176255, 0.6566884305882014],
            [0.9225293674529872, 0.8395453757623866]],
           [[0.2947121080128837, 0.5607215973721555],
            [0.7630434112121387, 0.7593572272132831],
            [0.06302112045357511, 0.0575558332630387]],
           [[0.16744060179820508, 0.03763374058266933],
            [0.1221898423754062, 0.8475435705335523],
            [0.013204739557714507, 0.9498624465599965]],
           [[0.10993588570774804, 0.9936169389547],
            [0.9241920732946692, 0.8652042633507657],
            [0.6603051061881838, 0.30734958208251295]],
           [[0.8948998153826417, 0.7310225719574913],
            [0.9489609645126155, 0.8393899714828684],
            [0.13912210147648818, 0.9578552725190992]]]


def _near(got, exp, tol=1e-9):
    if isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), (got, exp)
        for a, b in zip(got, exp):
            _near(a, b, tol)
    elif isinstance(exp, float):
        assert isinstance(got, (int, float)) and abs(got - exp) <= tol, \
            (got, exp)
    else:
        assert got == exp, (got, exp)


def _err(fn, key):
    try:
        fn()
    except Exception as e:
        assert key in str(e), "expected %r in %r" % (key, str(e))
        return
    raise AssertionError("expected an error containing %r" % key)


# ------------------------------------------------------------------ norms/dtype
def check_norms_mapping(solve):
    assert solve("norms", normalized=False) == {
        "frame_length_norm": False, "window_norm": False}
    assert solve("norms", normalized=True) == {
        "frame_length_norm": False, "window_norm": True}
    assert solve("norms", normalized="window") == {
        "frame_length_norm": False, "window_norm": True}
    assert solve("norms", normalized="frame_length") == {
        "frame_length_norm": True, "window_norm": False}
    assert solve("norms") == solve("norms", normalized=False)
    assert solve("norms", normalized=True) == solve("norms",
                                                    normalized="window")


def check_norms_errors(solve):
    _err(lambda: solve("norms", normalized="bogus"),
         "Invalid normalized parameter")
    _err(lambda: solve("norms", normalized=1), "Input type not supported")
    _err(lambda: solve("norms", normalized=None), "Input type not supported")


def check_complex_dtype(solve):
    assert solve("complex_dtype", real_dtype="float64")["complex_dtype"] == \
        "torch.complex128"
    assert solve("complex_dtype", real_dtype="double")["complex_dtype"] == \
        "torch.complex128"
    assert solve("complex_dtype", real_dtype="float32")["complex_dtype"] == \
        "torch.complex64"
    assert solve("complex_dtype", real_dtype="float")["complex_dtype"] == \
        "torch.complex64"
    assert solve("complex_dtype", real_dtype="float16")["complex_dtype"] == \
        "torch.complex32"
    assert solve("complex_dtype", real_dtype="half")["complex_dtype"] == \
        "torch.complex32"
    assert solve("complex_dtype")["real_dtype"] == "float64"


def check_complex_dtype_errors(solve):
    _err(lambda: solve("complex_dtype", real_dtype="int64"),
         "Unexpected dtype")
    _err(lambda: solve("complex_dtype", real_dtype=3),
         "real_dtype must be a str")


# ------------------------------------------------------------------ stft
def check_stft_reflect_rect(solve):
    r = solve("stft", waveform=WF, **_g())
    assert r["shape"] == [5, 3]
    assert r["n_frames"] == 3 and r["padded_length"] == 16
    assert r["center"] is True and r["normalized"] is False
    _near(r["real"], STFT_RECT_R)
    _near(r["imag"], STFT_RECT_I)
    # The DC row of a reflect-padded linear ramp must be [24, 36, 48].
    r2 = solve("stft", waveform=LIN, **_g())
    _near(r2["real"][0], [24.0, 36.0, 48.0])


def check_stft_center_false(solve):
    r = solve("stft", waveform=LIN, center=False, **_g())
    assert r["shape"] == [5, 1] and r["padded_length"] == 8
    _near(r["real"], STFT_NOC_R)
    _near(r["imag"], STFT_NOC_I)


def check_stft_hann(solve):
    r = solve("stft", waveform=LIN, **_g(window=HANN))
    _near(r["real"], STFT_HANN_R)
    _near(r["imag"], STFT_HANN_I)
    assert r["shape"] == [5, 3]


def check_stft_short_window(solve):
    r = solve("stft", waveform=LIN, window=HANN6, win_length=6,
              n_fft=8, hop_length=4)
    _near(r["real"], STFT_W6_R)
    _near(r["imag"], STFT_W6_I)
    assert r["win_length"] == 6


def check_stft_normalized(solve):
    r = solve("stft", waveform=LIN, normalized=True, **_g())
    _near(r["real"], STFT_NORM_R)
    _near(r["imag"], STFT_NORM_I)
    u = solve("stft", waveform=LIN, **_g())
    _near(u["real"], [[v * math.sqrt(8) for v in row]
                      for row in r["real"]])


def check_stft_shapes_and_errors(solve):
    assert solve("stft", waveform=LIN, hop_length=2, n_fft=8,
                 win_length=8, window=RECT)["shape"] == [5, 5]
    assert solve("stft", waveform=[math.sin(i * 0.7) for i in range(16)],
                 n_fft=16, hop_length=8, win_length=16,
                 window=[1.0] * 16)["shape"] == [9, 3]
    _err(lambda: solve("stft", waveform=[], **_g()), "waveform")
    _err(lambda: solve("stft", waveform=[True] * 8, **_g()), "waveform")
    _err(lambda: solve("stft", waveform=[1.0] * 4, **_g()), "reflect pad")
    _err(lambda: solve("stft", waveform=LIN, n_fft=1, hop_length=1,
                       win_length=1, window=[1.0]), "n_fft")
    _err(lambda: solve("stft", waveform=LIN, win_length=9,
                       n_fft=8, hop_length=4, window=[1.0] * 9),
         "win_length must be <= n_fft")
    _err(lambda: solve("stft", waveform=LIN, **_g(hop_length=9)),
         "hop_length must be <= win_length")
    _err(lambda: solve("stft", waveform=LIN, **_g(window=[1.0] * 7)),
         "window length must equal win_length")
    _err(lambda: solve("stft", waveform=LIN, center="yes", **_g()),
         "center must be a bool")


# ------------------------------------------------------------------ istft
def check_istft_rect_ola(solve):
    r = solve("istft", real=SPEC, imag=ZERO, length=8, **_g())
    _near(r["out"], ISTFT_RECT)
    assert r["shape"] == [8] and r["target_length"] == 8
    assert r["envelope_min"] == 2.0
    assert r["window_l2"] == math.sqrt(8.0)
    u = solve("istft", real=SPEC, imag=ZERO, **_g())
    _near(u["out"], ISTFT_RECT)


def check_istft_center_false(solve):
    r = solve("istft", real=SPEC, imag=ZERO, center=False, length=16, **_g())
    _near(r["out"], ISTFT_NOC)
    u = solve("istft", real=SPEC, imag=ZERO, center=False, **_g())
    _near(u["out"], ISTFT_NOC)
    assert u["target_length"] == 16


def check_istft_lengths(solve):
    _near(solve("istft", real=SPEC, imag=ZERO, length=5, **_g())["out"],
          ISTFT_L5)
    _near(solve("istft", real=SPEC, imag=ZERO, length=12, **_g())["out"],
          ISTFT_L12)
    _near(solve("istft", real=SPEC, imag=ZERO, length=20, **_g())["out"],
          ISTFT_L20)
    assert solve("istft", real=SPEC, imag=ZERO, length=20,
                 **_g())["out"][-4:] == [0.0] * 4


def check_istft_hann_and_short_window(solve):
    _near(solve("istft", real=SPEC, imag=ZERO, length=8,
                **_g(window=HANN))["out"], ISTFT_HANN)
    r = solve("istft", real=SPEC, imag=ZERO, window=HANN6, win_length=6,
              n_fft=8, hop_length=4, length=8)
    _near(r["out"], ISTFT_W6)
    assert r["window_l2"] == math.sqrt(sum(v * v for v in HANN6))


def check_istft_normalized(solve):
    _near(solve("istft", real=SPEC, imag=ZERO, normalized=True, length=8,
                **_g())["out"], ISTFT_NORM)
    u = solve("istft", real=SPEC, imag=ZERO, length=8, **_g())
    _near(ISTFT_NORM, [v * math.sqrt(8) for v in u["out"]])


def check_istft_imag_dc_nyquist(solve):
    im = [[0.3] * 3 for _ in SPEC]
    im[0][0] = 0.7
    im[4][0] = 0.9
    _near(solve("istft", real=SPEC, imag=im, length=8, **_g())["out"],
          ISTFT_IMAG)
    _near(solve("istft", real=[[0.0] * 3 for _ in SPEC],
                imag=[[0.0] * 3 for _ in SPEC], length=8,
                **_g())["out"], [0.0] * 8)


def check_istft_nola(solve):
    _err(lambda: solve("istft", real=SPEC, imag=ZERO, length=12,
                       **_g(window=SYM_HANN)), "window overlap add min")
    assert len(solve("istft", real=SPEC, imag=ZERO,
                     **_g(window=SYM_HANN))["out"]) == 8


def check_istft_errors(solve):
    _err(lambda: solve("istft", real=SPEC[:4], imag=ZERO[:4], **_g()),
         "freq dimension must be n_fft // 2 + 1")
    _err(lambda: solve("istft", real=SPEC, imag=[[0.0] * 2 for _ in SPEC],
                       **_g()), "real and imag shapes must agree")
    _err(lambda: solve("istft", real=SPEC, imag=ZERO, length=-1, **_g()),
         "length must be >= 0")
    _err(lambda: solve("istft", real=SPEC, imag=ZERO, normalized="x", **_g()),
         "normalized must be a bool")
    _err(lambda: solve("istft", real=[], imag=ZERO, **_g()), "real")


# ------------------------------------------------------------------ inverse
def check_inverse_norms(solve):
    base = solve("inverse", real=SPEC, imag=ZERO, length=8, **_g())
    _near(base["out"], ISTFT_RECT)
    for v in (True, "window", "frame_length"):
        r = solve("inverse", real=SPEC, imag=ZERO, normalized=v,
                  length=8, **_g())
        _near(r["out"], [x * math.sqrt(8) for x in ISTFT_RECT])
    assert solve("inverse", real=SPEC, imag=ZERO, normalized=True, length=8,
                 **_g())["window_norm"] is True
    r = solve("inverse", real=SPEC, imag=ZERO, normalized="frame_length",
              length=8, **_g())
    assert r["frame_length_norm"] is True and r["window_norm"] is False
    assert r["window_scale"] == 1.0


def check_inverse_pad(solve):
    _near(solve("inverse", real=SPEC, imag=ZERO, pad=1, length=8,
                **_g())["out"], INV_P1)
    _near(solve("inverse", real=SPEC, imag=ZERO, pad=2, length=8,
                **_g())["out"], INV_P2)
    r = solve("inverse", real=SPEC, imag=ZERO, pad=2, length=8, **_g())
    assert r["istft_length"] == 12 and r["pad"] == 2
    assert r["target_length"] == 8


def check_inverse_pad_without_length(solve):
    r = solve("inverse", real=SPEC, imag=ZERO, pad=2, **_g())
    _near(r["out"], ISTFT_RECT)
    assert r["istft_length"] is None
    _near(r["out"], solve("inverse", real=SPEC, imag=ZERO, pad=0,
                          **_g())["out"])


def check_inverse_center_false(solve):
    r = solve("inverse", real=SPEC, imag=ZERO, center=False, length=8, **_g())
    _near(r["out"], ISTFT_NOC[:8])
    assert r["target_length"] == 8


def check_inverse_metadata(solve):
    r = solve("inverse", real=SPEC, imag=ZERO, length=8, **_g())
    assert r["shape"] == [8] and r["target_length"] == 8
    assert r["frame_length_norm"] is False and r["window_norm"] is False
    assert r["window_scale"] == 1.0 and r["istft_length"] == 8
    assert r["envelope_min"] == 2.0
    w = solve("inverse", real=SPEC, imag=ZERO, normalized="window",
              length=8, **_g())
    assert w["window_scale"] == math.sqrt(8.0)


def check_inverse_errors(solve):
    _err(lambda: solve("inverse", real=SPEC, imag=ZERO, normalized="bogus",
                       **_g()), "Invalid normalized parameter")
    _err(lambda: solve("inverse", real=SPEC, imag=ZERO, pad=-1, **_g()),
         "pad must be >= 0")
    _err(lambda: solve("inverse", real=SPEC[:4], imag=ZERO[:4], **_g()),
         "freq dimension must be n_fft // 2 + 1")
    _err(lambda: solve("inverse", real=SPEC, imag=ZERO, length=2.5, **_g()),
         "length must be an int")


# ------------------------------------------------------------------ griffinlim
def _gl(solve, **kw):
    cfg = dict(specgram=SPEC, **_g())
    cfg.update(kw)
    return solve("griffinlim", **cfg)


def check_griffinlim_zero_iter(solve):
    r = _gl(solve, n_iter=0, length=8)
    _near(r["out"], GL_N0)
    assert r["n_iter"] == 0 and r["n_frames"] == 3
    assert r["momentum_effective"] == 0.0 and r["power"] == 1.0
    assert r["shape"] == [8] and r["target_length"] == 8


def check_griffinlim_iterations(solve):
    _near(_gl(solve, n_iter=1, length=8)["out"], GL_N1)
    _near(_gl(solve, n_iter=2, length=8)["out"], GL_N2)
    assert _gl(solve, n_iter=2, length=8)["envelope_min"] == 2.0


def check_griffinlim_momentum(solve):
    r = _gl(solve, n_iter=3, momentum=0.5, length=8)
    _near(r["out"], GL_M05)
    assert abs(r["momentum_effective"] - 1.0 / 3.0) < 1e-12
    r2 = _gl(solve, n_iter=4, momentum=0.9, length=8)
    _near(r2["out"], GL_M09)
    assert abs(r2["momentum_effective"] - 0.9 / 1.9) < 1e-12


def check_griffinlim_power(solve):
    r = _gl(solve, n_iter=2, power=2.0, length=8)
    _near(r["out"], GL_P2)
    assert r["power"] == 2.0


def check_griffinlim_angles_init(solve):
    r = _gl(solve, n_iter=2, momentum=0.3, length=8,
            angles_init=ANGLES7)
    _near(r["out"], GL_RAND)
    assert abs(r["momentum_effective"] - 0.3 / 1.3) < 1e-12
    _err(lambda: _gl(solve, angles_init=[[[1.0, 0.0]] * 3] * 4),
         "angles_init must have shape")
    _err(lambda: _gl(solve, angles_init=[[1.0] * 3] * 5),
         "angles_init entries must be")


def check_griffinlim_windows_and_length(solve):
    _near(_gl(solve, n_iter=2, window=HANN, length=8)["out"], GL_HANN)
    _near(_gl(solve, n_iter=2, window=HANN6, win_length=6,
              length=8)["out"], GL_W6)
    _near(_gl(solve, n_iter=2)["out"], GL_N2)
    assert _gl(solve, n_iter=2)["target_length"] == 8


def check_griffinlim_errors(solve):
    _err(lambda: _gl(solve, momentum=1.0), "momentum must be in range")
    _err(lambda: _gl(solve, momentum=-0.1), "momentum must be in range")
    _err(lambda: _gl(solve, power=0.0), "power must be > 0")
    _err(lambda: _gl(solve, n_iter=-1), "n_iter must be >= 0")
    _err(lambda: _gl(solve, specgram=SPEC[:4]),
         "freq dimension must be n_fft // 2 + 1")
    _err(lambda: _gl(solve, length=12, hop_length=2),
         "rebuilt frame count does not match specgram")


def check_unknown_mode(solve):
    _err(lambda: solve("bogus"), "unknown mode")


# ------------------------------------------------------------------ hardening
W5 = [0.5 - 0.5 * math.cos(2 * math.pi * i / 5) for i in range(5)]
W5_R = [[4.4409830056250525, 11.25, 18.05901699437495],
        [-2.672275450153843, -8.546675160832244, -12.914028523215656],
        [-0.1319660112501051, 3.4860679774997894, 3.9131189606246326],
        [0.8632584557788954, -0.49840981104249327, 0.2509095625910227],
        [-0.5590169943749475, -0.13196601125010687, -0.5590169943749483]]
W5_I = [[0.0, 0.0, 0.0],
        [-1.8270439234796734, -2.1290145695596303, -6.0693169488182],
        [1.1180339887498947, 1.5450849718747364, 5.163118960624631],
        [0.24590509339548428, -0.056065552684472886, -1.2324359094428314],
        [0.0, 0.0, 0.0]]
W5_ISTFT = [0.043186437851565784, 0.14731545884466785, 0.10175720593359079,
            0.028134749105188404, 0.021593218925782892,
            0.07365772942233392, 0.050878602966795396, 0.014067374552594202]
GL_T4 = [0.24874549042124966, 0.1597967813700037, 0.05976610751009888,
         0.0629180431137483, 0.04609129808180874, 0.028721090325451646,
         0.023675620222252683, 0.05578849224876123, 0.02460372631203036,
         0.0037506788875213345, -0.0015150550474403949, 0.012267101554513707]
GL_SPARSE = [0.125] * 8
INV_IMAG = [0.107421875, 0.033300042862973916, 0.052734375,
            0.13894567323109333, 0.0537109375, -0.004858862044996224,
            0.026367187499999997, 0.0892139531390635]
IM_MAT = [[0.1 + 0.02 * k + 0.01 * t for t in range(3)] for k in range(5)]
SP4 = [row + [row[-1] * 0.5] for row in SPEC]
SPARSE = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def _err_kind(fn, kind, key):
    try:
        fn()
    except Exception as e:
        assert isinstance(e, kind), (type(e).__name__, str(e))
        assert key in str(e), str(e)
        return
    raise AssertionError("expected %s containing %r" % (kind.__name__, key))


def check_hardening_nola_threshold(solve):
    """A 1e-6 window gives an envelope minimum of 2e-12: below 1e-11."""
    _err(lambda: solve("istft", real=SPEC, imag=ZERO, length=8,
                       **_g(window=[1e-6] * 8)), "window overlap add min")
    r = solve("istft", real=SPEC, imag=ZERO, length=8,
              **_g(window=[1e-3] * 8))
    _near(r["out"], [v * 1000.0 for v in ISTFT_RECT])
    assert abs(r["envelope_min"] - 2e-6) < 1e-12


def check_hardening_winlen5(solve):
    """Odd (n_fft - win_length) splits the zero padding 1/2."""
    g = dict(n_fft=8, hop_length=4, win_length=5, window=W5)
    r = solve("stft", waveform=LIN, **g)
    _near(r["real"], W5_R)
    _near(r["imag"], W5_I)
    _near(solve("istft", real=SPEC, imag=ZERO, length=8, **g)["out"],
          W5_ISTFT)


def check_hardening_defaults(solve):
    a = solve("stft", waveform=LIN, hop_length=2, win_length=2,
              window=[1.0] * 2)
    assert a["shape"] == [5, 5]
    _near(a["real"][0], [3.0, 5.0, 9.0, 13.0, 15.0])
    b = solve("stft", waveform=LIN, n_fft=8, hop_length=4,
              window=[1.0] * 8)
    assert b["shape"] == [5, 3]
    _near(b["real"][0], [24.0, 36.0, 48.0])
    c = solve("stft", waveform=LIN, n_fft=8, win_length=8,
              window=[1.0] * 8)
    assert c["shape"] == [5, 5]
    _near(c["real"][0], [24.0, 26.0, 36.0, 46.0, 48.0])
    d = solve("stft", waveform=LIN, n_fft=8, hop_length=4, win_length=8)
    assert d["shape"] == [5, 3]
    _near(d["real"][0], [24.0, 36.0, 48.0])


def check_hardening_error_messages(solve):
    _err(lambda: solve("stft", waveform=[], **_g()),
         "waveform must be a non-empty list of numbers")
    _err(lambda: solve("stft", waveform=[True] * 8, **_g()),
         "waveform must contain only numbers")
    _err(lambda: solve("stft", waveform=LIN, **_g(n_fft=True)),
         "n_fft must be an int")
    _err(lambda: solve("stft", waveform=LIN, **_g(win_length=0)),
         "win_length must be >= 1")
    _err(lambda: solve("stft", waveform=LIN, **_g(hop_length=0)),
         "hop_length must be >= 1")
    _err(lambda: solve("stft", waveform=[1.0] * 6, center=False, **_g()),
         "waveform shorter than n_fft")
    _err(lambda: solve("griffinlim", specgram=SPEC, power="x", **_g()),
         "power must be a number")
    _err(lambda: solve("istft", real=[], imag=ZERO, **_g()),
         "real must be a non-empty 2-D list")
    _err(lambda: solve("istft", real=[[]], imag=ZERO, **_g()),
         "real must have at least one column")
    _err(lambda: solve("istft", real=[[1.0, 2.0, 3.0], [1.0, 2.0]],
                       imag=ZERO, **_g()), "real must be rectangular")
    _err(lambda: solve("istft", real=[["x"] * 3 for _ in SPEC],
                       imag=ZERO, **_g()), "real must contain only numbers")
    _err(lambda: solve("inverse", real=SPEC, imag=ZERO, length=-1, **_g()),
         "length must be >= 0")
    _err(lambda: _gl(solve, length=-1), "length must be >= 0")


def check_hardening_error_types(solve):
    _err_kind(lambda: solve("norms", normalized=1), TypeError,
              "Input type not supported")
    _err_kind(lambda: solve("norms", normalized=None), TypeError,
              "Input type not supported")


def check_hardening_missing_args(solve):
    _err(lambda: solve("stft", **_g()),
         "waveform must be a non-empty list of numbers")
    _err(lambda: solve("istft", imag=ZERO, **_g()),
         "real must be a non-empty 2-D list")
    _err(lambda: solve("istft", real=SPEC, **_g()),
         "imag must be a non-empty 2-D list")
    _err(lambda: solve("inverse", imag=ZERO, **_g()),
         "real must be a non-empty 2-D list")
    _err(lambda: solve("inverse", real=SPEC, **_g()),
         "imag must be a non-empty 2-D list")
    _err(lambda: solve("griffinlim", **_g()),
         "specgram must be a non-empty 2-D list")
    _err(lambda: solve("inverse", real=SPEC,
                       imag=[[0.0] * 2 for _ in SPEC], **_g()),
         "real and imag shapes must agree")


def check_hardening_metadata(solve):
    r = solve("stft", waveform=LIN, **_g())
    assert r["n_fft"] == 8 and r["hop_length"] == 4
    assert r["win_length"] == 8 and r["center"] is True
    assert r["normalized"] is False
    n = solve("stft", waveform=LIN, normalized=True, **_g())
    assert n["normalized"] is True
    d = solve("complex_dtype", real_dtype="half")
    assert d["real_dtype"] == "half"
    assert d["complex_dtype"] == "torch.complex32"
    g = _gl(solve, n_iter=3, momentum=0.5, length=8)
    assert g["n_iter"] == 3 and g["n_frames"] == 3
    assert g["shape"] == [8] and g["target_length"] == 8
    i = solve("inverse", real=SPEC, imag=ZERO, length=8, **_g())
    assert i["shape"] == [8] and i["target_length"] == 8


def check_hardening_inverse_imag(solve):
    r = solve("inverse", real=SPEC, imag=IM_MAT, length=8, **_g())
    _near(r["out"], INV_IMAG)
    assert r["envelope_min"] == 2.0
    w = solve("inverse", real=SPEC, imag=IM_MAT, normalized="window",
              length=8, **_g())
    _near(w["out"], [v * math.sqrt(8) for v in INV_IMAG])


def check_hardening_gl_frames(solve):
    _near(_gl(solve)["out"], GL_N2)
    r = _gl(solve, specgram=SP4, n_iter=2)
    _near(r["out"], GL_T4)
    assert r["target_length"] == 12 and r["n_frames"] == 4
    assert _gl(solve, n_iter=1, length=8)["n_iter"] == 1


def check_hardening_gl_sparse(solve):
    r = _gl(solve, specgram=SPARSE, n_iter=2, length=8)
    _near(r["out"], GL_SPARSE)
    assert r["envelope_min"] == 2.0


def check_hardening_angles_shape(solve):
    _err(lambda: _gl(solve, angles_init=[[[1.0, 0.0]] * 2] * 5),
         "angles_init must have shape [freq, frames]")
    _err(lambda: _gl(solve, angles_init=[[[1.0]] * 3] * 5),
         "angles_init entries must be [re, im] pairs")
    _err(lambda: _gl(solve, angles_init=[[["x", 0.0]] * 3] * 5),
         "angles_init entries must be [re, im] pairs")


def check_hardening_winlen_default_n16(solve):
    """win_length defaults to n_fft, not to the literal 8."""
    wf = [float(i + 1) for i in range(16)]
    r = solve("stft", waveform=wf, n_fft=16, hop_length=8,
              window=[1.0] * 16)
    assert r["shape"] == [9, 3]
    _near(r["real"][0], [80.0, 136.0, 192.0])
    assert r["win_length"] == 16


def check_hardening_zero_phase(solve):
    """An all-zero initial phase makes rebuilt magnitudes exactly zero;
    the official +1e-16 divisor keeps the iteration finite."""
    zero_angles = [[[0.0, 0.0] for _ in range(3)] for _ in SPEC]
    r = _gl(solve, n_iter=2, length=8, angles_init=zero_angles)
    _near(r["out"], [0.0] * 8)
    assert r["envelope_min"] == 2.0
