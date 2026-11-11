#!/usr/bin/env python3
"""859 judge cases: torchaudio sinc-resampling ledger.

Every check takes `solve` and asserts on explicit, hand-verified values taken
from the verbatim official implementation (see notes.md).  No torch, no import
from the solution module, no global RNG.
"""
import math

WF = [1., 0., -1., 0., 1., 0., -1., 0.]
W11 = [math.sin(i * 0.7) for i in range(11)]
WB = [WF, [v * 0.5 for v in WF]]
W3 = [[WF, WB[1]], [WF, WF]]

K84 = [[-7.234822658643318e-50, -1.226306971405509e-06,
        -0.0005975946585144695, 0.0003668864744661114,
        0.005422914672238354, -0.0012922085819558737,
        -0.017151031358732645, 0.0025355107264958595,
        0.04034644726557991, -0.0037701217556779898,
        -0.09075743127027464, 0.004670810380106644,
        0.312955514012274, 0.495, 0.312955514012274,
        0.004670810380106644, -0.09075743127027464,
        -0.0037701217556779898, 0.04034644726557991,
        0.0025355107264958595, -0.017151031358732645,
        -0.0012922085819558737, 0.005422914672238354,
        0.0003668864744661114, -0.0005975946585144695,
        -1.226306971405509e-06, -7.234822658643318e-50,
        -7.234822658643318e-50]]
KK84 = [[-7.096615577065457e-23, -4.4196508172410285e-08,
         -8.493700717316224e-06, 1.0835479076694143e-05,
         0.0003277348383496393, -0.00014648058283526128,
         -0.003329125546552203, 0.0007735241634293603,
         0.0178659671796759, -0.002249625832757898,
         -0.06802554339938416, 0.0041118020160471475,
         0.3031694134164198, 0.49500000944153827,
         0.3031694134164198, 0.0041118020160471475,
         -0.06802554339938416, -0.002249625832757898,
         0.0178659671796759, 0.0007735241634293603,
         -0.003329125546552203, -0.00014648058283526128,
         0.0003277348383496393, 1.0835479076694143e-05,
         -8.493700717316224e-06, -4.4196508172410285e-08,
         -7.096615577065457e-23, -7.096615577065457e-23]]
K48 = [
    [-1.4469645317286637e-49, -2.452613942811018e-06,
     0.0007337729489322228, -0.0025844171639117473,
     0.005071021452991719, -0.0075402435113559796,
     0.009341620760213288, 0.99, 0.009341620760213288,
     -0.0075402435113559796, 0.005071021452991719,
     -0.0025844171639117473, 0.0007337729489322228,
     -2.452613942811018e-06, -1.4469645317286637e-49],
    [-1.4469645317286637e-49, -1.4469645317286637e-49,
     -0.001195189317028939, 0.010845829344476709,
     -0.03430206271746529, 0.08069289453115983,
     -0.18151486254054927, 0.625911028024548,
     0.625911028024548, -0.18151486254054927,
     0.08069289453115983, -0.03430206271746529,
     0.010845829344476709, -0.001195189317028939,
     -1.4469645317286637e-49]]
K69 = [
    [-1.1692642680635668e-49, -0.00446656387869955,
     0.034863627179566474, -0.09906990495878168,
     0.16923160124292963, 0.8, 0.16923160124292963,
     -0.09906990495878168, 0.034863627179566474,
     -0.00446656387869955, -1.1692642680635668e-49,
     -1.1692642680635668e-49],
    [-1.1692642680635668e-49, -0.0005538420623003991,
     0.002985961522349498, 0.021737844879665916,
     -0.1240490014699032, 0.45432279067915676,
     0.7018972455751414, -0.04142386579389932,
     -0.030643160665053872, 0.020674833578317182,
     -0.0031579132775057797, -1.1692642680635668e-49],
    [-1.1692642680635668e-49, -1.1692642680635668e-49,
     -0.0031579132775057797, 0.020674833578317228,
     -0.03064316066505402, -0.04142386579389916,
     0.7018972455751414, 0.4543227906791566,
     -0.12404900146990322, 0.021737844879665985,
     0.002985961522349498, -0.0005538420623003991]]
K84B3 = [[-3.953446106398387e-18, -0.0010644785354061213,
          -0.008399732246881808, 0.00189604309518253,
          0.016463128590303473, -0.0027941245068294875,
          -0.029307638344755263, 0.003653548225922573,
          0.05126216944283959, -0.00436555266661752,
          -0.09830067276134037, 0.00483572233613906,
          0.31564631313721025, 0.4949999967614379,
          0.31564631313721025, 0.00483572233613906,
          -0.09830067276134037, -0.00436555266661752,
          0.05126216944283959, 0.003653548225922573,
          -0.029307638344755263, -0.0027941245068294875,
          0.016463128590303473, 0.00189604309518253,
          -0.008399732246881808, -0.0010644785354061213,
          -3.953446106398387e-18, -3.953446106398387e-18]]

A84_8 = [0.4840235571377195, -0.48188825748410874,
         0.48188825748410874, -0.4840235571377195]
A84_11 = [0.1282439922565054, 0.9538656271040834, 0.36213906649805266,
          -0.9101263168581649, -0.5588309602338636, 0.3940843738013485]
A48_8 = [0.994958278961387, 0.7743190171646609,
         0.0043372485040594965, -0.7375788253784142,
         -1.0024960698588001, -0.6924309333164722, 0.0,
         0.6924309333164722, 1.0024960698588001, 0.7375788253784141,
         -0.0043372485040594965, -0.7743190171646609,
         -0.994958278961387, -0.5560639628378649,
         -0.005004372256153792, 0.1484079891401129]
R69 = [0.03999194323411476, 0.39293558681887514, 0.7995238421803346,
       1.0019402070702361, 0.9556374061112767, 0.7194992110772539,
       0.3344552304100679, -0.12465106592599279, -0.5563426322428952,
       -0.8701892160864573, -0.9935122018795693, -0.9131308495195042,
       -0.6608963883746345, -0.20051898393973056, 0.3496286321146087,
       0.560615335349284, 0.2802250107754695]


def _near(got, exp, tol=1e-8):
    if isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), (got, exp)
        for a, b in zip(got, exp):
            _near(a, b, tol)
    elif isinstance(exp, float):
        assert isinstance(got, (int, float)) and abs(got - exp) <= tol, (got, exp)
    else:
        assert got == exp, (got, exp)


def _err(fn, key):
    try:
        fn()
    except ValueError as e:
        assert key in str(e), "expected %r in %r" % (key, str(e))
        return
    raise AssertionError("expected ValueError containing %r" % key)


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
def check_meta_reduction(solve):
    """gcd reduction, base_freq, width, kernel_length and official 3-D shape."""
    for cfg, exp in [
        (dict(orig_freq=8, new_freq=4),
         (4, 2, 1, 0.99, 13, 28, [1, 1, 28])),
        (dict(orig_freq=4, new_freq=8),
         (4, 1, 2, 0.99, 7, 15, [2, 1, 15])),
        (dict(orig_freq=6, new_freq=9, lowpass_filter_width=4, rolloff=0.8),
         (3, 2, 3, 1.6, 5, 12, [3, 1, 12])),
        (dict(orig_freq=44100, new_freq=48000,
              lowpass_filter_width=2, rolloff=0.9),
         (300, 147, 160, 132.3, 3, 153, [160, 1, 153])),
        (dict(orig_freq=7, new_freq=3, lowpass_filter_width=3, rolloff=0.95),
         (1, 7, 3, 2.8499999999999996, 8, 23, [3, 1, 23])),
    ]:
        r = solve("meta", **cfg)
        assert r["gcd"] == exp[0]
        assert r["orig_freq_reduced"] == exp[1]
        assert r["new_freq_reduced"] == exp[2]
        _near(r["base_freq"], exp[3])
        assert r["width"] == exp[4] and r["kernel_length"] == exp[5]
        assert r["shape"] == exp[6]
        assert r["lowpass_filter_width"] == cfg.get("lowpass_filter_width", 6)
        _near(r["rolloff"], cfg.get("rolloff", 0.99))


def check_meta_rolloff_and_width(solve):
    """base_freq = min(o,n) * rolloff; width = ceil(W * o / base_freq)."""
    # 8->4 reduces to o=2, n=1, so min(o,n) = 1 and base_freq = rolloff.
    for rl, exp_base, exp_width in [(1.0, 1.0, 12), (0.5, 0.5, 24),
                                    (0.8, 0.8, 15)]:
        r = solve("meta", orig_freq=8, new_freq=4, rolloff=rl)
        _near(r["base_freq"], exp_base)
        assert r["width"] == exp_width
        assert r["kernel_length"] == 2 * exp_width + 2
    for W in (1, 2, 6, 10):
        r = solve("meta", orig_freq=8, new_freq=4,
                  lowpass_filter_width=W)
        assert r["width"] == math.ceil(W * 2 / 0.99)
        assert r["lowpass_filter_width"] == W


def check_meta_scale(solve):
    """scale = base_freq / orig_freq_reduced (not new_freq)."""
    _near(solve("meta", orig_freq=8, new_freq=4)["scale"], 0.495)
    _near(solve("meta", orig_freq=4, new_freq=8)["scale"], 0.99)
    _near(solve("meta", orig_freq=6, new_freq=9,
                lowpass_filter_width=4, rolloff=0.8)["scale"], 0.8)


def check_meta_method_ledger(solve):
    """Only the literal sinc_interp_hann takes the Hann branch."""
    for m, dep, rep, branch in [
        ("sinc_interp_hann", False, None, "hann"),
        ("sinc_interp_kaiser", False, None, "kaiser"),
        ("sinc_interpolation", True, "sinc_interp_hann", "kaiser"),
        ("kaiser_window", True, "sinc_interp_kaiser", "kaiser"),
    ]:
        r = solve("meta", orig_freq=8, new_freq=4,
                  resampling_method=m)
        assert r["method"] == m and r["deprecated"] is dep
        assert r["replacement"] == rep and r["window_branch"] == branch
        assert (len(r["warnings"]) == 1) is dep
        if dep:
            assert "deprecated" in r["warnings"][0]
    _err(lambda: solve("meta", orig_freq=8, new_freq=4,
                       resampling_method="bogus"),
         "Invalid resampling method")


def check_meta_beta_ledger(solve):
    """beta is quantised through torch.tensor(float(beta)) -> float32."""
    a = solve("meta", orig_freq=8, new_freq=4,
              resampling_method="sinc_interp_kaiser")
    assert a["beta_default_used"] is True
    _near(a["beta"], 14.76965618133545, 1e-12)
    for b in (3.0, 8.6, 12.0):
        r = solve("meta", orig_freq=8, new_freq=4, beta=b)
        assert r["beta_default_used"] is False
        _near(r["beta"], b, 1e-6) if b == 8.6 else _near(r["beta"], b, 1e-12)
    r = solve("meta", orig_freq=8, new_freq=4, beta=0.1)
    assert r["beta"] != 0.1  # float32 quantisation is observable
    _near(r["beta"], 0.10000000149011612, 1e-15)


def check_meta_errors(solve):
    """Validation mirrors the official guards and the judged input contract."""
    _err(lambda: solve("meta", orig_freq=8.5, new_freq=4),
         "Frequencies must be of integer type")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4.25),
         "Frequencies must be of integer type")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4,
                       lowpass_filter_width=0), "lowpass_filter_width")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4,
                       lowpass_filter_width=-1), "lowpass_filter_width")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4,
                       lowpass_filter_width=1.5), "lowpass_filter_width")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4, rolloff=0),
         "rolloff must be positive")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4, rolloff=-1),
         "rolloff must be positive")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4, beta="x"),
         "beta must be a number")
    _err(lambda: solve("meta", orig_freq=8, new_freq=4,
                       resampling_method=7), "Invalid resampling method")
    _err(lambda: solve("meta", orig_freq=True, new_freq=4),
         "Frequencies must be of integer type")


# ---------------------------------------------------------------------------
# i0
# ---------------------------------------------------------------------------
def check_i0_series(solve):
    """I0 power series against verbatim torch.i0 float64 values."""
    vals = [0., 1., 2., 3., 5., 8.6, 14.769656459379492, 17.]
    exp = [1.0, 1.2660658777520082, 2.279585302336067,
           4.880792585865024, 27.239871823604442,
           750.4611595631659, 271903.6406814566, 2354970.223168293]
    r = solve("i0", values=vals)
    _near(r["i0"], exp, 1e-6)
    assert r["shape"] == [8]
    assert r["i0"][0] == 1.0
    one = solve("i0", values=3.0)
    _near(one["i0"], 4.880792585865024, 1e-12)
    assert one["shape"] == []
    neg = solve("i0", values=[-3.0, 3.0])
    assert neg["i0"][0] == neg["i0"][1]


def check_i0_float32_ledger(solve):
    """float32 column, default beta and its float32 i0 denominator."""
    # The judged float32 column is the exact float64 series re-quantised to
    # float32; torch.i0(float32) uses a Cephes polynomial and differs by a
    # few ULP for some x (1.0 -> 1.2660659551620483, 5.0 -> 27.2398738861084).
    # Only the default beta is bit-identical on both paths, hence judged.
    r = solve("i0", values=[1., 3., 5.])
    _near(r["i0_float32"], [1.2660658359527588, 4.880792617797852,
                            27.239871978759766], 1e-12)
    _near(r["beta_float32"], 14.76965618133545, 1e-12)
    _near(r["beta_default"], 14.769656459379492, 1e-15)
    _near(r["i0_beta_float32"], 271903.5625, 1e-9)
    assert r["i0_beta_float32"] != r["i0"][0]


def check_i0_errors(solve):
    _err(lambda: solve("i0", values=[]), "values")
    _err(lambda: solve("i0", values=["a"]), "values")
    _err(lambda: solve("i0", values=[1.0, None]), "values")


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------
def check_kernel_hann_ledger(solve):
    """t grid, clamp, Hann window, sinc branch and the boundary residue."""
    r = solve("kernel", orig_freq=8, new_freq=4)
    assert r["shape"] == [1, 28] and r["width"] == 13
    assert r["kernel_length"] == 28
    _near(r["scale"], 0.495, 1e-15)
    assert len(r["t"]) == 1 and len(r["t"][0]) == 28
    assert len(r["window"]) == 1 and len(r["sinc"]) == 1
    assert r["t"][0][13] == 0.0
    assert r["window"][0][13] == 1.0 and r["sinc"][0][13] == 1.0
    _near(r["kernels"][0][13], 0.495, 1e-15)
    # clamp saturates the first tap at -W, and cos(pi/2)**2 is not exactly 0
    assert r["t"][0][0] == -6.0
    assert r["window"][0][0] == math.cos(math.pi / 2) ** 2
    assert 0.0 < r["window"][0][0] < 1e-30
    assert r["kernels"][0][0] != 0.0
    assert abs(r["kernels"][0][0]) < 1e-40
    assert r["meta"]["window_branch"] == "hann"
    assert r["meta"]["gcd"] == 4 and r["meta"]["shape"] == [1, 1, 28]


def check_kernel_hann_values(solve):
    _near(solve("kernel", orig_freq=8, new_freq=4)["kernels"], K84)


def check_kernel_upsample_values(solve):
    """Two phase rows for 4 -> 8, with the -j/n row offsets."""
    r = solve("kernel", orig_freq=4, new_freq=8)
    assert r["shape"] == [2, 15] and r["width"] == 7
    _near(r["kernels"], K48)
    _near(r["t"][0][7], 0.0, 1e-15)
    _near(r["t"][1][7], -0.495, 1e-15)


def check_kernel_ratio_values(solve):
    """Three phase rows, W=4 and rolloff=0.8."""
    r = solve("kernel", orig_freq=6, new_freq=9,
              lowpass_filter_width=4, rolloff=0.8)
    assert r["shape"] == [3, 12] and r["width"] == 5
    _near(r["kernels"], K69)
    assert r["t"][0][0] == -4.0


def check_kernel_kaiser_values(solve):
    """Kaiser window, default beta, and its nonzero clamp boundary."""
    r = solve("kernel", orig_freq=8, new_freq=4,
              resampling_method="sinc_interp_kaiser")
    _near(r["kernels"], KK84)
    _near(r["beta_float32"], 14.76965618133545, 1e-12)
    _near(r["i0_beta_float32"], 271903.5625, 1e-9)
    _near(r["window"][0][0], 1.0 / 271903.5625, 1e-12)
    assert r["window"][0][0] > 1e-8
    assert r["kernels"][0][0] != 0.0
    assert r["meta"]["window_branch"] == "kaiser"
    assert r["meta"]["beta_default_used"] is True


def check_kernel_alias_quirk(solve):
    """Deprecated aliases only warn; they never rewrite the branch."""
    h = solve("kernel", orig_freq=8, new_freq=4)
    a = solve("kernel", orig_freq=8, new_freq=4,
              resampling_method="sinc_interpolation")
    k = solve("kernel", orig_freq=8, new_freq=4,
              resampling_method="sinc_interp_kaiser")
    assert a["meta"]["deprecated"] is True
    assert a["meta"]["replacement"] == "sinc_interp_hann"
    assert a["meta"]["window_branch"] == "kaiser"
    _near(a["kernels"], k["kernels"])
    assert not all(abs(x - y) < 1e-12
                   for x, y in zip(a["kernels"][0], h["kernels"][0]))
    b = solve("kernel", orig_freq=8, new_freq=4,
              resampling_method="kaiser_window")
    _near(b["kernels"], k["kernels"])
    assert b["meta"]["replacement"] == "sinc_interp_kaiser"


def check_kernel_explicit_beta(solve):
    """A judged explicit beta (float32 i0 agrees bit-for-bit at 3.0)."""
    r = solve("kernel", orig_freq=8, new_freq=4,
              resampling_method="sinc_interp_kaiser", beta=3.0)
    _near(r["kernels"], K84B3)
    assert r["beta_float32"] == 3.0
    _near(r["i0_beta_float32"], 4.880792617797852, 1e-12)
    assert r["kernels"][0][13] != 0.495
    _near(r["kernels"][0][13], 0.4949999967614379, 1e-12)


def check_kernel_errors(solve):
    _err(lambda: solve("kernel", orig_freq=8.5, new_freq=4),
         "Frequencies must be of integer type")
    _err(lambda: solve("kernel", orig_freq=8, new_freq=4,
                       lowpass_filter_width=0), "lowpass_filter_width")
    _err(lambda: solve("kernel", orig_freq=8, new_freq=4, rolloff=0),
         "rolloff must be positive")
    _err(lambda: solve("kernel", orig_freq=8, new_freq=4,
                       resampling_method="bad"), "Invalid resampling method")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def check_apply_hann_default(solve):
    r = solve("apply", waveform=WF, orig_freq=8, new_freq=4,
              kernels=K84, width=13)
    _near(r["out"], A84_8)
    assert r["shape"] == [4] and r["target_length"] == 4
    assert r["num_frames"] == 5 and r["conv_length"] == 5
    assert r["padded_length"] == 8 + 26 + 2
    assert r["stride"] == 2 and r["width"] == 13


def check_apply_nonintegral_length(solve):
    r = solve("apply", waveform=W11, orig_freq=8, new_freq=4,
              kernels=K84, width=13)
    _near(r["out"], A84_11)
    assert r["shape"] == [6] and r["target_length"] == 6
    assert r["num_frames"] == 6 and r["conv_length"] == 6


def check_apply_upsample(solve):
    r = solve("apply", waveform=WF, orig_freq=4, new_freq=8,
              kernels=K48, width=7)
    _near(r["out"], A48_8)
    assert r["shape"] == [16] and r["target_length"] == 16
    assert r["stride"] == 1 and r["num_frames"] == 9
    assert r["conv_length"] == 18
    assert r["padded_length"] == 8 + 14 + 1


def check_apply_batch(solve):
    r2 = solve("apply", waveform=WB, orig_freq=8, new_freq=4,
               kernels=K84, width=13)
    assert r2["shape"] == [2, 4]
    _near(r2["out"], [A84_8, [v * 0.5 for v in A84_8]])
    r3 = solve("apply", waveform=W3, orig_freq=8, new_freq=4,
               kernels=K84, width=13)
    assert r3["shape"] == [2, 2, 4]
    _near(r3["out"][0][0], A84_8)
    _near(r3["out"][0][1], [v * 0.5 for v in A84_8])
    _near(r3["out"][1][1], A84_8)


def check_apply_errors(solve):
    _err(lambda: solve("apply", waveform=WF, orig_freq=8, new_freq=4,
                       kernels=[K84[0], K84[0]], width=13),
         "kernel shape must be")
    _err(lambda: solve("apply", waveform=WF, orig_freq=8, new_freq=4,
                       kernels=K84[0], width=13),
         "kernels must be a 2-D")
    _err(lambda: solve("apply", waveform=WF, orig_freq=8, new_freq=4,
                       kernels=K84, width=12),
         "kernel shape must be")
    _err(lambda: solve("apply", waveform=WF, orig_freq=8, new_freq=4,
                       kernels=K84, width=1.5), "width must be an integer")
    _err(lambda: solve("apply", waveform=[], orig_freq=8, new_freq=4,
                       kernels=K84, width=13), "waveform")
    _err(lambda: solve("apply", waveform=WF, orig_freq=8.5, new_freq=4,
                       kernels=K84, width=13),
         "Frequencies must be of integer type")


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------
def check_resample_hann(solve):
    r = solve("resample", waveform=WF, orig_freq=8, new_freq=4)
    _near(r["out"], A84_8)
    assert r["shape"] == [4] and r["target_length"] == 4
    assert r["identity"] is False
    assert r["meta"]["gcd"] == 4 and r["meta"]["width"] == 13


def check_resample_kaiser(solve):
    r = solve("resample", waveform=WF, orig_freq=8, new_freq=4,
              resampling_method="sinc_interp_kaiser")
    _near(r["out"], [0.48786505742930386, -0.48452677957668605,
                     0.48452677957668605, -0.48786505742930386])
    assert r["shape"] == [4] and r["identity"] is False
    assert r["meta"]["window_branch"] == "kaiser"


def check_resample_upsample(solve):
    r = solve("resample", waveform=WF, orig_freq=4, new_freq=8)
    _near(r["out"], A48_8)
    assert r["shape"] == [16] and r["target_length"] == 16
    assert r["meta"]["kernel_length"] == 15


def check_resample_ratio(solve):
    r = solve("resample", waveform=W11, orig_freq=6, new_freq=9,
              lowpass_filter_width=4, rolloff=0.8)
    _near(r["out"], R69)
    assert r["shape"] == [17] and r["target_length"] == 17
    _near(r["meta"]["base_freq"], 1.6, 1e-15)


def check_resample_alias_quirk(solve):
    """The alias really changes the end-to-end waveform (Kaiser, not Hann)."""
    a = solve("resample", waveform=WF, orig_freq=8, new_freq=4,
              resampling_method="sinc_interpolation")
    k = solve("resample", waveform=WF, orig_freq=8, new_freq=4,
              resampling_method="sinc_interp_kaiser")
    h = solve("resample", waveform=WF, orig_freq=8, new_freq=4)
    _near(a["out"], k["out"])
    assert a["meta"]["deprecated"] is True
    assert abs(a["out"][0] - h["out"][0]) > 1e-6


def check_resample_identity(solve):
    for wf, shape in [(WF, [8]), (WB, [2, 8]), (W3, [2, 2, 8])]:
        r = solve("resample", waveform=wf, orig_freq=8, new_freq=8)
        assert r["identity"] is True
        assert r["shape"] == shape and r["target_length"] == 8
        _near(r["out"], wf)
        # The short circuit still reports the reduced-parameter ledger.
        assert r["meta"]["gcd"] == 8
        assert r["meta"]["orig_freq_reduced"] == 1
        assert r["meta"]["new_freq_reduced"] == 1
        assert r["meta"]["width"] == 7
        assert r["meta"]["shape"] == [1, 1, 15]


def check_resample_long_ratio(solve):
    wf = [math.cos(i * 0.31) for i in range(12)]
    r = solve("resample", waveform=wf, orig_freq=44100, new_freq=48000,
              lowpass_filter_width=2, rolloff=0.9)
    assert r["shape"] == [14] and r["target_length"] == 14
    _near(r["out"][:4], [0.9523010190446559, 0.9706783662841323,
                         0.8497862181256776, 0.6658942840912664])
    assert r["meta"]["kernel_length"] == 153
    assert r["meta"]["shape"] == [160, 1, 153]


def check_resample_errors(solve):
    _err(lambda: solve("resample", waveform=WF, orig_freq=0, new_freq=4),
         "should be positive")
    _err(lambda: solve("resample", waveform=WF, orig_freq=8, new_freq=-1),
         "should be positive")
    _err(lambda: solve("resample", waveform=WF, orig_freq=8.5, new_freq=4),
         "Frequencies must be of integer type")
    _err(lambda: solve("resample", waveform=WF, orig_freq=8, new_freq=4,
                       lowpass_filter_width=0), "lowpass_filter_width")
    _err(lambda: solve("resample", waveform=WF, orig_freq=8, new_freq=4,
                       resampling_method="bad"), "Invalid resampling method")
    _err(lambda: solve("resample", waveform=[], orig_freq=8, new_freq=4),
         "waveform")


# ---------------------------------------------------------------------------
# hardening: exact contracts that the mutation sweep showed to be unguarded
# ---------------------------------------------------------------------------
def _msg(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    raise AssertionError("expected ValueError")


def check_hardening_defaults(solve):
    """Unspecified frequencies must default to 8 -> 4 in every mode."""
    r = solve("meta")
    assert r["gcd"] == 4 and r["orig_freq_reduced"] == 2
    assert r["new_freq_reduced"] == 1 and r["width"] == 13
    assert r["shape"] == [1, 1, 28] and r["method"] == "sinc_interp_hann"
    k = solve("kernel", orig_freq=8, new_freq=4)["kernels"]
    _near(solve("apply", waveform=WF, kernels=k, width=13)["out"], A84_8)
    _near(solve("apply", waveform=WF, kernels=k, width=13)["shape"], [4])
    _near(solve("resample", waveform=WF)["out"], A84_8)
    assert solve("resample", waveform=WF)["target_length"] == 4


def check_hardening_sinc_column(solve):
    """The sinc ledger column is not the window column."""
    r = solve("kernel", orig_freq=8, new_freq=4)
    _near(r["sinc"][0][12], 0.6429709436021601, 1e-12)
    _near(r["sinc"][0][11], 0.010099348633439868, 1e-12)
    _near(r["window"][0][12], 0.9833000510084535, 1e-12)
    assert abs(r["sinc"][0][12] - r["window"][0][12]) > 0.1
    assert r["sinc"][0][13] == 1.0


def check_hardening_error_messages(solve):
    """Exact messages pin down which guard fired."""
    k = solve("kernel", orig_freq=8, new_freq=4)["kernels"]
    assert _msg(lambda: solve("i0", values=[])) == \
        "values must be a non-empty numeric array"
    assert _msg(lambda: solve("i0", values=[True])) == \
        "values must be a non-empty numeric array"
    assert _msg(lambda: solve("apply", waveform=[], orig_freq=8,
                              new_freq=4, kernels=k, width=13)) == \
        "waveform must be a non-empty numeric array"
    assert _msg(lambda: solve("apply", waveform=WF, orig_freq=8,
                              new_freq=4, kernels=[], width=13)) == \
        "kernels must be a non-empty numeric array"
    assert _msg(lambda: solve("meta", orig_freq=8, new_freq=4,
                              lowpass_filter_width=0)) == \
        "lowpass_filter_width must be at least 1"
    assert _msg(lambda: solve("meta", orig_freq=8, new_freq=4,
                              lowpass_filter_width=True)) == \
        "lowpass_filter_width must be an integer"
    assert _msg(lambda: solve("meta", orig_freq=8, new_freq=4,
                              resampling_method=7)) == \
        "Invalid resampling method"
    assert _msg(lambda: solve("meta", orig_freq=8, new_freq=4,
                              rolloff="x")) == "rolloff must be positive"


def check_hardening_types(solve):
    """Integral floats, beta=0 and the identity short-circuit guard."""
    a = solve("meta", orig_freq=8.0, new_freq=4.0)
    b = solve("meta", orig_freq=8, new_freq=4)
    assert a == b
    z = solve("meta", orig_freq=8, new_freq=4, beta=0.0)
    assert z["beta"] == 0.0 and z["beta_default_used"] is False
    _err(lambda: solve("resample", waveform=[], orig_freq=8, new_freq=8),
         "waveform")
    _err(lambda: solve("resample", waveform=[True], orig_freq=8, new_freq=8),
         "waveform")


def check_hardening_unknown_mode(solve):
    _err(lambda: solve("bogus"), "unknown mode")
