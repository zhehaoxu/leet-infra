#!/usr/bin/env python3
"""861 judge cases: torchaudio ITU-R BS.1770-4 loudness-gate ledger.

Every check takes `solve` and asserts on explicit, hand-verified values taken
from the verbatim official implementation (see notes.md / fixtures861.txt).
No torch, no import from the solution module, no global RNG.  Helpers use
internal `assert`, never a returned bool (a bare bool would silently disable
the assertion).
"""
import math

# ------------------------------------------------------------------ helpers
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
    if want == float("-inf"):
        assert got == float("-inf"), "%s: %r != -inf" % (tag, got)
        return
    assert abs(float(got) - float(want)) <= atol + rtol * abs(float(want)), \
        "%s: %r != %r" % (tag, got, want)


def _flat(v):
    out = []
    for e in v:
        if isinstance(e, list):
            out.extend(_flat(e))
        else:
            out.append(e)
    return out


def _stats(v, tag, total, sq, vmax, vmin):
    f = [x for x in _flat(v) if isinstance(x, float) and math.isfinite(x)]
    _near(math.fsum(f), total, tag + " sum")
    _near(math.fsum(x * x for x in f), sq, tag + " sq")
    _near(max(f), vmax, tag + " max")
    _near(min(f), vmin, tag + " min")


def _raises(fn, tag, needle=None):
    try:
        fn()
    except ValueError as e:
        if needle is not None:
            assert needle in str(e), "%s: message %r lacks %r" % (tag, str(e), needle)
        return
    except BaseException as e:
        raise AssertionError("%s: wrong exception %s: %s" % (tag, type(e).__name__, e))
    raise AssertionError("%s: no exception" % tag)


# ------------------------------------------------------------------ fixtures
C8000_TREBLE_A = [3.625826704781014, -1.2110558364958972, 0.6938511881834482]
C8000_TREBLE_B = [4.814130315702842, -2.828500090334113, 1.1229918310998355]
C8000_TREBLE_AN = [1.0, -0.33400819595128456, 0.191363582619251]
C8000_TREBLE_BN = [1.3277331509955872, -0.7800979805803884, 0.30972021625276763]
C8000_HIGHPASS_A = [1.0298406997386809, -1.9991093343177666, 0.9701593002613191]
C8000_HIGHPASS_B = [0.9997773335794417, -1.9995546671588833, 0.9997773335794417]
C8000_HIGHPASS_AN = [1.0, -1.9411830730957078, 0.9420479308183239]
C8000_HIGHPASS_BN = [0.9708077509785079, -1.9416155019570158, 0.9708077509785079]
C4000_TREBLE_A = [3.564031780597306, 3.7124537773368744, 1.3199948719933783]
C4000_TREBLE_B = [4.025862120720109, 3.369831277732851, 1.2007870314745994]
C4000_HIGHPASS_A = [1.0596548213901706, -1.9964381305564236, 0.9403451786098294]
C4000_HIGHPASS_B = [0.999109532639106, -1.998219065278212, 0.999109532639106]
C16000_TREBLE_A = [2.9252010427967363, -3.238604849141393, 1.162072557272052]
C16000_TREBLE_B = [4.224674535812143, -5.3810330660117, 2.0050272811269534]
C16000_HIGHPASS_A = [1.01492201127725, -1.9997773211829777, 0.9850779887227501]
C16000_HIGHPASS_B = [0.9999443302957445, -1.999888660591489, 0.9999443302957445]
C44100_TREBLE_A = [2.342430424963229, -3.897218877772631, 1.6693506831063156]
C44100_TREBLE_B = [3.586049212352069, -6.2101790032196975, 2.738692021164543]
C44100_HIGHPASS_A = [1.005414056127794, -1.9999706877814436, 0.9945859438722061]
C44100_HIGHPASS_B = [0.9999926719453609, -1.9999853438907218, 0.9999926719453609]
C25_TREBLE_A = [1.9999999999999316, -4.0, 2.0000000000000684]
C25_TREBLE_B = [3.169786384922141, -6.339572769844454, 3.1697863849223133]
C25_HIGHPASS_A = [0.8746667664356969, 1.984229402628956, 1.1253332335643031]
C25_HIGHPASS_B = [0.0039426493427610065, -0.007885298685522013, 0.0039426493427610065]

W1 = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
W2 = [0.5, 0.25, -0.5, -0.25, 0.5, 0.25, -0.5, -0.25]
W3 = [3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0]
CS_A = [2.0, -0.5, 0.25]
CS_B = [0.5, 0.25, 0.125]

LF_HP_W1_CL1 = [0.9708077509785079, -0.05709992852742318, -1.0, 0.004938896855361197, 0.9755527452957511, -0.05254170343997355, -1.0, 0.009143984607933309]
LF_HP_W1_CL0 = [0.9708077509785079, -0.05709992852742318, -1.0253888477641027, 0.004938896855361197, 0.9755527452957511, -0.05254170343997355, -1.0210105104592888, 0.009143984607933309]
LF_HP_W2_CL1 = [0.48540387548925396, 0.21415197348091541, -0.5269694060139071, -0.253877763513345, 0.4890110968617159, 0.21761733460395105, -0.5236406810896378, -0.2506806353108556]
LF_HP_W3_CL1 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
LF_HP_W3_CL0 = [2.9124232529355236, -3.0837230385177934, 2.9199797481610084, -3.076462843177195, 2.92695453577214, -3.0697629555259773, 2.9333896600353633, -3.0635828165314845]
LF_TR_W1_CL1 = [1.0, -0.3366242261116378, -1.0, 0.38207191307110333, 1.0, -0.3820685803307799, -1.0, 0.38040338136632373]
LF_TR_W1_CL0 = [1.3277331509955872, -0.3366242261116378, -1.3845279577567275, 0.38207191307110333, 1.410576315384197, -0.3820685803307799, -1.4155599092585542, 0.38040338136632373]
LF_TR_W2_CL1 = [0.6638665754977936, 0.1636211746930779, -0.7764200354062731, -0.15509603290363017, 0.8008061359598744, 0.16160978868065934, -0.803297099711972, -0.16368828663147666]
LF_CS_W1_CL1 = [0.25, 0.1875, -0.171875, -0.19140625, 0.1611328125, 0.189208984375, -0.16033935546875, -0.1887359619140625]
LF_CS_W2_CL1 = [0.125, 0.15625, -0.0390625, -0.138671875, 0.03271484375, 0.1348876953125, -0.032867431640625, -0.13445281982421875]
LF_CS_W3_CL1 = [0.75, -0.1875, 0.421875, -0.43359375, 0.4013671875, -0.407958984375, 0.41033935546875, -0.4089202880859375]

KWA_T_FIRST = [1.0, -0.3366242261116378, -1.0, 0.38207191307110333, 1.0, -0.3820685803307799]
KWA_H_FIRST = [0.9708077509785079, -0.3838973364037428, -1.0, 0.3942305909617507, 0.9712951332868484, -0.42753470727101095]
KWB_T_FIRST = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
KWB_H_FIRST = [0.9708077509785079, -1.0, 0.9733265827203366, -1.0, 0.9756515119240472, -1.0]
KWC_T_FIRST = [1.0, -0.23111059102479747, -0.9702855843814208, 0.15077696642354366, 0.994967156513207, -0.14673357463633052]
KWC_H_FIRST = [0.9428631970252023, -0.3272346941908735, -0.9894006201799794, 0.1802638953005189, 0.956573765026289, -0.23016221028130923]

F1 = [[(i + 1) * 0.25 for i in range(20)],
      [(20 - i) * 0.1 for i in range(20)]]
F2 = [[0.0] * 10 + [1.0] * 10,
      [0.0] * 10 + [0.5] * 10]
F3 = [[0.5 + 0.1 * c for _ in range(20)] for c in range(5)]
F4 = [[0.0] * 20]
F5 = [[0.001] * 10 + [2.0] * 10,
      [0.001] * 10 + [1.0] * 10]
F6 = [[0.0] * 20, [1.0] * 20]
F1L = [[(i + 1) * 0.25 for i in range(60)],
       [(60 - i) * 0.1 for i in range(60)]]
F2L = [[0.0] * 30 + [1.0] * 30,
       [0.0] * 30 + [0.5] * 30]


def carrier(n, amp, period=20):
    return [amp * (1.0 if (i // (period // 2)) % 2 == 0 else -1.0) for i in range(n)]


RAW1 = [carrier(2000, 0.5)]
RAW2 = [carrier(2000, 0.4), carrier(2000, 0.2)]
RAW3 = [carrier(1800, 0.3) for _ in range(3)]


# ------------------------------------------------------------------ coeffs
def check_coeffs_8000(solve):
    r = solve("coeffs", sample_rate=8000)
    _near(r["treble"]["a"], C8000_TREBLE_A, "coeffs treble a")
    _near(r["treble"]["b"], C8000_TREBLE_B, "coeffs treble b")
    _near(r["highpass"]["a"], C8000_HIGHPASS_A, "coeffs highpass a")
    _near(r["highpass"]["b"], C8000_HIGHPASS_B, "coeffs highpass b")


def check_coeffs_8000_norm(solve):
    r = solve("coeffs", sample_rate=8000)
    _near(r["treble"]["a_norm"], C8000_TREBLE_AN, "coeffs treble a_norm")
    _near(r["treble"]["b_norm"], C8000_TREBLE_BN, "coeffs treble b_norm")
    _near(r["highpass"]["a_norm"], C8000_HIGHPASS_AN, "coeffs highpass a_norm")
    _near(r["highpass"]["b_norm"], C8000_HIGHPASS_BN, "coeffs highpass b_norm")


def check_coeffs_8000_intermediates(solve):
    r = solve("coeffs", sample_rate=8000)
    t, h = r["treble"], r["highpass"]
    _near(t["w0"], 2 * math.pi * 1500.0 / 8000, "treble w0")
    _near(t["alpha"], math.sin(2 * math.pi * 1500.0 / 8000) / 2 / (1 / math.sqrt(2)), "treble alpha")
    _near(t["A"], 10.0 ** 0.1, "treble A")
    _near(t["temp1"], 2 * math.sqrt(10.0 ** 0.1) * t["alpha"], "treble temp1")
    _near(t["temp2"], (10.0 ** 0.1 - 1) * math.cos(t["w0"]), "treble temp2")
    _near(t["temp3"], (10.0 ** 0.1 + 1) * math.cos(t["w0"]), "treble temp3")
    _near(h["w0"], 2 * math.pi * 38.0 / 8000, "highpass w0")
    _near(h["alpha"], math.sin(2 * math.pi * 38.0 / 8000) / 2.0 / 0.5, "highpass alpha")
    assert "A" not in h, "highpass must not expose A"
    assert "temp1" not in h, "highpass must not expose temp1"


def check_coeffs_4000(solve):
    r = solve("coeffs", sample_rate=4000)
    _near(r["treble"]["a"], C4000_TREBLE_A, "coeffs 4000 treble a")
    _near(r["treble"]["b"], C4000_TREBLE_B, "coeffs 4000 treble b")
    _near(r["highpass"]["a"], C4000_HIGHPASS_A, "coeffs 4000 highpass a")
    _near(r["highpass"]["b"], C4000_HIGHPASS_B, "coeffs 4000 highpass b")


def check_coeffs_16000(solve):
    r = solve("coeffs", sample_rate=16000)
    _near(r["treble"]["a"], C16000_TREBLE_A, "coeffs 16000 treble a")
    _near(r["treble"]["b"], C16000_TREBLE_B, "coeffs 16000 treble b")
    _near(r["highpass"]["a"], C16000_HIGHPASS_A, "coeffs 16000 highpass a")
    _near(r["highpass"]["b"], C16000_HIGHPASS_B, "coeffs 16000 highpass b")


def check_coeffs_44100(solve):
    r = solve("coeffs", sample_rate=44100)
    _near(r["treble"]["a"], C44100_TREBLE_A, "coeffs 44100 treble a")
    _near(r["treble"]["b"], C44100_TREBLE_B, "coeffs 44100 treble b")
    _near(r["highpass"]["a"], C44100_HIGHPASS_A, "coeffs 44100 highpass a")
    _near(r["highpass"]["b"], C44100_HIGHPASS_B, "coeffs 44100 highpass b")


def check_coeffs_25(solve):
    r = solve("coeffs", sample_rate=25)
    _near(r["treble"]["a"], C25_TREBLE_A, "coeffs 25 treble a")
    _near(r["treble"]["b"], C25_TREBLE_B, "coeffs 25 treble b")
    _near(r["highpass"]["a"], C25_HIGHPASS_A, "coeffs 25 highpass a")
    _near(r["highpass"]["b"], C25_HIGHPASS_B, "coeffs 25 highpass b")


def check_coeffs_highpass_symmetry(solve):
    for sr in (4000, 8000, 16000, 44100):
        h = solve("coeffs", sample_rate=sr)["highpass"]
        _near(h["b"][0], h["b"][2], "b0==b2 sr=%d" % sr)
        _near(h["a"][0] + h["a"][2], 2.0, "a0+a2==2 sr=%d" % sr)
        _near(h["b"][1], -1 - math.cos(h["w0"]), "b1 sr=%d" % sr)
        _near(h["a"][1], -2 * math.cos(h["w0"]), "a1 sr=%d" % sr)


def check_coeffs_validation(solve):
    _raises(lambda: solve("coeffs", sample_rate=True), "coeffs bool sr", "sample_rate must be an int")
    _raises(lambda: solve("coeffs", sample_rate=8000.0), "coeffs float sr", "sample_rate must be an int")
    _raises(lambda: solve("coeffs", sample_rate=0), "coeffs zero sr", "sample_rate must be >= 1")
    _raises(lambda: solve("coeffs", sample_rate=-8), "coeffs negative sr", "sample_rate must be >= 1")
    _raises(lambda: solve("coeffs"), "coeffs missing sr", "sample_rate must be an int")
    _near(solve("coeffs", sample_rate=8000)["sample_rate"], 8000, "coeffs echoes sr")


# ------------------------------------------------------------------ lfilter
def check_lfilter_hp_w1_clamp(solve):
    r = solve("lfilter", waveform=W1, a_coeffs=C8000_HIGHPASS_A,
              b_coeffs=C8000_HIGHPASS_B, clamp=True)
    _near(r["out"], LF_HP_W1_CL1, "lfilter hp w1 clamp")
    _near(r["shape"], [8], "lfilter hp w1 shape")
    assert r["clamp"] is True, "clamp echo"


def check_lfilter_hp_w1_noclamp(solve):
    r = solve("lfilter", waveform=W1, a_coeffs=C8000_HIGHPASS_A,
              b_coeffs=C8000_HIGHPASS_B, clamp=False)
    _near(r["out"], LF_HP_W1_CL0, "lfilter hp w1 noclamp")
    assert r["clamp"] is False, "clamp echo false"


def check_lfilter_hp_w2(solve):
    r = solve("lfilter", waveform=W2, a_coeffs=C8000_HIGHPASS_A,
              b_coeffs=C8000_HIGHPASS_B, clamp=True)
    _near(r["out"], LF_HP_W2_CL1, "lfilter hp w2")


def check_lfilter_hp_w3_clamp(solve):
    r = solve("lfilter", waveform=W3, a_coeffs=C8000_HIGHPASS_A,
              b_coeffs=C8000_HIGHPASS_B, clamp=True)
    _near(r["out"], LF_HP_W3_CL1, "lfilter hp w3 clamp")
    r0 = solve("lfilter", waveform=W3, a_coeffs=C8000_HIGHPASS_A,
               b_coeffs=C8000_HIGHPASS_B, clamp=False)
    _near(r0["out"], LF_HP_W3_CL0, "lfilter hp w3 noclamp")
    assert all(-1.0 <= v <= 1.0 for v in r["out"]), "clamp bounds"
    assert any(abs(v) > 1.0 for v in r0["out"]), "noclamp exceeds bounds"


def check_lfilter_tr_w1(solve):
    a, b = C8000_TREBLE_A, C8000_TREBLE_B
    _near(solve("lfilter", waveform=W1, a_coeffs=a, b_coeffs=b,
                clamp=True)["out"], LF_TR_W1_CL1, "lfilter tr w1 clamp")
    _near(solve("lfilter", waveform=W1, a_coeffs=a, b_coeffs=b,
                clamp=False)["out"], LF_TR_W1_CL0, "lfilter tr w1 noclamp")


def check_lfilter_tr_w2(solve):
    _near(solve("lfilter", waveform=W2, a_coeffs=C8000_TREBLE_A,
                b_coeffs=C8000_TREBLE_B, clamp=True)["out"],
          LF_TR_W2_CL1, "lfilter tr w2")


def check_lfilter_custom(solve):
    _near(solve("lfilter", waveform=W1, a_coeffs=CS_A, b_coeffs=CS_B)["out"],
          LF_CS_W1_CL1, "lfilter custom w1")
    _near(solve("lfilter", waveform=W2, a_coeffs=CS_A, b_coeffs=CS_B)["out"],
          LF_CS_W2_CL1, "lfilter custom w2")
    _near(solve("lfilter", waveform=W3, a_coeffs=CS_A, b_coeffs=CS_B)["out"],
          LF_CS_W3_CL1, "lfilter custom w3")


def check_lfilter_nested_shape(solve):
    w = [[W1, W2], [[0.2, 0.0, -0.2, 0.0, 0.2, 0.0, -0.2, 0.0], W1]]
    r = solve("lfilter", waveform=w, a_coeffs=CS_A, b_coeffs=CS_B)
    _near(r["shape"], [2, 2, 8], "lfilter nested shape")
    _near(r["out"][0][0], LF_CS_W1_CL1, "lfilter nested first row")
    _near(r["out"][0][1], LF_CS_W2_CL1, "lfilter nested second row")
    w2 = [W1, W2]
    r2 = solve("lfilter", waveform=w2, a_coeffs=CS_A, b_coeffs=CS_B)
    _near(r2["shape"], [2, 8], "lfilter 2d shape")


def check_lfilter_validation(solve):
    base = dict(waveform=W1, a_coeffs=CS_A, b_coeffs=CS_B)
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=[1.0, 2.0], b_coeffs=CS_B),
            "lfilter short a", "a_coeffs must have exactly 3 numbers")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=CS_A, b_coeffs=[1.0]),
            "lfilter short b", "b_coeffs must have exactly 3 numbers")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=CS_A + [0.0], b_coeffs=CS_B),
            "lfilter long a", "a_coeffs must have exactly 3 numbers")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=[0.0, 1.0, 2.0], b_coeffs=CS_B),
            "lfilter zero a0", "a_coeffs[0] must be nonzero")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=CS_A, b_coeffs=CS_B, clamp=1),
            "lfilter clamp int", "clamp must be a bool")
    _raises(lambda: solve("lfilter", waveform=[], a_coeffs=CS_A, b_coeffs=CS_B),
            "lfilter empty waveform", "waveform must be a non-empty list")
    _raises(lambda: solve("lfilter", waveform=[1.0, True], a_coeffs=CS_A, b_coeffs=CS_B),
            "lfilter bool sample", "waveform must be a non-empty list")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=[1.0, True, 2.0], b_coeffs=CS_B),
            "lfilter bool coeff", "a_coeffs must contain only numbers")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=CS_A, b_coeffs="bad"),
            "lfilter b not list", "b_coeffs must be a non-empty list of numbers")
    _near(solve("lfilter", **base)["a_coeffs"], CS_A, "lfilter echoes a")
    _near(solve("lfilter", **base)["b_coeffs"], CS_B, "lfilter echoes b")


# ------------------------------------------------------------------ kweight
def check_kweight_A(solve):
    r = solve("kweight", waveform=[W1, W2], sample_rate=8000)
    _near(r["shape"], [2, 8], "kweight A shape")
    _near(r["after_treble"][0][:6], KWA_T_FIRST, "kweight A treble first")
    _near(r["after_highpass"][0][:6], KWA_H_FIRST, "kweight A high first")
    _stats(r["after_treble"], "kweight A treble", -0.0648152918269374,
           6.983839795442221, 1.0, -1.0)
    _stats(r["after_highpass"], "kweight A high", -0.334983357225238,
           6.9562206763968, 0.9712951332868484, -1.0)


def check_kweight_B(solve):
    r = solve("kweight", waveform=[W3, [0.25, -0.25] * 4], sample_rate=8000)
    _near(r["after_treble"][0][:6], KWB_T_FIRST, "kweight B treble first")
    _near(r["after_highpass"][0][:6], KWB_H_FIRST, "kweight B high first")
    _stats(r["after_treble"], "kweight B treble", -0.07320960134407611,
           9.226256585298305, 1.0, -1.0)
    _stats(r["after_highpass"], "kweight B high", -0.22695015693505732,
           9.023134328499136, 0.9777965533451223, -1.0)
    for row in r["after_treble"] + r["after_highpass"]:
        assert all(-1.0 <= v <= 1.0 for v in row), "kweight B clamp"


def check_kweight_C(solve):
    r = solve("kweight", waveform=[W1], sample_rate=4000)
    _near(r["after_treble"][0][:6], KWC_T_FIRST, "kweight C treble first")
    _near(r["after_highpass"][0][:6], KWC_H_FIRST, "kweight C high first")
    _stats(r["after_treble"], "kweight C treble", -0.04324052631756936,
           4.0544174617165405, 1.0, -1.0)
    _stats(r["after_highpass"], "kweight C high", -0.2693471705304138,
           4.014595086999057, 0.956573765026289, -1.0)


def check_kweight_coeffs_and_validation(solve):
    r = solve("kweight", waveform=[W1], sample_rate=8000)
    _near(r["coeffs"]["treble"]["a"], C8000_TREBLE_A, "kweight coeffs")
    _near(r["sample_rate"], 8000, "kweight sr")
    _raises(lambda: solve("kweight", waveform=[W1], sample_rate=0),
            "kweight sr zero", "sample_rate must be >= 1")
    _raises(lambda: solve("kweight", waveform=[], sample_rate=8000),
            "kweight empty", "waveform must be a non-empty list")
    _raises(lambda: solve("kweight", waveform=[[1.0, 2.0], [3.0]], sample_rate=8000),
            "kweight ragged", "waveform must be rectangular")
    _raises(lambda: solve("kweight", waveform=[[1.0, True]], sample_rate=8000),
            "kweight bool", "waveform must be a non-empty list")


# ------------------------------------------------------------------ blocks
def check_blocks_F1_geometry(solve):
    r = solve("blocks", waveform=F1, sample_rate=25)
    _near([r["gate_samples"], r["step"], r["n_blocks"], r["n_channels"]],
          [10, 2, 6, 2], "blocks F1 geometry")
    _near(r["g"], [1.0, 1.0], "blocks F1 g")
    _near(r["sample_rate"], 25, "blocks F1 sr")


def check_blocks_F1_energy(solve):
    r = solve("blocks", waveform=F1, sample_rate=25)
    _near(r["energy"],
          [[2.40625, 4.03125, 6.15625, 8.78125, 11.90625, 15.53125],
           [2.4850000000000003, 1.9050000000000005, 1.4050000000000002,
            0.9850000000000001, 0.6450000000000001, 0.38500000000000006]],
          "blocks F1 energy")
    _near(r["energy_weighted"],
          [4.89125, 5.93625, 7.56125, 9.76625, 12.55125, 15.91625],
          "blocks F1 energy_weighted")
    _near(r["block_loudness"],
          [6.203198609087365, 7.044121825900748, 8.09493597515407,
           9.206278373176326, 10.295869800854959, 11.327407520857967],
          "blocks F1 block_loudness")


def check_blocks_F1_gating(solve):
    r = solve("blocks", waveform=F1, sample_rate=25)
    _near(r["gate_abs"], [True] * 6, "blocks F1 gate_abs")
    _near(r["n_gated_abs"], 6, "blocks F1 n_gated_abs")
    _near(r["energy_filtered_abs"], [8.135416666666666, 1.301666666666667],
          "blocks F1 efa")
    _near(r["ew_abs"], 9.437083333333334, "blocks F1 ew_abs")
    _near(r["gamma_rel"], -0.9426220993534962, "blocks F1 gamma_rel")
    _near(r["gate_rel"], [True] * 6, "blocks F1 gate_rel")
    _near(r["n_gated_rel"], 6, "blocks F1 n_gated_rel")
    _near(r["energy_filtered_rel"], [8.135416666666666, 1.301666666666667],
          "blocks F1 efr")
    _near(r["ew_rel"], 9.437083333333334, "blocks F1 ew_rel")
    _near(r["lkfs"], 9.057377900646504, "blocks F1 lkfs")
    _near(r["lkfs"], r["gamma_rel"] + 10.0, "blocks F1 lkfs==gamma_rel+10")


def check_blocks_F2_absolute_gate(solve):
    r = solve("blocks", waveform=F2, sample_rate=25)
    _near(r["energy"], [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]], "blocks F2 energy")
    _near(r["block_loudness"][0], float("-inf"), "blocks F2 -inf")
    _near(r["gate_abs"], [False, True, True, True, True, True], "blocks F2 gate_abs")
    _near(r["n_gated_abs"], 5, "blocks F2 n_gated_abs")
    _near(r["energy_filtered_abs"], [0.6, 0.15], "blocks F2 efa")
    _near(r["ew_abs"], 0.75, "blocks F2 ew_abs")
    _near(r["gamma_rel"], -11.940387366083, "blocks F2 gamma_rel")
    _near(r["gate_rel"], [False, True, True, True, True, True], "blocks F2 gate_rel")
    _near(r["lkfs"], -1.9403873660829993, "blocks F2 lkfs")


def check_blocks_F3_five_channels(solve):
    r = solve("blocks", waveform=F3, sample_rate=25)
    _near(r["g"], [1.0, 1.0, 1.0, 1.41, 1.41], "blocks F3 g")
    _near(r["n_channels"], 5, "blocks F3 n_channels")
    _near(r["energy_weighted"], [3.1445] * 6, "blocks F3 ew")
    _near(r["ew_abs"], 3.1445000000000003, "blocks F3 ew_abs")
    _near(r["lkfs"], 4.284515990645666, "blocks F3 lkfs")


def check_blocks_F4_all_silent(solve):
    r = solve("blocks", waveform=F4, sample_rate=25)
    _near(r["block_loudness"], [float("-inf")] * 6, "blocks F4 -inf")
    _near(r["gate_abs"], [False] * 6, "blocks F4 gate_abs")
    _near(r["n_gated_abs"], 0, "blocks F4 n_gated_abs")
    _near(r["energy_filtered_abs"], [float("nan")], "blocks F4 efa nan")
    _near(r["ew_abs"], float("nan"), "blocks F4 ew_abs nan")
    _near(r["gamma_rel"], float("nan"), "blocks F4 gamma_rel nan")
    _near(r["gate_rel"], [False] * 6, "blocks F4 gate_rel")
    _near(r["lkfs"], float("nan"), "blocks F4 lkfs nan")


def check_blocks_F5_relative_gate(solve):
    r = solve("blocks", waveform=F5, sample_rate=25)
    _near(r["gate_abs"], [True] * 6, "blocks F5 gate_abs all pass")
    _near(r["n_gated_abs"], 6, "blocks F5 n_gated_abs")
    _near(r["ew_abs"], 2.500001, "blocks F5 ew_abs")
    _near(r["gamma_rel"], -6.711598176102044, "blocks F5 gamma_rel")
    assert r["ew_abs"] != r["ew_rel"], "F5 must separate the two gate energies"
    _near(r["gate_rel"], [False, True, True, True, True, True], "blocks F5 gate_rel")
    _near(r["n_gated_rel"], 5, "blocks F5 n_gated_rel")
    _near(r["energy_filtered_rel"], [2.4000004, 0.6000004], "blocks F5 efr")
    _near(r["ew_rel"], 3.0000008, "blocks F5 ew_rel")
    _near(r["lkfs"], 4.080213705315089, "blocks F5 lkfs")


def check_blocks_F6_zero_channel(solve):
    r = solve("blocks", waveform=F6, sample_rate=25)
    _near(r["energy"], [[0.0] * 6, [1.0] * 6], "blocks F6 energy")
    _near(r["energy_filtered_abs"], [0.0, 1.0], "blocks F6 efa")
    _near(r["lkfs"], -0.691, "blocks F6 lkfs")


def check_blocks_F1B_geometry(solve):
    r = solve("blocks", waveform=F1L, sample_rate=100)
    _near([r["gate_samples"], r["step"], r["n_blocks"], r["n_channels"]],
          [40, 10, 3, 2], "blocks F1B geometry")
    _near(r["energy"], [[34.59375, 66.46875, 110.84375],
                        [17.735000000000003, 10.635, 5.535000000000001]],
          "blocks F1B energy")
    _near(r["lkfs"], 18.443805005643597, "blocks F1B lkfs")


def check_blocks_F2B(solve):
    r = solve("blocks", waveform=F2L, sample_rate=100)
    _near(r["energy"], [[0.25, 0.5, 0.75], [0.0625, 0.125, 0.1875]], "blocks F2B energy")
    _near(r["lkfs"], -2.7321998265592478, "blocks F2B lkfs")
    _near(r["gamma_rel"], -12.7321998265592478, "blocks F2B gamma_rel")


def check_blocks_validation(solve):
    _raises(lambda: solve("blocks", waveform=[[1.0] * 20] * 6, sample_rate=25),
            "blocks 6 channels", "Only up to 5 channels are supported.")
    _raises(lambda: solve("blocks", waveform=[1.0] * 20, sample_rate=25),
            "blocks 1d", "waveform must be a non-empty 2-D list")
    _raises(lambda: solve("blocks", waveform=[[1.0] * 9], sample_rate=25),
            "blocks too short", "waveform shorter than gate_samples")
    _raises(lambda: solve("blocks", waveform=[[1.0] * 20], sample_rate=2),
            "blocks zero step", "step must be positive (sample_rate too small)")
    _raises(lambda: solve("blocks", waveform=F1, sample_rate=True),
            "blocks bool sr", "sample_rate must be an int")
    _raises(lambda: solve("blocks", waveform=[[1.0, True] * 10], sample_rate=25),
            "blocks bool sample", "waveform must contain only numbers")
    _raises(lambda: solve("blocks", waveform=[[1.0] * 20, [2.0] * 19], sample_rate=25),
            "blocks ragged", "waveform must be rectangular")
    _raises(lambda: solve("blocks", waveform=[[]], sample_rate=25),
            "blocks empty col", "waveform must have at least one column")


# ------------------------------------------------------------------ loudness
def check_loudness_R1_ledger(solve):
    r = solve("loudness", waveform=RAW1, sample_rate=4000)
    _near([r["gate_samples"], r["step"], r["n_blocks"], r["n_channels"]],
          [1600, 400, 2, 1], "loudness R1 geometry")
    _near(r["block_loudness"], [-6.794206219812477, -6.783259990821409], "loudness R1 bl")
    _near(r["gate_abs"], [True, True], "loudness R1 gate_abs")
    _near(r["ew_abs"], 0.245599248923033, "loudness R1 ew_abs")
    _near(r["gamma_rel"], -16.788729656623072, "loudness R1 gamma_rel")
    _near(r["gate_rel"], [True, True], "loudness R1 gate_rel")
    _near(r["ew_rel"], 0.245599248923033, "loudness R1 ew_rel")
    _near(r["lkfs"], -6.788729656623073, "loudness R1 lkfs")
    _near(r["shape"], [1, 2000], "loudness R1 shape")


def check_loudness_R1_filtered(solve):
    r = solve("loudness", waveform=RAW1, sample_rate=4000)
    _near(r["filtered"][0][:6],
          [0.5325201153633747, 0.36181929325942197, 0.39317164295948404,
           0.30670371230747195, 0.2784480033674302, 0.242708236322913],
          "loudness R1 filtered first")
    _near(r["after_treble"][0][:6],
          [0.5647904351803231, 0.4492351396679244, 0.5288827826575371,
           0.48871597035691033, 0.5010567564228033, 0.5030784523164099],
          "loudness R1 treble first")
    _stats(r["filtered"], "loudness R1 filtered", -2.30650686501389,
           490.82708384353504, 0.8719193376517363, -0.9667415331757012)
    _stats(r["after_treble"], "loudness R1 treble", -0.03392248485294269,
           519.6976540474993, 0.6295436060002048, -0.629563732235323)


def check_loudness_R2(solve):
    r = solve("loudness", waveform=RAW2, sample_rate=4000)
    _near([r["gate_samples"], r["step"], r["n_blocks"], r["n_channels"]],
          [1600, 400, 2, 2], "loudness R2 geometry")
    _near(r["block_loudness"], [-7.763306349893038, -7.75236012090197], "loudness R2 bl")
    _near(r["lkfs"], -7.757829786703633, "loudness R2 lkfs")
    _near(r["gamma_rel"], -17.75782978670363, "loudness R2 gamma_rel")
    _near(r["shape"], [2, 2000], "loudness R2 shape")
    _stats(r["filtered"], "loudness R2 filtered", -2.7678082380163365,
           392.66166707482796, 0.6975354701213892, -0.7733932265405611)


def check_loudness_R3(solve):
    r = solve("loudness", waveform=RAW3, sample_rate=4000)
    _near([r["gate_samples"], r["step"], r["n_blocks"], r["n_channels"]],
          [1600, 400, 1, 3], "loudness R3 geometry")
    _near(r["block_loudness"], [-6.459968664942976], "loudness R3 bl")
    _near(r["lkfs"], -6.459968664942976, "loudness R3 lkfs")
    _near(r["shape"], [3, 1800], "loudness R3 shape")
    _stats(r["filtered"], "loudness R3 filtered", -4.1517123570183845,
           476.9769582633016, 0.5231516025910463, -0.5800449199054215)


def check_loudness_all_pass_relation(solve):
    for w, sr in ((RAW1, 4000), (RAW2, 4000), (RAW3, 4000)):
        r = solve("loudness", waveform=w, sample_rate=sr)
        assert all(r["gate_abs"]) and all(r["gate_rel"]), "all gates pass"
        _near(r["lkfs"], r["gamma_rel"] + 10.0,
              "loudness all-pass lkfs==gamma_rel+10 sr=%d" % sr)


def check_loudness_coeffs_and_validation(solve):
    r = solve("loudness", waveform=RAW1, sample_rate=4000)
    _near(r["coeffs"]["highpass"]["a"], C4000_HIGHPASS_A, "loudness coeffs")
    _raises(lambda: solve("loudness", waveform=[[1.0] * 2000] * 6, sample_rate=4000),
            "loudness 6 channels", "Only up to 5 channels are supported.")
    _raises(lambda: solve("loudness", waveform=[1.0] * 2000, sample_rate=4000),
            "loudness 1d", "waveform must be a non-empty 2-D list")
    _raises(lambda: solve("loudness", waveform=RAW1, sample_rate=-1),
            "loudness negative sr", "sample_rate must be >= 1")
    _raises(lambda: solve("loudness", waveform=RAW1, sample_rate=2),
            "loudness zero step", "step must be positive (sample_rate too small)")
    _raises(lambda: solve("loudness", waveform=[[1.0] * 100], sample_rate=4000),
            "loudness too short", "waveform shorter than gate_samples")


# ------------------------------------------------------------------ misc
def check_unknown_mode(solve):
    _raises(lambda: solve("nope"), "unknown mode", "unknown mode: nope")
    _raises(lambda: solve(""), "empty mode", "unknown mode: ")
    _raises(lambda: solve(None), "None mode", "unknown mode: None")


def check_input_not_mutated(solve):
    w = [list(row) for row in F1]
    snapshot = [list(row) for row in w]
    solve("blocks", waveform=w, sample_rate=25)
    assert w == snapshot, "blocks mutated its input"
    w2 = [list(row) for row in RAW1]
    snapshot2 = [list(row) for row in w2]
    solve("loudness", waveform=w2, sample_rate=4000)
    assert w2 == snapshot2, "loudness mutated its input"
    w3 = list(W1)
    solve("lfilter", waveform=w3, a_coeffs=CS_A, b_coeffs=CS_B)
    assert w3 == W1, "lfilter mutated its input"


def check_five_modes_return_dicts(solve):
    r1 = solve("coeffs", sample_rate=8000)
    r2 = solve("lfilter", waveform=W1, a_coeffs=CS_A, b_coeffs=CS_B)
    r3 = solve("kweight", waveform=[W1], sample_rate=8000)
    r4 = solve("blocks", waveform=F1, sample_rate=25)
    r5 = solve("loudness", waveform=RAW1, sample_rate=4000)
    for i, r in enumerate((r1, r2, r3, r4, r5), 1):
        assert isinstance(r, dict), "mode %d must return a dict" % i
    assert set(r1) == {"sample_rate", "treble", "highpass"}, "coeffs keys"
    assert "lkfs" in r4 and "lkfs" in r5, "both gating modes expose lkfs"


# ------------------------------------------------------------------ hardening cases (patch861)
F7 = [[0.00036] * 20]
F8 = [[1.0] * 10]
F9 = [[0.00019] * 20]
F10 = [[0.00036] * 10 + [0.00019] * 10]
F3L = [[0.5 + 0.1 * c for _ in range(20)] for c in range(5)]


def check_blocks_F7_gamma_abs_boundary(solve):
    """Block loudness -69.565 sits strictly between -70 and -60."""
    r = solve("blocks", waveform=F7, sample_rate=25)
    _near(r["block_loudness"], [-69.56494998465426] * 6, "F7 bl")
    _near(r["gate_abs"], [True] * 6, "F7 gate_abs")
    _near(r["n_gated_abs"], 6, "F7 n_gated_abs")
    _near(r["ew_abs"], 1.2960000000000002e-07, "F7 ew_abs")
    _near(r["gamma_rel"], -79.56494998465426, "F7 gamma_rel")
    _near(r["lkfs"], -69.56494998465426, "F7 lkfs")


def check_blocks_F8_exact_gate_length(solve):
    """time_len == gate_samples is legal: exactly one block."""
    r = solve("blocks", waveform=F8, sample_rate=25)
    _near([r["gate_samples"], r["step"], r["n_blocks"]], [10, 2, 1], "F8 geometry")
    _near(r["energy"], [[1.0]], "F8 energy")
    _near(r["block_loudness"], [-0.691], "F8 bl")
    _near(r["lkfs"], -0.691, "F8 lkfs")


def check_blocks_F9_all_below_abs_gate(solve):
    """Finite energies whose block loudness is -75.1: absolute gate drops all."""
    r = solve("blocks", waveform=F9, sample_rate=25)
    _near(r["block_loudness"], [-75.11592798094343] * 6, "F9 bl")
    _near(r["gate_abs"], [False] * 6, "F9 gate_abs")
    _near(r["n_gated_abs"], 0, "F9 n_gated_abs")
    _near(r["energy_filtered_abs"], [float("nan")], "F9 efa nan")
    _near(r["gamma_rel"], float("nan"), "F9 gamma_rel nan")
    _near(r["gate_rel"], [False] * 6, "F9 gate_rel")
    _near(r["lkfs"], float("nan"), "F9 lkfs nan")


def check_blocks_F10_mixed_gates(solve):
    """Five blocks fail the absolute gate yet exceed gamma_rel."""
    r = solve("blocks", waveform=F10, sample_rate=25)
    _near(r["energy"], [[1.2960000000000002e-07, 1.1090000000000001e-07,
                         9.220000000000002e-08, 7.350000000000002e-08,
                         5.480000000000001e-08, 3.6100000000000006e-08]], "F10 energy")
    _near(r["block_loudness"],
          [-69.56494998465426, -70.24168453850841, -71.04369078946371,
           -72.02812660915805, -73.30319441515631, -75.11592798094343],
          "F10 bl")
    _near(r["gate_abs"], [True, False, False, False, False, False], "F10 gate_abs")
    _near(r["ew_abs"], 1.2960000000000002e-07, "F10 ew_abs")
    _near(r["gamma_rel"], -79.56494998465426, "F10 gamma_rel")
    _near(r["gate_rel"], [True, False, False, False, False, False], "F10 gate_rel")
    _near(r["n_gated_rel"], 1, "F10 n_gated_rel")
    _near(r["lkfs"], -69.56494998465426, "F10 lkfs")


def check_blocks_F3L_bankers_step(solve):
    """sr=35 -> gate_samples=14, step=round(3.5)=4 (truncation would give 3)."""
    r = solve("blocks", waveform=F3L, sample_rate=35)
    _near([r["gate_samples"], r["step"], r["n_blocks"], r["n_channels"]],
          [14, 4, 2, 5], "F3L geometry")
    _near(r["energy_weighted"], [3.1445000000000003] * 2, "F3L ew")
    _near(r["g"], [1.0, 1.0, 1.0, 1.41, 1.41], "F3L g")
    _near(r["lkfs"], 4.284515990645666, "F3L lkfs")


def check_blocks_sr_one_step_error(solve):
    _raises(lambda: solve("blocks", waveform=F1, sample_rate=1),
            "blocks sr=1", "step must be positive (sample_rate too small)")


def check_lfilter_default_clamp(solve):
    r = solve("lfilter", waveform=W3, a_coeffs=C8000_HIGHPASS_A,
              b_coeffs=C8000_HIGHPASS_B)
    _near(r["out"], LF_HP_W3_CL1, "lfilter default clamp")
    assert r["clamp"] is True, "default clamp must be True"


def check_lfilter_empty_coeffs(solve):
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=[], b_coeffs=CS_B),
            "lfilter empty a", "a_coeffs must be a non-empty list of numbers")
    _raises(lambda: solve("lfilter", waveform=W1, a_coeffs=CS_A, b_coeffs=[]),
            "lfilter empty b", "b_coeffs must be a non-empty list of numbers")
