#!/usr/bin/env python3
"""862 judge cases: torchaudio MVDR beamforming ledger.

Every check takes `solve` and asserts on explicit, hand-verified values taken
from the verbatim official implementation (see notes.md / fixtures862.txt).
Complex values are [re, im] pairs.  No torch, no import from the solution
module, no global RNG.  Helpers use internal `assert`, never a returned bool.
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
    assert abs(float(got) - float(want)) <= atol + rtol * abs(float(want)), \
        "%s: %r != %r" % (tag, got, want)


def _raises(fn, tag, needle):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert needle in str(e), "%s: message %r lacks %r" % (tag, str(e), needle)
    else:
        raise AssertionError(tag + ": no exception")


def _flat(v):
    out = []
    for e in v:
        if isinstance(e, list):
            out.extend(_flat(e))
        else:
            out.append(e)
    return out


def _sum(v):
    return math.fsum(float(x) for x in _flat(v) if isinstance(x, (int, float)))


def _sq(v):
    return math.fsum(float(x) * float(x) for x in _flat(v) if isinstance(x, (int, float)))


# ------------------------------------------------------------------ fixtures
# specgram: (channel=2, freq=3, time=4)
SG = [[[[0.2 * (c + 1) + 0.01 * (f + t), 0.1 * (c - f + t)]
        for t in range(4)] for f in range(3)] for c in range(2)]
# (freq=2, C=2, C=2) complex PSD pairs
S = [[[[2.0, 0.1], [0.3, 0.2]], [[0.3, -0.2], [1.5, 0.0]]],
     [[[1.2, 0.0], [0.1, 0.4]], [[0.1, -0.4], [2.2, 0.1]]]]
N = [[[[3.0, 0.0], [0.2, 0.1]], [[0.2, -0.1], [2.0, 0.0]]],
     [[[2.0, 0.0], [0.1, 0.0]], [[0.1, 0.0], [3.0, 0.0]]]]
RTF = [[[0.6, 0.2], [0.3, -0.4]], [[0.7, 0.2], [0.25, -0.4]]]
W3 = [[[0.2 + 0.01 * f, 0.1], [0.3 - 0.01 * f, -0.2]] for f in range(3)]
SG1 = [[[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]]]
MASK = [[1.0, 1.0], [2.0, 2.0]]
ONEHOT0 = [[1.0, 0.0], [0.0, 0.0]]
ONEHOT1 = [[0.0, 0.0], [1.0, 0.0]]

PSD_MAIN = [[[[0.3254, 9.769962616701377e-18], [0.5574000000000001, 0.03400000000000003]], [[0.5574000000000001, -0.034], [0.9894000000000002, 2.0539125955565393e-17]]], [[[0.263, 6.938893903907227e-19], [0.4630000000000001, -0.05000000000000001]], [[0.4630000000000001, 0.05000000000000003], [0.8630000000000002, 2.3314683517128286e-17]]], [[[0.28140000000000004, -2.525757381022231e-18], [0.4494000000000001, -0.13400000000000004]], [[0.4494000000000001, 0.13400000000000004], [0.8174000000000001, 1.8318679906315084e-18]]]]
TRACE_MAIN = [[3.5, 0.1], [3.4000000000000004, 0.1]]
TRACE_DIAG = [[[2.0, 0.1], [1.5, 0.0]], [[1.2, 0.0], [2.2, 0.1]]]
TIK_OUT = [[[[3.00000051, 0.0], [0.2, 0.1]], [[0.2, -0.1], [2.00000051, 0.0]]], [[[2.00000051, 0.0], [0.1, 0.0]], [[0.1, 0.0], [3.00000051, 0.0]]]]
SOUDEN_R0 = [[[0.4703573180837463, 0.0139003040235231], [0.057512256120737314, -0.05173890265535765]], [[0.44971782639639674, -0.006258592093281915], [0.007507799662502332, -0.10043879121757104]]]
SOUDEN_R1 = [[[0.03666898455444739, 0.029096666829479475], [0.5296426747860618, -0.013900303852535285]], [[0.013753836944577822, 0.1487780982079294], [0.5502821661020498, 0.006258592281290775]]]
SOUDEN_NUMERATOR = [[[[0.6588235294117647, 0.03529411764705883], [0.05042016806722688, 0.04201680672268907]], [[0.08235294117647059, -0.07058823529411766], [0.742857142857143, -0.0016806722689075633]]], [[[0.5993322203672787, 0.00667779632721202], [0.013355592654424035, 0.1986644407345576]], [[0.013355592654424042, -0.1335559265442404], [0.7328881469115193, 0.02671118530884808]]]]
SOUDEN_TRACE_NUM = [[1.4016806722689077, 0.033613445378151266], [1.332220367278798, 0.0333889816360601]]
SOUDEN_WS = [[[[0.4703573180837463, 0.0139003040235231], [0.03666898455444739, 0.029096666829479475]], [[0.057512256120737314, -0.05173890265535765], [0.5296426747860618, -0.013900303852535285]]], [[[0.44971782639639674, -0.006258592093281915], [0.013753836944577822, 0.1487780982079294]], [[0.007507799662502332, -0.10043879121757104], [0.5502821661020498, 0.006258592281290775]]]]
SOUDEN_LOADED = [[[0.4703573397947382, 0.013900303766632779], [0.05751226180095608, -0.051738904299868854]], [[0.4497178053027905, -0.006258593078806503], [0.007507803682739627, -0.10043879510554998]]]
RTF_NONE = [[[0.758620658525566, 0.31034481485136794], [0.5241379095267547, -0.8137930700546984]], [[1.029265842434016, 0.31746030802784103], [0.21329364445620566, -0.40674601966067137]]]
RTF_NUMERATOR = [[[0.18487394957983194, 0.07563025210084034], [0.12773109243697478, -0.19831932773109248]], [[0.34641068447412354, 0.10684474123539232], [0.0717863105175292, -0.13689482470784642]]]
RTF_DENOMINATOR = [[0.24369747899159666, -1.0408340855860843e-17], [0.3365609348914858, -7.806255641895632e-18]]
RTF_R0 = [[[0.5172413580856132, 0.03448275720570755], [0.1517241317051131, -0.59310342393817]], [[0.7839781513093793, 0.016369047132685506], [0.06795634718720968, -0.32738094265371104]]]
RTF_SCALE = [[0.6, -0.2], [0.7, -0.2]]
RTF_LOADED = [[[0.5172413805231771, 0.034482749322237355], [0.15172413219024847, -0.593103393859695]], [[0.7839781361450795, 0.016369044016599176], [0.06795635828346275, -0.3273809638141083]]]
POWER_N1 = [[[2.0, 0.1], [0.30000000000000004, -0.20000000000000004]], [[1.2, 0.0], [0.1, -0.39999999999999997]]]
POWER_N2 = [[[1.3529411764705883, 0.13176470588235295], [0.32823529411764707, -0.22705882352941179]], [[0.773956594323873, 0.0], [0.10534223706176962, -0.5315525876460768]]]
POWER_N3 = [[[0.9297775580820564, 0.13151359367276322], [0.31796539792387546, -0.21832328225407815]], [[0.5362560305015871, 0.0018015557370241447], [0.10883832542272737, -0.5416896831391217]]]
POWER_N4 = [[[0.6495572512742122, 0.11792823537719585], [0.2889751189056938, -0.19408396050461713]], [[0.39520701573878037, 0.004800138491303063], [0.10899601626729358, -0.5009385948793403]]]
POWER_PHI = [[[[0.6588235294117647, 0.03529411764705883], [0.05042016806722688, 0.04201680672268907]], [[0.08235294117647059, -0.07058823529411766], [0.742857142857143, -0.0016806722689075633]]], [[[0.5993322203672787, 0.00667779632721202], [0.013355592654424035, 0.1986644407345576]], [[0.013355592654424042, -0.1335559265442404], [0.7328881469115193, 0.02671118530884808]]]]
POWER_INIT = [[[0.6588235294117647, 0.03529411764705883], [0.08235294117647059, -0.07058823529411766]], [[0.5993322203672787, 0.00667779632721202], [0.013355592654424042, -0.1335559265442404]]]
POWER_LOADED = [[[0.929777247358993, 0.13151354971994605], [0.31796529210578117, -0.21832319973450612]], [[0.5362557699370956, 0.00180155513558433], [0.10883829375085946, -0.5416894575954215]]]
BEAM_OUT = [[[0.14, 0.09000000000000001], [0.135, 0.14100000000000001], [0.13, 0.192], [0.125, 0.24300000000000002]], [[0.15300000000000002, 0.04000000000000001], [0.14800000000000002, 0.09100000000000001], [0.14300000000000002, 0.14200000000000002], [0.138, 0.193]], [[0.166, -0.009999999999999981], [0.161, 0.041000000000000016], [0.156, 0.092], [0.15099999999999997, 0.14300000000000002]]]
BEAM_CONJ = [[[0.2, -0.1], [0.3, 0.2]], [[0.21000000000000002, -0.1], [0.29, 0.2]], [[0.22, -0.1], [0.27999999999999997, 0.2]]]


# ------------------------------------------------------------------ psd
def check_psd_main(solve):
    r = solve("psd", specgram=SG)
    _near(r["psd"], PSD_MAIN, "psd main")
    _near(r["shape"], [3, 2, 2], "psd shape")
    _near(r["sum_dim"], -3, "psd sum dim")
    assert r["mask_norm"] is None, "psd mask_norm must be None"


def check_psd_signature(solve):
    r = solve("psd", specgram=SG)
    _near(_sum(r["psd"]), 6.479200000000001, "psd sum")
    _near(_sq(r["psd"]), 4.143336480000001, "psd sq")
    assert _flat(r["psd"])[0] == 0.3254, "psd first"


def check_psd_hermitian(solve):
    r = solve("psd", specgram=SG)
    for p in r["psd"]:
        _near([p[0][1][0], p[0][1][1]], [p[1][0][0], -p[1][0][1]], "psd hermitian")
        assert abs(p[0][0][1]) < 1e-12 and abs(p[1][1][1]) < 1e-12, "psd diag imag"


def check_psd_mask_normalize(solve):
    r = solve("psd", specgram=SG1, mask=MASK, normalize=True)
    _near(r["psd"], [[[[2.499999999875, 0.0]]], [[[12.4999999996875, 0.0]]]], "psd norm")
    _near(r["mask_norm"], [[0.499999999975, 0.499999999975],
                          [0.4999999999875, 0.4999999999875]], "psd mask_norm")
    _near(r["shape"], [2, 1, 1], "psd norm shape")


def check_psd_mask_raw(solve):
    r = solve("psd", specgram=SG1, mask=MASK, normalize=False)
    _near(r["psd"], [[[[5.0, 0.0]]], [[[50.0, 0.0]]]], "psd raw")
    _near(r["mask_norm"], [[1.0, 1.0], [2.0, 2.0]], "psd raw mask_norm")


def check_psd_eps_enters_denominator(solve):
    r = solve("psd", specgram=SG1, mask=MASK, normalize=True, eps=1.0)
    _near(r["mask_norm"][0], [1.0 / 3.0, 1.0 / 3.0], "psd eps=1")


def check_psd_bad_mask_raises(solve):
    _raises(lambda: solve("psd", specgram=SG1,
                          mask=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            "psd bad freq", "should be the same as specgram")
    _raises(lambda: solve("psd", specgram=SG1, mask=[[1.0, 1.0, 1.0]]),
            "psd bad time", "should be the same as specgram")


def check_psd_single_frame(solve):
    r = solve("psd", specgram=[[[[1.0, 2.0], [3.0, -1.0]]]])
    # |1+2j|^2 + |3-1j|^2 = 5 + 10 = 15
    _near(r["psd"][0][0][0], [15.0, 0.0], "psd single |z|^2")


# ------------------------------------------------------------------ trace
def check_trace_main(solve):
    r = solve("trace", mat=S)
    _near(r["trace"], TRACE_MAIN, "trace main")
    _near(r["diag"], TRACE_DIAG, "trace diag")
    _near(r["shape"], [2], "trace shape")


def check_trace_complex(solve):
    r = solve("trace", mat=S)
    # trace is the SUM of the complex diagonal (imag 0.1 survives)
    _near(r["trace"][0], [3.5, 0.1], "trace complex imag")


def check_trace_dim_swap(solve):
    r = solve("trace", mat=S, dim1=-2, dim2=-1)
    _near(r["trace"], TRACE_MAIN, "trace dim swap")


def check_trace_non_square_raises(solve):
    _raises(lambda: solve("trace", mat=[[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]),
            "trace non-square", "must be the same")


# ------------------------------------------------------------------ tik_reg
def check_tik_reg_main(solve):
    r = solve("tik_reg", mat=N)
    _near(r["out"], TIK_OUT, "tik out")
    _near(r["shape"], [2, 2, 2], "tik shape")
    _near(r["channels"], 2, "tik channels")


def check_tik_reg_ledger(solve):
    r = solve("tik_reg", mat=N)
    _near(r["trace_real"], [5.0, 5.0], "tik trace_real")
    _near(r["epsilon"], [[[5.1e-07]], [[5.1e-07]]], "tik epsilon")
    _near(r["reg"], 1e-7, "tik reg")
    _near(r["eps"], 1e-8, "tik eps")


def check_tik_reg_offdiag_untouched(solve):
    r = solve("tik_reg", mat=N)
    for f in range(2):
        _near(r["out"][f][0][1], N[f][0][1], "tik offdiag01")
        _near(r["out"][f][1][0], N[f][1][0], "tik offdiag10")


def check_tik_reg_real_only(solve):
    r = solve("tik_reg", mat=N, reg=0.01, eps=0.02)
    _near(_sum(r["out"]), 10.879999999999999, "tik big sum")
    _near(r["out"][0][0][0], [3.07, 0.0], "tik big diag00")
    # loading only touches the diagonal, with a REAL epsilon: imaginary parts
    # of the diagonal are preserved, off-diagonal entries are untouched.
    _near(r["out"][0][1][0], [0.2, -0.1], "tik offdiag imag preserved")
    Z = [[[[2.0, 0.5], [0.1, 0.2]], [[0.1, -0.2], [3.0, -0.7]]]]
    rz = solve("tik_reg", mat=Z, reg=0.01, eps=0.02)
    # trace.real = 2.0 + 3.0 = 5.0 -> epsilon = 0.07
    _near(rz["epsilon"][0][0][0], 0.07, "tik complex-diag epsilon from real trace")
    _near(rz["out"][0][0][0], [2.07, 0.5], "tik diag00 imag preserved")
    _near(rz["out"][0][1][1], [3.07, -0.7], "tik diag11 imag preserved")
    _near(rz["out"][0][0][1], [0.1, 0.2], "tik offdiag unchanged (complex diag)")


def check_tik_reg_zero_matrix(solve):
    z = [[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    r = solve("tik_reg", mat=z, reg=1e-7, eps=1e-8)
    _near(r["epsilon"][0][0][0], 1e-8, "tik zero epsilon")
    _near(r["out"][0][0][0], [1e-8, 0.0], "tik zero diag")


def check_tik_reg_non_square_raises(solve):
    _raises(lambda: solve("tik_reg", mat=[[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]),
            "tik non-square", "should be the same")


# ------------------------------------------------------------------ souden
def check_souden_r0(solve):
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=0, diagonal_loading=False)
    _near(r["weights"], SOUDEN_R0, "souden r0")
    _near(r["shape"], [2, 2], "souden shape")
    _near(r["branch"], "int", "souden branch")


def check_souden_ledger(solve):
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=0, diagonal_loading=False)
    _near(r["numerator"], SOUDEN_NUMERATOR, "souden numerator")
    _near(r["trace_numerator"], SOUDEN_TRACE_NUM, "souden trace_numerator")
    _near(r["ws"], SOUDEN_WS, "souden ws")
    _near(_sq(r["numerator"]), 1.9579497041535954, "souden numerator sq")


def check_souden_complex_trace_denominator(solve):
    # the denominator is the COMPLEX trace of numerator, not its real part
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=0, diagonal_loading=False)
    tn = r["trace_numerator"]
    assert abs(tn[0][1]) > 0.01, "souden trace must be complex"
    num = r["numerator"][0][0][0]
    tr = tn[0]
    d = tr[0] + 1e-8, tr[1]
    den = d[0] * d[0] + d[1] * d[1]
    want = [(num[0] * d[0] + num[1] * d[1]) / den,
            (num[1] * d[0] - num[0] * d[1]) / den]
    _near(r["ws"][0][0][0], want, "souden ws complex division")


def check_souden_r1_column(solve):
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=1, diagonal_loading=False)
    _near(r["weights"], SOUDEN_R1, "souden r1")


def check_souden_tensor_onehot(solve):
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=ONEHOT0,
              diagonal_loading=False)
    _near(r["branch"], "tensor", "souden tensor branch")
    _near(r["weights"], SOUDEN_R0, "souden tensor == int r0")


def check_souden_diagonal_loading(solve):
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=0,
              diagonal_loading=True, diag_eps=1e-7)
    _near(r["weights"], SOUDEN_LOADED, "souden loaded")
    _near(r["psd_n_reg"], TIK_OUT, "souden psd_n_reg")


def check_souden_shape_mismatch_raises(solve):
    _raises(lambda: solve("souden", psd_s=S, psd_n=N[0], reference_channel=0),
            "souden ndim", "at least 3D")


def check_souden_bad_rc_type(solve):
    _raises(lambda: solve("souden", psd_s=S, psd_n=N, reference_channel="0"),
            "souden bad rc", 'Expected "int" or "Tensor"')


# ------------------------------------------------------------------ rtf
def check_rtf_none(solve):
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=None, diagonal_loading=False)
    _near(r["weights"], RTF_NONE, "rtf none")
    _near(r["shape"], [2, 2], "rtf shape")
    _near(r["branch"], "none", "rtf branch none")


def check_rtf_ledger(solve):
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=None, diagonal_loading=False)
    _near(r["numerator"], RTF_NUMERATOR, "rtf numerator")
    _near(r["denominator"], RTF_DENOMINATOR, "rtf denominator")
    _near(r["denominator_real"], [0.24369747899159666, 0.3365609348914858],
          "rtf denominator_real")
    _near(r["weights_unnorm"], RTF_NONE, "rtf unnorm == weights when no scale")
    assert r["scale"] is None, "rtf scale None"


def check_rtf_real_denominator(solve):
    # official divides by denominator.real only
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=None, diagonal_loading=False)
    num = r["numerator"][0][0]
    den = r["denominator_real"][0] + 1e-8
    _near(r["weights"][0][0], [num[0] / den, num[1] / den], "rtf real denominator")


def check_rtf_scale_int(solve):
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=0, diagonal_loading=False)
    _near(r["weights"], RTF_R0, "rtf r0")
    _near(r["scale"], RTF_SCALE, "rtf scale")
    _near(r["branch"], "int", "rtf branch int")


def check_rtf_scale_tensor(solve):
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=ONEHOT0,
              diagonal_loading=False)
    _near(r["branch"], "tensor", "rtf branch tensor")
    _near(r["weights"], RTF_R0, "rtf tensor == int")


def check_rtf_diagonal_loading(solve):
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=0,
              diagonal_loading=True, diag_eps=1e-7)
    _near(r["weights"], RTF_LOADED, "rtf loaded")


def check_rtf_shape_mismatch_raises(solve):
    _raises(lambda: solve("rtf", rtf=[[[1.0, 0.0]]], psd_n=N),
            "rtf shape", "should be the same")


# ------------------------------------------------------------------ rtf_power
def check_power_n1_identity(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              n_iter=1, diagonal_loading=False)
    _near(r["rtf"], POWER_N1, "power n1")
    _near(r["n_loop"], 0, "power n1 n_loop")
    _near(r["tail"], "psd_n", "power n1 tail")
    # psd_n @ solve(psd_n,psd_s)[...,0] == psd_s[...,0]
    _near(r["rtf"][0], [S[0][0][0], S[0][1][0]], "power n1 identity")


def check_power_n2(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              n_iter=2, diagonal_loading=False)
    _near(r["rtf"], POWER_N2, "power n2")
    _near(r["n_loop"], 0, "power n2 n_loop")
    _near(r["tail"], "psd_s", "power n2 tail")


def check_power_n3(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              n_iter=3, diagonal_loading=False)
    _near(r["rtf"], POWER_N3, "power n3")
    _near(r["n_loop"], 1, "power n3 n_loop")
    _near(r["phi"], POWER_PHI, "power n3 phi")
    _near(r["rtf_init"], POWER_INIT, "power n3 init")
    _near(r["tail"], "psd_s", "power n3 tail")


def check_power_n4(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              n_iter=4, diagonal_loading=False)
    _near(r["rtf"], POWER_N4, "power n4")
    _near(r["n_loop"], 2, "power n4 n_loop")


def check_power_tensor_onehot(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=ONEHOT0,
              n_iter=3, diagonal_loading=False)
    _near(r["branch"], "tensor", "power tensor branch")
    _near(r["rtf"], POWER_N3, "power tensor == int")


def check_power_loaded(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              n_iter=3, diagonal_loading=True, diag_eps=1e-7)
    _near(r["rtf"], POWER_LOADED, "power loaded")


def check_power_signature(solve):
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              n_iter=3, diagonal_loading=False)
    _near(_sum(r["rtf"]), 1.266139495946834, "power n3 sum")
    _near(_sq(r["rtf"]), 1.6233964524374171, "power n3 sq")


def check_power_iter_zero_raises(solve):
    _raises(lambda: solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0, n_iter=0),
            "power n_iter=0", "must be greater than 0")
    _raises(lambda: solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0, n_iter=-1),
            "power n_iter<0", "must be greater than 0")


# ------------------------------------------------------------------ beamform
def check_beamform_main(solve):
    r = solve("beamform", weights=W3, specgram=SG)
    _near(r["out"], BEAM_OUT, "beamform out")
    _near(r["shape"], [3, 4], "beamform shape")
    _near(r["weights_conj"], BEAM_CONJ, "beamform conj")


def check_beamform_signature(solve):
    r = solve("beamform", weights=W3, specgram=SG)
    _near(_sum(r["out"]), 3.144, "beamform sum")
    _near(_sq(r["out"]), 0.4776520000000001, "beamform sq")


def check_beamform_conjugate(solve):
    # out[f,t] = sum_c conj(w[f,c]) * sg[c,f,t]
    r = solve("beamform", weights=W3, specgram=SG)
    re, im = 0.0, 0.0
    for c in range(2):
        wr, wi = W3[0][c]
        sr, si = SG[c][0][0]
        re += wr * sr + wi * si
        im += wr * si - wi * sr
    _near(r["out"][0][0], [re, im], "beamform manual conj")


def check_beamform_freq_mismatch_raises(solve):
    _raises(lambda: solve("beamform", weights=W3[:2], specgram=SG),
            "beamform freq mismatch", "must match")


# ------------------------------------------------------------------ mode dispatch
def check_all_modes_dispatch(solve):
    assert solve("psd", specgram=SG)["shape"] == [3, 2, 2]
    assert solve("trace", mat=S)["shape"] == [2]
    assert solve("tik_reg", mat=N)["channels"] == 2
    assert solve("souden", psd_s=S, psd_n=N, reference_channel=0)["shape"] == [2, 2]
    assert solve("rtf", rtf=RTF, psd_n=N)["shape"] == [2, 2]
    assert solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0)["shape"] == [2, 2]
    assert solve("beamform", weights=W3, specgram=SG)["shape"] == [3, 4]


def check_unknown_mode(solve):
    _raises(lambda: solve("evd", psd_s=S), "unknown mode", "unknown mode")
    try:
        solve("evd", psd_s=S)
    except (ValueError, TypeError) as e:
        msg = str(e)
        for m in ("psd", "trace", "tik_reg", "souden", "rtf",
                  "rtf_power", "beamform"):
            assert m in msg, \
                "unknown-mode message must list MODES entry %r: %r" % (m, msg)
        assert "evd" in msg, "unknown-mode message must echo mode: %r" % msg
    else:
        raise AssertionError("unknown mode must raise")


def check_reference_channel_bad_tensor_leaf(solve):
    """A tensor reference_channel with non-numeric leaves is rejected by
    pack() itself; the leaf-level rectangular error must name the argument."""
    _raises(lambda: solve("souden", psd_s=S, psd_n=N,
                          reference_channel=[[True, False]]),
            "souden bool tensor rc", "reference_channel must be rectangular")


def check_souden_singular_raises(solve):
    """A singular psd_n must be reported, not silently divided by zero."""
    Z = [[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    _raises(lambda: solve("souden", psd_s=S[:1], psd_n=Z,
                          reference_channel=0, diagonal_loading=False),
            "souden singular psd_n", "singular")


# ------------------------------------------------------------------ hardening cases (patch862)
SMALL_MASK = [[1e-9, 1e-9], [2e-9, 2e-9]]
NH_N = [[[[1.0, 0.5], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.25]]]]
NH_RTF = [[[0.6, 0.2], [0.3, -0.4]]]
S_PIV = [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]]
N_PIV = [[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]]

PSD_DEFAULT_NORM = [[[[2.499999999875, 0.0]]], [[[12.4999999996875, 0.0]]]]
PSD_SMALL_EPS10 = [[0.47619047619047616, 0.47619047619047616],
                   [0.4878048780487805, 0.4878048780487805]]
SOUDEN_DEFAULT = [[[0.4703573397947382, 0.013900303766632779], [0.05751226180095608, -0.051738904299868854]], [[0.4497178053027905, -0.006258593078806503], [0.007507803682739627, -0.10043879510554998]]]
RTF_DEFAULT_RC = [[[0.758620658525566, 0.31034481485136794], [0.5241379095267547, -0.8137930700546984]], [[1.029265842434016, 0.31746030802784103], [0.21329364445620566, -0.40674601966067137]]]
POWER_DEFAULT_N = [[[0.9297775580820564, 0.13151359367276322], [0.31796539792387546, -0.21832328225407815]], [[0.5362560305015871, 0.0018015557370241447], [0.10883832542272737, -0.5416896831391217]]]
TIK_EYE = [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]
SOUDEN_EPS05_WS = [[[[0.3466625178687541, 0.012431948424767934], [0.02689561055749341, 0.021619167290190124]], [[0.04263592600017685, -0.03787248653999429], [0.390494281510644, -0.007786023535318497]]], [[[0.32706486500505744, -0.002315532847472572], [0.00926213138989029, 0.10825946002022979]], [[0.005958969836864903, -0.07300153020203855], [0.40013279041419203, 0.007286873978784657]]]]
NH_DENOMINATOR = [[0.5552941176470587, -0.21882352941176467]]
NH_DEN_REAL = [0.5552941176470587]
NH_WEIGHTS = [[[1.008474558110098, -0.14406779401572828], [0.3389830447428901, -0.8050847312643641]]]
NH_SOUDEN_NREG = [[[[1.00000021, 0.5], [0.0, 0.0]], [[0.0, 0.0], [1.00000021, 0.25]]]]
PIV_NUMERATOR = [[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]]
PSD_INT_SINGLE = [[[[30.0, 0.0]]]]


def check_default_psd_normalize(solve):
    """Omitting normalize must default to True."""
    r = solve("psd", specgram=SG1, mask=MASK)
    _near(r["psd"], PSD_DEFAULT_NORM, "default normalize=True")


def check_default_psd_eps(solve):
    """Default eps must be 1e-10, not the docstring's 1e-15."""
    r = solve("psd", specgram=SG1, mask=SMALL_MASK)
    _near(r["mask_norm"], PSD_SMALL_EPS10, "default eps=1e-10")


def check_default_souden_loading(solve):
    """Omitting diagonal_loading / diag_eps must default to True / 1e-7."""
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=0)
    _near(r["weights"], SOUDEN_DEFAULT, "souden default loading")


def check_default_rtf_rc(solve):
    """Omitting reference_channel means None (no scaling)."""
    r = solve("rtf", rtf=RTF, psd_n=N, diagonal_loading=False)
    _near(r["weights"], RTF_DEFAULT_RC, "rtf default rc=None")
    _near(r["branch"], "none", "rtf default branch")


def check_default_power_n_iter(solve):
    """Omitting n_iter must default to 3."""
    r = solve("rtf_power", psd_s=S, psd_n=N, reference_channel=0,
              diagonal_loading=False)
    _near(r["rtf"], POWER_DEFAULT_N, "power default n_iter=3")


def check_trace_shape_F_ne_C(solve):
    r = solve("trace", mat=S[:1])
    _near(r["shape"], [1], "trace shape F=1")


def check_souden_shape_F_ne_C(solve):
    r = solve("souden", psd_s=S[:1], psd_n=N[:1], reference_channel=0,
              diagonal_loading=False)
    _near(r["shape"], [1, 2], "souden shape F=1,C=2")
    _near(r["weights"], SOUDEN_R0[:1], "souden F=1 weights")


def check_tik_eye_ledger(solve):
    r = solve("tik_reg", mat=N)
    _near(r["eye"], TIK_EYE, "tik eye")


def check_souden_big_eps(solve):
    """A large eps exposes the exact ws denominator."""
    r = solve("souden", psd_s=S, psd_n=N, reference_channel=0,
              diagonal_loading=False, eps=0.5)
    _near(r["ws"], SOUDEN_EPS05_WS, "souden ws eps=0.5")


def check_rtf_complex_denominator(solve):
    """Non-Hermitian psd_n -> complex denominator; .real != abs."""
    r = solve("rtf", rtf=NH_RTF, psd_n=NH_N, diagonal_loading=False)
    _near(r["denominator"], NH_DENOMINATOR, "rtf complex denominator")
    _near(r["denominator_real"], NH_DEN_REAL, "rtf denominator .real")
    _near(r["weights"], NH_WEIGHTS, "rtf weights complex den")
    assert abs(r["denominator"][0][1]) > 0.1, "denominator must have imag part"


def check_tik_core_real_trace(solve):
    """diagonal_loading uses trace.real even for complex diagonal."""
    r = solve("souden", psd_s=NH_N, psd_n=NH_N, reference_channel=0,
              diagonal_loading=True, diag_eps=1e-7)
    _near(r["psd_n_reg"], NH_SOUDEN_NREG, "souden nreg complex diag")


def check_rtf_unnorm_vs_scaled(solve):
    """weights_unnorm excludes scale; weights includes it."""
    r = solve("rtf", rtf=RTF, psd_n=N, reference_channel=0, diagonal_loading=False)
    _near(r["weights_unnorm"], RTF_NONE, "rtf unnorm no scale")
    _near(r["weights"], RTF_R0, "rtf weights scaled")


def check_lu_pivoting(solve):
    """A[0][0]==0 requires partial pivoting."""
    r = solve("souden", psd_s=S_PIV, psd_n=N_PIV, reference_channel=0,
              diagonal_loading=False)
    _near(r["numerator"], PIV_NUMERATOR, "solve with pivot")


def check_psd_int_leaves(solve):
    """int leaves are valid numbers."""
    r = solve("psd", specgram=[[[[1, 0], [2, 0], [3, 0], [4, 0]]]])
    _near(r["psd"], PSD_INT_SINGLE, "psd int leaves")


def check_psd_bool_leaf_raises(solve):
    _raises(lambda: solve("psd", specgram=[[[[1.0, True], [2.0, 0.0],
                                            [3.0, 0.0], [4.0, 0.0]]]]),
            "psd bool leaf", "must be rectangular")


def check_psd_triple_leaf_raises(solve):
    _raises(lambda: solve("psd", specgram=[[[[1.0, 0.0, 9.0], [2.0, 0.0],
                                            [3.0, 0.0], [4.0, 0.0]]]]),
            "psd triple leaf", "must be rectangular")


def check_trace_low_dim_raises(solve):
    _raises(lambda: solve("trace", mat=[[1.0, 0.0]]),
            "trace 1D", "at least 2D")


def check_souden_bad_bool_rc(solve):
    _raises(lambda: solve("souden", psd_s=S, psd_n=N, reference_channel=True),
            "souden bool rc", 'Expected "int" or "Tensor"')


def check_souden_psd_shape_mismatch(solve):
    _raises(lambda: solve("souden", psd_s=S, psd_n=S[:1], reference_channel=0),
            "souden psd shape diff", "should be the same")


def check_souden_non_square_psd(solve):
    ns = [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]
    _raises(lambda: solve("souden", psd_s=ns, psd_n=ns, reference_channel=0),
            "souden non-square", "last two dimensions")


def check_souden_non_square_3d(solve):
    ns = [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
          [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]
    _raises(lambda: solve("souden", psd_s=ns, psd_n=ns, reference_channel=0),
            "souden non-square 3d", "last two dimensions")


def check_rtf_low_dim_psd_n(solve):
    _raises(lambda: solve("rtf", rtf=RTF[:1], psd_n=N[0]),
            "rtf psd_n 2D", "at least 3D")


def check_rtf_non_square_psd_n(solve):
    # psd_n shape (F=1, C1=1, C2=3); rtf shape (F=1, C=1) matches
    # psd_n.shape[:-1] == [1,1], so the dedicated `sn[-1] != sn[-2]` guard
    # (last two dims 3 != 1) is the one that fires.
    ns = [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]
    _raises(lambda: solve("rtf", rtf=[[[1.0, 0.0]]], psd_n=ns),
            "rtf psd_n non-square", "last two dimensions")


def check_beamform_channel_mismatch(solve):
    _raises(lambda: solve("beamform", weights=W3, specgram=SG[:1]),
            "beamform channel mismatch", "should be the same as specgram")
